# dashapps/shooting_performance.py
from __future__ import annotations

import os
import ast
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.graph_objects as go

try:
    from .common import page_shell as PageShell
except Exception:
    PageShell = None

# =============================================================================
# File discovery
# =============================================================================
def _find_pbp_all() -> Optional[Path]:
    env = os.environ.get("PBP_ALL")
    if env and Path(env).exists():
        return Path(env)

    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "_cumulative" / "pbp_all.csv",
        data_root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # last resort: first pbp_all.csv anywhere under data_root
    for c in data_root.rglob("pbp_all.csv"):
        return c
    return None

# =============================================================================
# Helpers for schema variance
# =============================================================================
def _first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def _mmss_to_sec(s) -> float:
    try:
        txt = str(s).strip()
        if not txt or txt.lower() in {"nan", "none"}:
            return np.nan
        parts = txt.split(":")
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return np.nan
    return np.nan

def _safe_literal_list(v) -> List[str]:
    if pd.isna(v):
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip().lower() for x in v]
    s = str(v).strip()
    if s == "" or s == "[]":
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, (list, tuple, set)):
            return [str(x).strip().lower() for x in out]
    except Exception:
        pass
    sep = "," if "," in s else (";" if ";" in s else None)
    if sep:
        return [x.strip().lower() for x in s.split(sep) if x.strip()]
    return [s.lower()] if s else []

def _normalize_qualifiers(qs: List[str]) -> List[str]:
    out = []
    for q in qs:
        t = q.replace(" ", "").replace("-", "").lower()
        if t in {"fastbreak", "fastbreakpts", "fastbreakpoint", "fb"}:
            out.append("fastbreak")
        elif t in {"2ndchance", "secondchance", "offreb", "offensive_rebound_pts"}:
            out.append("2ndchance")
        elif t in {"pointsinthepaint", "pointsinthepaintpts", "pip", "paint"}:
            out.append("pointsinthepaint")
        else:
            out.append(t)
    return list(dict.fromkeys(out))  # dedupe, keep order

SHOT_GROUP_MAP = {
    "layup": {"layup", "drivinglayup", "reverselayup", "tipinlayup", "eurostep", "alleyoop", "alleyoopdunk"},
    "jumpshot": {"jumpshot", "pullupjumpshot", "stepbackjumpshot", "fadeaway", "floatingjumpshot", "turnaroundjumpshot"},
    "hookshot": {"hookshot", "babyhook"},
    "dunk": {"dunk", "alleyoopdunk", "windmill", "putbackdunk"},
    "tip-in": {"tipin", "tipinlayup"},
}
def _shot_group(subtype: str) -> str:
    st = str(subtype or "").strip().lower().replace(" ", "")
    for g, members in SHOT_GROUP_MAP.items():
        if st in members:
            return g
    if "jump" in st: return "jumpshot"
    if "layup" in st: return "layup"
    if "hook" in st: return "hookshot"
    if "dunk" in st: return "dunk"
    if "tip" in st:  return "tip-in"
    return st or "unknown"

def _canon_action(x: str) -> str:
    s = str(x or "").strip().lower().replace(" ", "").replace("-", "")
    if s in {"2pt","2p","p2","twopt","2pointer"}: return "2pt"
    if s in {"3pt","3p","p3","threept","3pointer","threepointer"}: return "3pt"
    return s

# ---- Week helpers ----
def _extract_week_num(val) -> float:
    if pd.isna(val): return np.nan
    s = str(val)
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return float(int(m.group(1)))
        except Exception:
            return np.nan
    return np.nan

def _fallback_week_map(series: pd.Series) -> Dict[str, int]:
    uniq = sorted(set([str(x) for x in series.fillna("").tolist() if str(x).strip()]))
    return {lab: i + 1 for i, lab in enumerate(uniq)}

def _week_marks(wmin: int, wmax: int) -> Dict[int, str]:
    if wmin > wmax:
        wmin, wmax = wmax, wmin
    span = wmax - wmin
    if span <= 12:
        return {i: str(i) for i in range(wmin, wmax + 1)}
    qs = [wmin, wmin + span // 4, wmin + span // 2, wmin + 3 * span // 4, wmax]
    return {int(q): str(int(q)) for q in sorted(set(qs))}

# =============================================================================
# Data loading / normalization (robust)
# =============================================================================
def load_data() -> pd.DataFrame:
    path = _find_pbp_all()
    if not path:
        return pd.DataFrame(columns=[
            "team_name","player","period","gt","lead","qualifier","subType","actionType",
            "points","success","week","week_num"
        ])

    df = pd.read_csv(path, low_memory=False)

    # Flexible column picks
    action_col  = _first_col(df, ["actionType", "action_type", "ac", "event_type", "event"])
    team_col    = _first_col(df, ["team_name", "team", "team_label", "teamName", "team_name_x"])
    player_col  = _first_col(df, ["player", "player_name", "shooter", "name"])
    period_col  = _first_col(df, ["period", "quarter", "qtr", "q"])
    clock_col   = _first_col(df, ["gt", "clock", "time", "time_remaining", "pcTime"])
    lead_col    = _first_col(df, ["lead", "score_margin", "margin"])
    qual_col    = _first_col(df, ["qualifier", "qualifiers", "tags", "play_tags"])
    subtype_col = _first_col(df, ["subType", "subtype", "shotSubType", "shot_type", "type_detail", "shotDescription"])
    success_col = _first_col(df, ["success", "made", "is_made", "result"])
    points_col  = _first_col(df, ["points", "pts"])
    week_col    = _first_col(df, ["week", "round", "matchweek", "gameweek", "gw", "wk"])

    if action_col is None:
        return pd.DataFrame(columns=[
            "team_name","player","period","gt","lead","qualifier","subType",
            "shot_subtype","shot_group","is_make","is_2","is_3","points","sec_left_in_period","lead_abs",
            "week","week_num"
        ])

    # Canonicalize action
    act = df[action_col].map(_canon_action)

    # Identify FG attempts
    is_fg = act.isin({"2pt", "3pt"})
    shots = df.loc[is_fg].copy()

    if shots.empty:
        sv_col = _first_col(df, ["shotValue", "shot_value", "value"])
        if sv_col is not None:
            m = df[sv_col].fillna(0).astype(float)
            shots = df.loc[m.isin([2.0, 3.0])].copy()
            act = pd.Series(np.where(df[sv_col].eq(3.0), "3pt",
                                     np.where(df[sv_col].eq(2.0), "2pt", "")),
                            index=df.index)
            act = act.loc[shots.index]
        else:
            return pd.DataFrame(columns=[
                "team_name","player","period","gt","lead","qualifier","subType",
                "shot_subtype","shot_group","is_make","is_2","is_3","points","sec_left_in_period","lead_abs",
                "week","week_num"
            ])

    # Team / Player
    if team_col is None:  raise ValueError("Expected a team column (e.g., 'team_name' or 'team').")
    if player_col is None: raise ValueError("Expected a player column (e.g., 'player' or 'player_name').")
    shots["team_name"] = shots[team_col].astype(str)
    shots["player"] = shots[player_col].astype(str)

    # Period
    shots["period"] = pd.to_numeric(shots[period_col], errors="coerce").fillna(0).astype(int) if period_col else 0

    # Clock -> seconds left in period
    shots["sec_left_in_period"] = shots[clock_col].apply(_mmss_to_sec) if clock_col else np.nan

    # Lead / abs lead
    if lead_col:
        shots["lead"] = pd.to_numeric(shots[lead_col], errors="coerce")
        shots["lead_abs"] = shots["lead"].abs()
    else:
        shots["lead_abs"] = np.nan

    # Make / Miss
    if success_col is None:
        if points_col is not None:
            shots["_tmp_pts"] = pd.to_numeric(shots[points_col], errors="coerce").fillna(0)
            shots["is_make"] = shots["_tmp_pts"] > 0
        else:
            shots["is_make"] = False
    else:
        s = shots[success_col]
        if s.dtype == bool:
            shots["is_make"] = s
        else:
            sv = s.astype(str).str.lower().str.strip()
            shots["is_make"] = sv.isin({"1","true","made","make","success","yes"})

    # 2pt / 3pt flags
    act_local = act.loc[shots.index] if not act.index.equals(shots.index) else act
    shots["is_3"] = act_local.eq("3pt")
    shots["is_2"] = act_local.eq("2pt")

    # Points
    if points_col is not None:
        shots["points"] = pd.to_numeric(shots[points_col], errors="coerce").fillna(0)
    else:
        shots["points"] = (shots["is_2"].astype(int) * 2 + shots["is_3"].astype(int) * 3) * shots["is_make"].astype(int)

    # Qualifiers -> standardized flags
    if qual_col is not None:
        quals = shots[qual_col].apply(_safe_literal_list).apply(_normalize_qualifiers)
        shots["q_has_fastbreak"] = quals.apply(lambda qs: "fastbreak" in qs)
        shots["q_has_2ndchance"] = quals.apply(lambda qs: "2ndchance" in qs)
        shots["q_has_paint"]     = quals.apply(lambda qs: "pointsinthepaint" in qs)
    else:
        shots["q_has_fastbreak"] = shots["q_has_2ndchance"] = shots["q_has_paint"] = False

    # Shot subtype / group
    shots["shot_subtype"] = shots[subtype_col].astype(str) if subtype_col else ""
    shots["shot_group"] = shots["shot_subtype"].apply(_shot_group)

    # --- Week & week_num ---
    shots["week"] = shots[week_col] if week_col else np.nan
    shots["week_num"] = shots["week"].apply(_extract_week_num)
    if shots["week_num"].isna().all():
        mapping = _fallback_week_map(shots["week"])
        shots["week_num"] = shots["week"].map(mapping).astype(float)
    shots["week_num"] = pd.to_numeric(shots["week_num"], errors="coerce")

    return shots

# ------------------ Load base once ------------------
_BASE = load_data()

def _opt(series: pd.Series) -> List[Dict[str, str]]:
    vals = [v for v in series.dropna().astype(str).unique() if v.strip()]
    vals.sort()
    return [{"label": v, "value": v} for v in vals]

TEAM_OPTIONS   = _opt(_BASE["team_name"]) if "team_name" in _BASE else []
PLAYER_OPTIONS = _opt(_BASE["player"]) if "player" in _BASE else []
SHOTTYPE_OPTIONS = _opt(_BASE["shot_subtype"]) if "shot_subtype" in _BASE else []
SHOTGROUP_OPTIONS = _opt(_BASE["shot_group"]) if "shot_group" in _BASE else []

# Week bounds for slider
if "week_num" in _BASE.columns and not _BASE["week_num"].dropna().empty:
    _WEEK_MIN = int(np.nanmin(_BASE["week_num"]))
    _WEEK_MAX = int(np.nanmax(_BASE["week_num"]))
else:
    _WEEK_MIN = _WEEK_MAX = 1

# =============================================================================
# Scenario logic
# =============================================================================
SCENARIOS = [
    ("all", "All Plays"),
    ("clutch_l5_m5", "Clutch (4Q: last 5:00 & margin ≤ 5)"),
    ("close_m5", "Close Game (margin ≤ M)"),
    ("lastn_4q", "Last N min of 4Q"),
    ("last2_each_q", "Last 2:00 of each quarter"),
    ("first_half", "First Half (Q1–Q2)"),
    ("second_half", "Second Half (Q3–Q4)"),
    ("q1", "Q1 only"),
    ("q2", "Q2 only"),
    ("q3", "Q3 only"),
    ("q4", "Q4 only"),
]

QUAL_OPTIONS = [
    {"label": "Fast Break", "value": "fastbreak"},
    {"label": "Second Chance", "value": "2ndchance"},
    {"label": "Points in the Paint", "value": "pointsinthepaint"},
]

def scenario_mask(df: pd.DataFrame, scenario: str, lastn_sec: int = 300, margin: int = 5) -> pd.Series:
    sec = df.get("sec_left_in_period", pd.Series(np.nan, index=df.index))
    per = df.get("period", pd.Series(0, index=df.index)).astype(int)
    lead_abs = df.get("lead_abs", pd.Series(np.nan, index=df.index))

    if scenario == "all":            return pd.Series(True, index=df.index)
    if scenario == "clutch_l5_m5":   return (per == 4) & (sec <= 300) & (lead_abs <= 5)
    if scenario == "close_m5":       return (lead_abs <= margin)
    if scenario == "lastn_4q":       return (per == 4) & (sec <= lastn_sec)
    if scenario == "last2_each_q":   return (sec <= 120)
    if scenario == "first_half":     return per.isin([1, 2])
    if scenario == "second_half":    return per.isin([3, 4])
    if scenario in {"q1","q2","q3","q4"}:
        q = int(scenario[-1]);       return per == q
    return pd.Series(True, index=df.index)

# =============================================================================
# Aggregation
# =============================================================================
def compute_summary(df: pd.DataFrame, groupby: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            groupby, "FGA", "FGM", "FG%", "3PA", "3PM", "3P%", "2PA", "2PM", "2P%", "eFG%", "PTS",
            "PIP Pts", "2ndCh Pts", "FB Pts", "2P_share_of_PTS", "3P_share_of_PTS"
        ])

    for c, default in [
        ("is_make", False), ("is_2", False), ("is_3", False),
        ("q_has_paint", False), ("q_has_2ndchance", False), ("q_has_fastbreak", False),
    ]:
        if c not in df.columns:
            df[c] = default
    if "points" not in df.columns:
        df["points"] = 0

    g = df.groupby(groupby, dropna=False)
    agg = g.agg(
        FGA=("is_make", "size"),
        FGM=("is_make", "sum"),
        PTS=("points", "sum"),
        **{
            "3PA": ("is_3", "sum"),
            "3PM": ("is_make", lambda s: ((df.loc[s.index, "is_3"]) & (df.loc[s.index, "is_make"])).sum()),
            "2PA": ("is_2", "sum"),
            "2PM": ("is_make", lambda s: ((df.loc[s.index, "is_2"]) & (df.loc[s.index, "is_make"])).sum()),
            "PIP Pts": ("points", lambda s: df.loc[s.index].pipe(
                lambda d: d.loc[d["q_has_paint"] & d["is_make"], "points"].sum()
            )),
            "2ndCh Pts": ("points", lambda s: df.loc[s.index].pipe(
                lambda d: d.loc[d["q_has_2ndchance"] & d["is_make"], "points"].sum()
            )),
            "FB Pts": ("points", lambda s: df.loc[s.index].pipe(
                lambda d: d.loc[d["q_has_fastbreak"] & d["is_make"], "points"].sum()
            )),
        }
    ).reset_index()

    with np.errstate(divide="ignore", invalid="ignore"):
        agg["FG%"]  = (agg["FGM"] / agg["FGA"]).replace([np.inf, -np.inf], np.nan)
        agg["3P%"]  = (agg["3PM"] / agg["3PA"]).replace([np.inf, -np.inf], np.nan)
        agg["2P%"]  = (agg["2PM"] / agg["2PA"]).replace([np.inf, -np.inf], np.nan)
        agg["eFG%"] = ((agg["FGM"] + 0.5 * agg["3PM"]) / agg["FGA"]).replace([np.inf, -np.inf], np.nan)

        p2_pts = 2.0 * agg["2PM"]
        p3_pts = 3.0 * agg["3PM"]
        total_pts = agg["PTS"].replace(0, np.nan)
        agg["2P_share_of_PTS"] = (p2_pts / total_pts).replace([np.inf, -np.inf], np.nan)
        agg["3P_share_of_PTS"] = (p3_pts / total_pts).replace([np.inf, -np.inf], np.nan)

    agg = agg.sort_values(["PTS", "FGM", "FGA"], ascending=[False, False, True])

    pct_cols = ["FG%", "3P%", "2P%", "eFG%", "2P_share_of_PTS", "3P_share_of_PTS"]
    agg[pct_cols] = (agg[pct_cols] * 100).round(1)

    for c in ["FGA", "FGM", "3PA", "3PM", "2PA", "2PM", "PTS", "PIP Pts", "2ndCh Pts", "FB Pts"]:
        agg[c] = agg[c].fillna(0).astype(int)

    return agg

def subtype_breakdown(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["shot_subtype", "FGA", "FGM", "FG%", "PTS"])
    g = df.groupby("shot_subtype", dropna=False).agg(
        FGA=("is_make", "size"),
        FGM=("is_make", "sum"),
        PTS=("points", "sum")
    ).reset_index()
    g["FG%"] = np.where(g["FGA"] > 0, 100.0 * g["FGM"] / g["FGA"], np.nan)
    g = g.sort_values(["FGA", "PTS"], ascending=[False, False]).head(top_n)
    return g

def group_share_stacked(df: pd.DataFrame, groupby: str, top_k: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[groupby, "2P_pts", "3P_pts", "Total"])
    g = df.groupby(groupby, dropna=False).agg(
        P2=("is_make", lambda s: ((df.loc[s.index, "is_2"]) & (df.loc[s.index, "is_make"])).sum()),
        P3=("is_make", lambda s: ((df.loc[s.index, "is_3"]) & (df.loc[s.index, "is_make"])).sum()),
    ).reset_index()
    g["2P_pts"] = 2 * g["P2"]
    g["3P_pts"] = 3 * g["P3"]
    g["Total"] = g["2P_pts"] + g["3P_pts"]
    g = g.sort_values("Total", ascending=False).head(top_k)
    return g[[groupby, "2P_pts", "3P_pts", "Total"]]

# =============================================================================
# Small UI helpers
# =============================================================================
def _kpi(label: str, value) -> html.Div:
    return html.Div([
        html.Div(str(label), className="kpi-label", style={"fontSize":"11px","color":"#667"}),
        html.Div(str(value), className="kpi-value", style={"fontSize":"16px","fontWeight":700,"marginTop":"2px"}),
    ], className="kpi", style={"padding":"6px 8px","border":"1px solid #eee","borderRadius":"8px"})

# =============================================================================
# App factory
# =============================================================================
def create_dash_shooting_performance(server, base_pathname: str = "/shooting_performance/") -> Dash:
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
        title="Shooting Performance",
    )

    scenario_options = [{"label": label, "value": key} for key, label in SCENARIOS]

    # ---- Dense control bar with Week Range slider ----
    controls = html.Div([
        html.Div([
            html.Label("Mode / Entity"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"6px"}, children=[
                dcc.Dropdown(
                    id="sp-mode",
                    options=[{"label": "Team", "value": "team"}, {"label": "Player", "value": "player"}],
                    value="team", clearable=False
                ),
                dcc.Dropdown(id="sp-scenario", options=scenario_options, value="all", clearable=False),
            ])
        ], className="c"),

        html.Div([
            html.Label("Team(s) / Player(s)"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"6px"}, children=[
                dcc.Dropdown(id="sp-teams", options=TEAM_OPTIONS, multi=True, placeholder="All teams"),
                dcc.Dropdown(id="sp-players", options=PLAYER_OPTIONS, multi=True, placeholder="All players"),
            ])
        ], className="c"),

        html.Div([
            html.Label("Period / Week Range"),
            html.Div(style={"display":"grid","gridTemplateColumns":"minmax(180px,1fr) 1fr","gap":"6px"}, children=[
                dcc.RangeSlider(id="sp-periods", min=1, max=4, step=1, value=[1,4], marks={i:str(i) for i in range(1,5)}),
                dcc.RangeSlider(
                    id="sp-weeks", min=_WEEK_MIN, max=_WEEK_MAX, step=1, value=[_WEEK_MIN, _WEEK_MAX],
                    marks=_week_marks(_WEEK_MIN, _WEEK_MAX)
                ),
            ])
        ], className="c"),

        html.Div([
            html.Label("Scenario Params"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"6px","alignItems":"center"}, children=[
                html.Div(children=[html.Span("Last N (4Q)", style={"marginRight":"6px"}), dcc.Input(id="sp-lastn", type="number", value=5, min=1, step=1, style={"width":"90px"})]),
                html.Div(children=[html.Span("Margin M", style={"marginRight":"6px"}), dcc.Input(id="sp-margin", type="number", value=5, min=1, step=1, style={"width":"90px"})]),
            ])
        ], className="c"),

        html.Div([
            html.Label("Shot Subtypes / Grouping"),
            html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr","gap":"6px"}, children=[
                dcc.Dropdown(id="sp-shottypes", options=SHOTTYPE_OPTIONS, multi=True, placeholder="All subtypes"),
                dcc.Checklist(id="sp-grouping", options=[{"label":" Group similar types","value":"group"}], value=["group"]),
            ])
        ], className="c"),

        html.Div([
            html.Label("Qualifiers / Options"),
            html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr 1fr","gap":"6px"}, children=[
                dcc.Checklist(id="sp-quals", options=QUAL_OPTIONS, value=[], inline=True),
                dcc.Checklist(id="sp-compact", options=[{"label":" Compact table","value":"on"}], value=[]),
                dcc.Checklist(id="sp-include-unknown-week", options=[{"label":" Include unknown weeks","value":"on"}], value=["on"]),
            ])
        ], className="c"),
    ], className="controls", style={
        "display":"grid",
        "gridTemplateColumns":"repeat(3, minmax(260px,1fr))",
        "gap":"10px"
    })

    # KPI strip (mini cards)
    kpis = html.Div(id="sp-kpis", style={"display":"grid","gridTemplateColumns":"repeat(10, minmax(90px, 1fr))","gap":"6px","margin":"8px 0"})

    # Table
    table = dash_table.DataTable(
        id="sp-table",
        columns=[
            {"name": "Group", "id": "group"},
            {"name": "FGA", "id": "FGA", "type": "numeric"},
            {"name": "FGM", "id": "FGM", "type": "numeric"},
            {"name": "FG%", "id": "FG%", "type": "numeric"},
            {"name": "3PA", "id": "3PA", "type": "numeric"},
            {"name": "3PM", "id": "3PM", "type": "numeric"},
            {"name": "3P%", "id": "3P%", "type": "numeric"},
            {"name": "2PA", "id": "2PA", "type": "numeric"},
            {"name": "2PM", "id": "2PM", "type": "numeric"},
            {"name": "2P%", "id": "2P%", "type": "numeric"},
            {"name": "eFG%", "id": "eFG%", "type": "numeric"},
            {"name": "PTS", "id": "PTS", "type": "numeric"},
            {"name": "PIP Pts", "id": "PIP Pts", "type": "numeric"},
            {"name": "2ndCh Pts", "id": "2ndCh Pts", "type": "numeric"},
            {"name": "FB Pts", "id": "FB Pts", "type": "numeric"},
            {"name": "2P Share (%)", "id": "2P_share_of_PTS", "type": "numeric"},
            {"name": "3P Share (%)", "id": "3P_share_of_PTS", "type": "numeric"},
        ],
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px 8px", "minWidth": 70},
        style_header={"fontWeight": "600"},
    )

    # Graphs
    subtypes_graph = dcc.Graph(id="sp-subtypes-graph")
    share_graph = dcc.Graph(id="sp-share-graph")

    # Top-N sliders for graphs (space-friendly, inline)
    graph_opts = html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"10px","margin":"6px 0"}, children=[
        html.Div(children=[
            html.Label("Top N Subtypes"),
            dcc.Slider(id="sp-topn-subtypes", min=6, max=20, value=12, step=1, marks={6:"6",12:"12",16:"16",20:"20"})
        ]),
        html.Div(children=[
            html.Label("Top K Groups (2P vs 3P share)"),
            dcc.Slider(id="sp-topk-groups", min=5, max=20, value=10, step=1, marks={5:"5",10:"10",15:"15",20:"20"})
        ])
    ])

    content = html.Div([
        controls,
        html.Hr(style={"margin":"10px 0"}),
        kpis,
        html.Div([
            html.Div([html.H4("Field Goal Summary", style={"margin":"6px 0"}), table], style={"flex":"1 1 60%","minWidth":"420px"}),
            html.Div([html.H4("2P vs 3P Points Share", style={"margin":"6px 0"}), graph_opts, share_graph], style={"flex":"1 1 40%","minWidth":"360px"}),
        ], style={"display":"flex","gap":"18px","flexWrap":"wrap"}),
        html.Hr(style={"margin":"10px 0"}),
        html.H4("Shot Subtype Breakdown", style={"margin":"6px 0"}),
        subtypes_graph,
    ], className="page-content", style={"padding":"8px 0"})

    if PageShell:
        app.layout = PageShell(content, nav_items=server.config["NAV"], current_endpoint=base_pathname)
    else:
        app.layout = html.Div([html.H2("Shooting Performance"), content])

    # -----------------------
    # Show/hide team vs player pickers (just style; both exist in dense bar)
    # -----------------------
    @app.callback(
        Output("sp-teams", "style"),
        Output("sp-players", "style"),
        Input("sp-mode", "value"),
    )
    def _toggle_entity_inputs(mode: str):
        if mode == "team":
            return ({}, {"display":"none"})
        return ({"display":"none"}, {})

    # -----------------------
    # Main compute
    # -----------------------
    @app.callback(
        Output("sp-kpis", "children"),
        Output("sp-table", "data"),
        Output("sp-subtypes-graph", "figure"),
        Output("sp-share-graph", "figure"),
        Input("sp-mode", "value"),
        Input("sp-teams", "value"),
        Input("sp-players", "value"),
        Input("sp-scenario", "value"),
        Input("sp-lastn", "value"),
        Input("sp-margin", "value"),
        Input("sp-shottypes", "value"),
        Input("sp-grouping", "value"),
        Input("sp-quals", "value"),
        Input("sp-periods", "value"),
        Input("sp-weeks", "value"),
        Input("sp-compact", "value"),
        Input("sp-include-unknown-week", "value"),
        Input("sp-topn-subtypes", "value"),
        Input("sp-topk-groups", "value"),
    )
    def _run(
        mode: str,
        teams: List[str],
        players: List[str],
        scenario: str,
        lastn_min: int,
        margin: int,
        shottypes: List[str],
        grouping_flags: List[str],
        quals: List[str],
        periods: List[int],
        week_range: List[int],
        compact: List[str],
        include_unknown_week: List[str],
        topn_subtypes: int,
        topk_groups: int,
    ):
        df = _BASE

        if df.empty:
            msg = html.Div("No shot attempts found in pbp_all.csv (after normalization).", style={"color": "#a00"})
            return msg, [], go.Figure(), go.Figure()

        # Scenario filter
        lastn_sec = max(1, int(lastn_min or 5)) * 60
        margin_int = max(1, int(margin or 5))
        m = scenario_mask(df, scenario, lastn_sec=lastn_sec, margin=margin_int)
        df2 = df[m].copy()

        # Periods filter
        if periods and len(periods) == 2:
            p_lo, p_hi = int(periods[0]), int(periods[1])
            per = pd.to_numeric(df2.get("period", pd.Series(np.nan, index=df2.index)), errors="coerce")
            df2 = df2[(per >= p_lo) & (per <= p_hi)]

        # Weeks filter
        if week_range and "week_num" in df2.columns:
            w_lo, w_hi = int(week_range[0]), int(week_range[1])
            wnum = pd.to_numeric(df2["week_num"], errors="coerce")
            mask_w = (wnum >= w_lo) & (wnum <= w_hi)
            if "on" in (include_unknown_week or []):
                df2 = df2[mask_w | wnum.isna()]
            else:
                df2 = df2[mask_w]

        # Mode filter (team or player)
        if mode == "team" and teams:
            df2 = df2[df2["team_name"].isin(teams)]
        if mode == "player" and players:
            df2 = df2[df2["player"].isin(players)]

        # Shot subtype filter
        if shottypes:
            df2 = df2[df2["shot_subtype"].isin(shottypes)]

        # Qualifiers (require at least one of selected)
        if quals:
            flags_map = {"fastbreak":"q_has_fastbreak","2ndchance":"q_has_2ndchance","pointsinthepaint":"q_has_paint"}
            mask = pd.Series(False, index=df2.index)
            for q in quals:
                col = flags_map.get(q)
                if col in df2.columns:
                    mask = mask | df2[col]
            df2 = df2[mask]

        # Optionally group subtypes
        group_subtypes = "group" in (grouping_flags or [])
        if group_subtypes:
            df2["shot_subtype"] = df2["shot_group"]

        # Aggregation by entity
        groupby = "team_name" if mode == "team" else "player"
        summary = compute_summary(df2, groupby=groupby)

        # KPI strip (global over current filter)
        overall = compute_summary(df2.assign(dummy=1), groupby="dummy")
        if overall.empty:
            kpi_children = [html.Div("No shots match the current filters.", style={"color":"#a00"})]
        else:
            row = overall.iloc[0].to_dict()
            def fmt_pct(x): return f"{x:.1f}"
            def fmt_int(x): return f"{int(x):,}"
            kpi_children = [
                _kpi("FGA", fmt_int(row.get("FGA", 0))),
                _kpi("FGM", fmt_int(row.get("FGM", 0))),
                _kpi("FG%", fmt_pct(row.get("FG%", 0))),
                _kpi("eFG%", fmt_pct(row.get("eFG%", 0))),
                _kpi("PTS", fmt_int(row.get("PTS", 0))),
                _kpi("2P Share", fmt_pct(row.get("2P_share_of_PTS", 0)) + "%"),
                _kpi("3P Share", fmt_pct(row.get("3P_share_of_PTS", 0)) + "%"),
                _kpi("PIP", fmt_int(row.get("PIP Pts", 0))),
                _kpi("2ndCh", fmt_int(row.get("2ndCh Pts", 0))),
                _kpi("FB", fmt_int(row.get("FB Pts", 0))),
            ]

        # Prepare table
        table_data = []
        if not summary.empty:
            td = summary.copy()
            td["2P_share_of_PTS"] = td["2P_share_of_PTS"].round(1)
            td["3P_share_of_PTS"] = td["3P_share_of_PTS"].round(1)
            td.insert(0, "group", td[groupby])
            table_data = td.to_dict("records")

        # Apply compact mode (lighter padding)
        compact_on = "on" in (compact or [])
        if compact_on:
            table.style_cell = {"padding":"4px 6px","minWidth":60}
            table.style_header = {"fontWeight":"600","padding":"4px 6px"}
        else:
            table.style_cell = {"padding":"6px 8px","minWidth":70}
            table.style_header = {"fontWeight":"600"}

        # Subtype graph
        sub_df = subtype_breakdown(df2, top_n=int(topn_subtypes or 12))
        fig_sub = go.Figure()
        if not sub_df.empty:
            fig_sub.add_bar(
                x=sub_df["shot_subtype"], y=sub_df["FG%"],
                hovertext=[f"FGA {a}, FGM {m}, PTS {p}" for a, m, p in zip(sub_df["FGA"], sub_df["FGM"], sub_df["PTS"])],
                name="FG% by subtype",
            )
            fig_sub.update_layout(xaxis_title="Shot subtype", yaxis_title="FG% (Top by attempts)",
                                  margin=dict(l=40, r=10, t=10, b=60))

        # 2P vs 3P stacked
        share_df = group_share_stacked(df2, groupby=groupby, top_k=int(topk_groups or 10))
        fig_share = go.Figure()
        if not share_df.empty:
            fig_share.add_bar(x=share_df[groupby], y=share_df["2P_pts"], name="2P pts")
            fig_share.add_bar(x=share_df[groupby], y=share_df["3P_pts"], name="3P pts")
            fig_share.update_layout(barmode="stack", xaxis_title=("Team" if mode == "team" else "Player"),
                                    yaxis_title="Points", margin=dict(l=40, r=10, t=10, b=80))

        return kpi_children, table_data, fig_sub, fig_share

    return app
