# dashapps/scoring_leaders.py
from __future__ import annotations

import os
import re
import ast
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict, Set

import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go

# (optional) silence pandas/plotly length-1 grouping FutureWarning locally
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"When grouping with a length-1 list-like.*get_group",
    category=FutureWarning,
    module="plotly"
)

from .common import page_shell as PageShell  # shared sidebar shell

# =============================================================================
# File discovery
# =============================================================================
def _find_csv(env_key: str, default_candidates: Iterable[Path]) -> Optional[Path]:
    env = os.environ.get(env_key)
    if env and Path(env).exists():
        return Path(env)
    for c in default_candidates:
        if c and c.exists():
            return c
    return None

def _find_pbp_all() -> Optional[Path]:
    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "_cumulative" / "pbp_all.csv",
        data_root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),
    ]
    p = _find_csv("PBP_ALL", candidates)
    if p:
        return p
    for c in data_root.rglob("pbp_all.csv"):
        return c
    return None

def _find_players_all() -> Optional[Path]:
    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "_cumulative" / "players_all.csv",
        data_root / "players_all.csv",
        Path("/mnt/data/players_all.csv"),
    ]
    p = _find_csv("PLAYERS_ALL", candidates)
    if p:
        return p
    for c in data_root.rglob("players_all.csv"):
        return c
    return None

# =============================================================================
# Team aliasing / canonicalization
# =============================================================================
def _norm_team_key(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_TEAM_ALIASES_DEFAULT: Dict[str, str] = {
    _norm_team_key("GLİNT MANİSA BASKET"): "MANISA BASKET",
    _norm_team_key("GLINT MANISA BASKET"): "MANISA BASKET",
    _norm_team_key("GLINT MANISA"): "MANISA BASKET",
    _norm_team_key("MANISA BASKET"): "MANISA BASKET",
    _norm_team_key("MANISA BBSK"): "MANISA BASKET",

    _norm_team_key("FRUTTI EXTRA BURSASPOR"): "BURSASPOR",
    _norm_team_key("BURSASPOR"): "BURSASPOR",

    _norm_team_key("ALIAGA PETKIMSPOR"): "PETKIMSPOR",
    _norm_team_key("PETKIM SPOR"): "PETKIMSPOR",
    _norm_team_key("PETKIMSPOR"): "PETKIMSPOR",

    _norm_team_key("TÜRK TELEKOM"): "TÜRK TELEKOM",
    _norm_team_key("TURK TELEKOM"): "TÜRK TELEKOM",
}
_TEAM_ALIAS_MAP: Optional[Dict[str, str]] = None

def _load_team_aliases_from_csv() -> Dict[str, str]:
    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates: List[Path] = []
    envp = os.environ.get("TEAM_ALIASES")
    if envp:
        p = Path(envp)
        if p.exists():
            candidates.append(p)
    for c in [data_root / "_config" / "team_aliases.csv",
              data_root / "team_aliases.csv",
              Path("/mnt/data/team_aliases.csv")]:
        if c.exists():
            candidates.append(c)

    alias_map: Dict[str, str] = {}
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "alias" not in cols or "canonical" not in cols:
            continue
        for alias, canon in zip(df[cols["alias"]], df[cols["canonical"]]):
            if pd.isna(alias) or pd.isna(canon):
                continue
            alias_key = _norm_team_key(str(alias))
            canon_val = str(canon).strip()
            if alias_key:
                alias_map[alias_key] = canon_val
    return alias_map

def _team_alias_map() -> Dict[str, str]:
    global _TEAM_ALIAS_MAP
    if _TEAM_ALIAS_MAP is not None:
        return _TEAM_ALIAS_MAP
    m: Dict[str, str] = dict(_TEAM_ALIASES_DEFAULT)
    try:
        from_csv = _load_team_aliases_from_csv()
        m.update(from_csv)
    except Exception:
        pass
    _TEAM_ALIAS_MAP = m
    return _TEAM_ALIAS_MAP

def _canon_team_name(raw: Optional[str]) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return raw
    s = str(raw).strip()
    key = _norm_team_key(s)
    canon = _team_alias_map().get(key)
    return canon if canon else s

# =============================================================================
# Utilities
# =============================================================================
def _ensure(df: pd.DataFrame, cols: List[str], fill=None) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def _lc(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower()

def _as_num(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- Qualifier parsing ----
QUAL_TOKENS = {
    "paint": {"pointsinthepaint"},
    "fastbreak": {"fastbreak"},
    "second_chance": {"2ndchance"},
    "off_turnover": {"fromturnover"},
}

def _parse_quali_cell(val) -> Set[str]:
    if pd.isna(val):
        return set()
    txt = str(val).strip()
    if not txt:
        return set()
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, (list, tuple)):
            return {str(x).strip().lower() for x in obj if str(x).strip()}
        return {str(obj).strip().lower()}
    except Exception:
        toks = [t for t in re.split(r"[^a-zA-Z0-9]+", txt.lower()) if t]
        return set(toks)

def _expand_qualifiers(df: pd.DataFrame) -> pd.DataFrame:
    if "qualifier" not in df.columns:
        for k in ["q_paint","q_fastbreak","q_second_chance","q_off_turnover"]:
            df[k] = False
        return df
    quals = df["qualifier"].apply(_parse_quali_cell)
    df["q_paint"]          = quals.apply(lambda s: len(QUAL_TOKENS["paint"] & s) > 0)
    df["q_fastbreak"]      = quals.apply(lambda s: len(QUAL_TOKENS["fastbreak"] & s) > 0)
    df["q_second_chance"]  = quals.apply(lambda s: len(QUAL_TOKENS["second_chance"] & s) > 0)
    df["q_off_turnover"]   = quals.apply(lambda s: len(QUAL_TOKENS["off_turnover"] & s) > 0)
    return df

# =============================================================================
# Week helpers
# =============================================================================
def _extract_week_num(val) -> float:
    """Extract integer digits from labels like 'week01', 'W2', '2024W03'. NaN if none."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return float(int(m.group(1)))
        except Exception:
            return np.nan
    return np.nan

def _fallback_week_map(series: pd.Series) -> Dict[str, int]:
    """If no numeric weeks, map sorted unique labels to 1..N."""
    uniq = sorted(set([str(x) for x in series.fillna("").tolist() if str(x).strip()]))
    return {lab: i + 1 for i, lab in enumerate(uniq)}

def _week_marks(wmin: int, wmax: int) -> Dict[int, str]:
    """Readable marks for the slider: dense for small ranges, sparse for large."""
    if wmin > wmax:
        wmin, wmax = wmax, wmin
    span = wmax - wmin
    if span <= 12:
        return {i: str(i) for i in range(wmin, wmax + 1)}
    qs = [wmin, wmin + span // 4, wmin + span // 2, wmin + 3 * span // 4, wmax]
    return {int(q): str(int(q)) for q in sorted(set(qs))}

# =============================================================================
# Hot-reload (mtime watch)
# =============================================================================
_DF_PBP: Optional[pd.DataFrame] = None
_DF_PLAYERS: Optional[pd.DataFrame] = None

_PBP_PATH: Optional[Path] = None
_PLAYERS_PATH: Optional[Path] = None
_PBP_MTIME: Optional[float] = None
_PLAYERS_MTIME: Optional[float] = None

def _mtime(p: Optional[Path]) -> Optional[float]:
    try:
        return p.stat().st_mtime if p and p.exists() else None
    except Exception:
        return None

# =============================================================================
# Loaders & canonicalization (with path/mtime bookkeeping)
# =============================================================================
def _load_pbp() -> pd.DataFrame:
    global _PBP_PATH, _PBP_MTIME
    src = _find_pbp_all()
    if not src or not src.exists():
        raise FileNotFoundError("pbp_all.csv not found. Set PBP_ALL or place it under out_data/ or /mnt/data/")
    _PBP_PATH, _PBP_MTIME = src, _mtime(src)
    df = pd.read_csv(src)

    print(f"[scoring_leaders] Loaded {src} "
          f"(rows={len(df):,}, games={df.get('game_id', pd.Series()).nunique()})")

    _ensure(df, ["week","game_id","label"], np.nan)
    _ensure(df, ["team_name","tno","pno","player","player_name_from_roster"], np.nan)
    _ensure(df, ["ev_code","ev_result","subType","actionNumber","period","points"], np.nan)
    _ensure(df, ["possession_id","possession_owner_tno"], np.nan)

    _as_num(df, ["actionNumber","period","tno","pno","points","possession_id","possession_owner_tno"])

    df["ev_code"] = _lc(df["ev_code"])
    df["ev_result"] = _lc(df["ev_result"])
    df["subType"] = _lc(df["subType"])

    # canonicalize team names
    if "team_name" in df.columns:
        df["team_name_raw"] = df["team_name"]
        df["team_name"] = df["team_name"].apply(_canon_team_name)

    # derive week_num for range slider
    df["week_num"] = df["week"].apply(_extract_week_num)
    if df["week_num"].isna().all():
        mapping = _fallback_week_map(df["week"])
        df["week_num"] = df["week"].map(mapping).astype(float)
    df["week_num"] = pd.to_numeric(df["week_num"], errors="coerce")

    # scorer
    df["scorer"] = df["player"].fillna(df["player_name_from_roster"]).fillna("Unknown Player")

    # shot/make flags
    df["is_fg_att"]  = df["ev_code"].isin(["p2","p3"])
    df["is_made_fg"] = df["is_fg_att"] & (df["ev_result"] == "made")
    df["is_p2_att"]  = (df["ev_code"] == "p2")
    df["is_p3_att"]  = (df["ev_code"] == "p3")
    df["is_p2_made"] = df["is_p2_att"] & (df["ev_result"] == "made")
    df["is_p3_made"] = df["is_p3_att"] & (df["ev_result"] == "made")
    df["is_ft_made"] = (df["ev_code"] == "ft") & (df["ev_result"] == "made")

    # points
    if "points" not in df.columns or df["points"].isna().all():
        df["points"] = np.select(
            [df["is_p3_made"], df["is_p2_made"], df["is_ft_made"]], [3, 2, 1], default=0
        )
    df["points"] = df["points"].fillna(0).astype(float)

    # expand qualifiers
    df = _expand_qualifiers(df)

    # lenses
    df["is_paint_fg"]         = df["is_made_fg"] & df["q_paint"]
    df["is_fastbreak_fg"]     = df["is_made_fg"] & df["q_fastbreak"]
    df["is_second_chance_fg"] = df["is_made_fg"] & df["q_second_chance"]
    df["is_pot_point"]        = df["q_off_turnover"] & (df["points"] > 0)

    # points by lens (overlaps allowed)
    df["paint_pts"] = np.where(df["is_paint_fg"], df["points"], 0.0)
    df["fb_pts"]    = np.where(df["is_fastbreak_fg"], df["points"], 0.0)
    df["sc_pts"]    = np.where(df["is_second_chance_fg"], df["points"], 0.0)
    df["pot_pts"]   = np.where(df["is_pot_point"], df["points"], 0.0)
    df["p2_pts"]    = np.where(df["is_p2_made"], df["points"], 0.0)
    df["p3_pts"]    = np.where(df["is_p3_made"], df["points"], 0.0)
    df["ft_pts"]    = np.where(df["is_ft_made"], df["points"], 0.0)

    df["total_pts"] = df["points"].astype(float)

    # convenient attempt/made ints per row
    df["fga"]  = df["is_fg_att"].astype(int)
    df["fgm"]  = df["is_made_fg"].astype(int)
    df["p2a"]  = df["is_p2_att"].astype(int)
    df["p3a"]  = df["is_p3_att"].astype(int)
    df["p3m"]  = df["is_p3_made"].astype(int)

    df["shot_type"] = np.where(
        df["is_p3_made"], "3PT",
        np.where(df["is_p2_made"], "2PT", np.where(df["is_ft_made"], "FT", None))
    )
    return df

def _load_players() -> Optional[pd.DataFrame]:
    global _PLAYERS_PATH, _PLAYERS_MTIME
    src = _find_players_all()
    if not src or not src.exists():
        _PLAYERS_PATH, _PLAYERS_MTIME = None, None
        return None
    _PLAYERS_PATH, _PLAYERS_MTIME = src, _mtime(src)
    pl = pd.read_csv(src)
    print(f"[scoring_leaders] Loaded {src} (rows={len(pl):,})")
    if "team_name" in pl.columns:
        pl["team_name_raw"] = pl["team_name"]
        pl["team_name"] = pl["team_name"].apply(_canon_team_name)
    return pl

def _maybe_reload() -> None:
    """Reload dataframes if file path or mtime changed."""
    global _DF_PBP, _DF_PLAYERS, _PBP_PATH, _PLAYERS_PATH, _PBP_MTIME, _PLAYERS_MTIME
    cur_pbp, cur_pl = _find_pbp_all(), _find_players_all()
    if (
        _DF_PBP is None
        or cur_pbp != _PBP_PATH
        or _mtime(cur_pbp) != _PBP_MTIME
        or (_PLAYERS_PATH is not None and _mtime(cur_pl) != _PLAYERS_MTIME)
    ):
        _DF_PBP = _load_pbp()
        _DF_PLAYERS = _load_players()

def _pbp() -> pd.DataFrame:
    _maybe_reload()
    return _DF_PBP

# =============================================================================
# Opponent mapping
# =============================================================================
def _attach_opp(df: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent team name per (game_id, team_name) under a two-team assumption."""
    if "game_id" not in df.columns or "team_name" not in df.columns:
        df["opp_team_name"] = np.nan
        return df
    pairs = df[["game_id","team_name"]].dropna().drop_duplicates()
    opp_map: Dict[Tuple[object, str], str] = {}
    for gid, grp in pairs.groupby("game_id", dropna=False):
        uniq = [t for t in grp["team_name"].unique() if pd.notna(t)]
        if len(uniq) == 2:
            a, b = uniq[0], uniq[1]
            opp_map[(gid, a)] = b
            opp_map[(gid, b)] = a
    df = df.copy()
    df["opp_team_name"] = df.apply(lambda r: opp_map.get((r["game_id"], r["team_name"])), axis=1)
    return df

# =============================================================================
# Filtering
# =============================================================================
def _filtered(df: pd.DataFrame,
              games: List[str],
              periods: Tuple[int, int],
              weeks: Tuple[int, int]) -> pd.DataFrame:
    """Filter by games, periods, and week range (via week_num). Team filter happens after agg."""
    f = df.copy()
    if games:
        f = f[f["game_id"].astype(str).isin(games)]
    if periods:
        lo, hi = periods
        per = pd.to_numeric(f["period"], errors="coerce")
        f = f[(per >= lo) & (per <= hi)]
    if weeks and "week_num" in f.columns:
        wlo, whi = weeks
        wnum = pd.to_numeric(f["week_num"], errors="coerce")
        mask = (wnum >= wlo) & (wnum <= whi)
        f = f[mask | wnum.isna()]  # include unknown-week rows
    return f

# =============================================================================
# Aggregation
# =============================================================================
_TEAM_METRICS = ["paint_pts","sc_pts","fb_pts","pot_pts","p2_pts","p3_pts","ft_pts","total_pts"]

def _with_shares(g: pd.DataFrame) -> pd.DataFrame:
    for m in ["paint_pts","sc_pts","fb_pts","pot_pts","p2_pts","p3_pts","ft_pts"]:
        g[m+"_share"] = np.where(g["total_pts"]>0, (100.0 * g[m] / g["total_pts"]).round(1), np.nan)
    return g

def _agg_team_for(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame(columns=["team_name"] + _TEAM_METRICS + [m+"_share" for m in ["paint_pts","sc_pts","fb_pts","pot_pts","p2_pts","p3_pts","ft_pts"]])
    g = (f.groupby("team_name", dropna=False).agg({m:"sum" for m in _TEAM_METRICS}).reset_index())
    g = _with_shares(g)
    return g.sort_values(["total_pts"], ascending=False)

def _agg_team_against(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame(columns=["team_name"] + _TEAM_METRICS + [m+"_share" for m in ["paint_pts","sc_pts","fb_pts","pot_pts","p2_pts","p3_pts","ft_pts"]])
    df2 = _attach_opp(f)
    df2 = df2[~df2["opp_team_name"].isna()].copy()
    g = (df2.groupby("opp_team_name", dropna=False)
             .agg({m:"sum" for m in _TEAM_METRICS})
             .reset_index()
             .rename(columns={"opp_team_name":"team_name"}))
    g = _with_shares(g)
    return g.sort_values(["total_pts"], ascending=False)

def _agg_team_net(f: pd.DataFrame) -> pd.DataFrame:
    gf = _agg_team_for(f)
    ga = _agg_team_against(f)
    if gf.empty and ga.empty:
        return pd.DataFrame(columns=["team_name"] + [m+"_net" for m in _TEAM_METRICS] + [m+"_share_net" for m in ["paint_pts","sc_pts","fb_pts","pot_pts","p2_pts","p3_pts","ft_pts"]])
    merged = gf.merge(ga, on="team_name", how="outer", suffixes=("_for","_against")).fillna(0)
    out = pd.DataFrame({"team_name": merged["team_name"]})
    for m in _TEAM_METRICS:
        out[m+"_net"] = merged[m+"_for"] - merged[m+"_against"]
    for m in ["paint_pts","sc_pts","fb_pts","pot_pts","p2_pts","p3_pts","ft_pts"]:
        out[m+"_share_net"] = merged.get(m+"_share_for", 0) - merged.get(m+"_share_against", 0)
        out[m+"_share_net"] = out[m+"_share_net"].round(1)
    out = out.sort_values(["total_pts_net"], ascending=False)
    return out

def _agg_player(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame(columns=["scorer","team_name"] + _TEAM_METRICS + [m+"_share" for m in ["paint_pts","sc_pts","fb_pts","pot_pts","p2_pts","p3_pts","ft_pts"]])
    g = (f.groupby(["scorer","team_name"], dropna=False)
          .agg({m:"sum" for m in _TEAM_METRICS})
          .reset_index())
    g = _with_shares(g)
    return g.sort_values(["total_pts"], ascending=False)

# =============================================================================
# Figures
# =============================================================================
def _stacked_bar(df: pd.DataFrame, label_col: str, top_n: int, by_share: bool, title_suffix: str = "") -> px.bar:
    if df.empty:
        return px.bar()
    base = df.copy()
    is_net = any(c.endswith("_net") for c in df.columns if c not in [label_col])
    if is_net:
        cols = [("paint_pts_net","Paint Δ"),
                ("sc_pts_net","2ndCh Δ"),
                ("fb_pts_net","FB Δ"),
                ("pot_pts_net","POT Δ"),
                ("p2_pts_net","2PT Δ"),
                ("p3_pts_net","3PT Δ"),
                ("ft_pts_net","FT Δ")]
        base = base.sort_values("total_pts_net", ascending=False).head(top_n)
        plot_df = base[[label_col] + [c for c,_ in cols]].copy()
        plot_df = plot_df.rename(columns={c:new for c,new in cols})
        plot_df = plot_df.melt(id_vars=[label_col], var_name="Component", value_name="Value")
        fig = px.bar(plot_df, x="Value", y=label_col, color="Component", orientation="h",
                     title=f"Scoring Lenses (Net For–Against) {title_suffix}".strip())
    else:
        base = base.sort_values("total_pts", ascending=False).head(top_n)
        if by_share:
            cols = [("paint_pts_share","Paint %"),
                    ("sc_pts_share","2ndCh %"),
                    ("fb_pts_share","FB %"),
                    ("pot_pts_share","POT %"),
                    ("p2_pts_share","2PT %"),
                    ("p3_pts_share","3PT %"),
                    ("ft_pts_share","FT %")]
            plot_df = base[[label_col] + [c for c,_ in cols]].copy()
            plot_df = plot_df.rename(columns={c:new for c,new in cols})
            plot_df = plot_df.melt(id_vars=[label_col], var_name="Component", value_name="Value")
            fig = px.bar(plot_df, x="Value", y=label_col, color="Component", orientation="h",
                         title=f"Scoring Lenses (% of Total) {title_suffix}".strip())
        else:
            cols = [("paint_pts","Paint"),
                    ("sc_pts","Second-Chance"),
                    ("fb_pts","Fast-Break"),
                    ("pot_pts","POT"),
                    ("p2_pts","2PT"),
                    ("p3_pts","3PT"),
                    ("ft_pts","FT")]
            plot_df = base[[label_col] + [c for c,_ in cols]].copy()
            plot_df = plot_df.rename(columns={c:new for c,new in cols})
            plot_df = plot_df.melt(id_vars=[label_col], var_name="Component", value_name="Points")
            fig = px.bar(plot_df, x="Points", y=label_col, color="Component", orientation="h",
                         title=f"Scoring Lenses (Points) {title_suffix}".strip())
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), legend_title=None, yaxis=dict(autorange="reversed"))
    return fig

# =============================================================================
# Diagnostics
# =============================================================================
def _diag_summary(df_full: pd.DataFrame) -> html.Ul:
    keys = ["q_paint","q_fastbreak","q_second_chance","q_off_turnover","is_ft_made"]
    items = []
    for k in keys:
        if k in df_full.columns:
            items.append(html.Li(f"{k}: {int(df_full[k].sum())} events flagged"))
    if "qualifier" in df_full.columns:
        top_vals = (_lc(df_full["qualifier"]).value_counts().head(10).index.tolist())
        items.append(html.Li(f"Top qualifier values (first 10): {top_vals}"))
    return html.Ul(children=items)

def _diag_reconcile_team(df_full: pd.DataFrame, players: Optional[pd.DataFrame]) -> Optional[dash_table.DataTable]:
    if players is None:
        return None
    pbp_team = (df_full.groupby(["game_id","team_name"], dropna=False)
                .agg(pbp_paint=("paint_pts","sum"),
                     pbp_fb=("fb_pts","sum"),
                     pbp_sc=("sc_pts","sum"),
                     pbp_ft=("ft_pts","sum"))
                .reset_index())
    pl_cols = []
    if "sPointsInThePaint" in players.columns: pl_cols.append("sPointsInThePaint")
    if "sPointsFastBreak" in players.columns: pl_cols.append("sPointsFastBreak")
    if "sPointsSecondChance" in players.columns: pl_cols.append("sPointsSecondChance")
    if "sFreeThrowsMade" in players.columns: pl_cols.append("sFreeThrowsMade")
    if not pl_cols:
        return None
    pl_team = (players.groupby(["game_id","team_name"], dropna=False)
               .agg({c:"sum" for c in pl_cols})
               .reset_index())
    merged = pbp_team.merge(pl_team, on=["game_id","team_name"], how="left")
    if "sPointsInThePaint" in merged.columns:
        merged["delta_paint"] = (merged["pbp_paint"] - merged["sPointsInThePaint"]).round(1)
    if "sPointsFastBreak" in merged.columns:
        merged["delta_fb"] = (merged["pbp_fb"] - merged["sPointsFastBreak"]).round(1)
    if "sPointsSecondChance" in merged.columns:
        merged["delta_sc"] = (merged["pbp_sc"] - merged["sPointsSecondChance"]).round(1)
    if "sFreeThrowsMade" in merged.columns:
        merged["delta_ft"] = (merged["pbp_ft"] - merged["sFreeThrowsMade"]).round(1)
    show_cols = [c for c in ["game_id","team_name","pbp_paint","sPointsInThePaint","delta_paint",
                             "pbp_fb","sPointsFastBreak","delta_fb",
                             "pbp_sc","sPointsSecondChance","delta_sc",
                             "pbp_ft","sFreeThrowsMade","delta_ft"] if c in merged.columns]
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in show_cols],
        data=merged[show_cols].to_dict("records"),
        style_table={"overflowX":"auto"},
        page_size=12,
    )

# =============================================================================
# Small UI helpers
# =============================================================================
def _kpi(title: str, value: str):
    return html.Div(className="kpi", children=[
        html.Div(title, style={"fontSize":"12px","color":"#667"}),
        html.Div(value, style={"fontSize":"20px","fontWeight":700,"marginTop":"2px"}),
    ])

def _kpi_small(title: str, value: str):
    return html.Div(className="kpi-mini", children=[
        html.Div(title, style={"fontSize":"11px","color":"#667"}),
        html.Div(value, style={"fontSize":"16px","fontWeight":700,"marginTop":"2px"}),
    ])

# =============================================================================
# Page
# =============================================================================
def _page_content(df: pd.DataFrame) -> html.Div:
    team_opts = [{"label": t, "value": t} for t in sorted([t for t in df["team_name"].dropna().unique()])]
    _games_view = df[["week","game_id","label"]].copy()
    _games_view["week"] = _games_view["week"].fillna("")
    _games_view["game_id"] = _games_view["game_id"].astype(str).fillna("")
    _games_view["label"] = _games_view["label"].fillna("")
    game_opts = [{"label": f'{w} • {gid} • {lab}', "value": str(gid)}
                 for w, gid, lab in _games_view.drop_duplicates().sort_values(["week","game_id"]).itertuples(index=False)]
    period_max = int(pd.to_numeric(df["period"], errors="coerce").max() or 4)

    # Week slider bounds & marks
    if "week_num" in df.columns and not df["week_num"].dropna().empty:
        week_min = int(np.nanmin(df["week_num"]))
        week_max = int(np.nanmax(df["week_num"]))
    else:
        week_min = 1
        week_max = 1

    metric_opts = [
        {"label":"Paint Points", "value":"paint_pts"},
        {"label":"Second-Chance Points", "value":"sc_pts"},
        {"label":"Fast-Break Points", "value":"fb_pts"},
        {"label":"Points off Turnovers (POT)", "value":"pot_pts"},
        {"label":"2PT Points", "value":"p2_pts"},
        {"label":"3PT Points", "value":"p3_pts"},
        {"label":"Free-Throw Points", "value":"ft_pts"},
        {"label":"Total Points", "value":"total_pts"},
    ]

    return html.Div(children=[
        html.H2("Scoring Leaderboards", style={"marginBottom":"6px"}),

        # Dense control bar (better space use)
        html.Div(style={
            "display":"grid",
            "gridTemplateColumns":"repeat(6, minmax(140px,1fr))",
            "gap":"8px"
        }, children=[
            html.Div(children=[
                html.Label("Mode / Perspective"),
                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"6px"}, children=[
                    dcc.Dropdown(id="sl-mode", options=[{"label":"Teams","value":"teams"},{"label":"Players","value":"players"}], value="teams", clearable=False),
                    dcc.Dropdown(id="sl-perspective", options=[{"label":"For","value":"for"},{"label":"Against","value":"against"},{"label":"Net (F−A)","value":"net"}], value="for", clearable=False),
                ])
            ]),
            html.Div(children=[
                html.Label("Team(s)"),
                dcc.Dropdown(id="sl-teams", options=team_opts, multi=True, placeholder="Select teams")
            ]),
            html.Div(children=[
                html.Label("Game(s)"),
                dcc.Dropdown(id="sl-games", options=game_opts, multi=True, placeholder="Filter by game")
            ]),
            html.Div(children=[
                html.Label("Periods"),
                dcc.RangeSlider(id="sl-periods", min=1, max=period_max, step=1, value=[1, period_max],
                                marks={i:str(i) for i in range(1, period_max+1)})
            ]),
            html.Div(children=[
                html.Label("Week Range"),
                dcc.RangeSlider(id="sl-weeks",
                                min=week_min, max=week_max, step=1, value=[week_min, week_max],
                                marks=_week_marks(week_min, week_max))
            ]),
            html.Div(children=[
                html.Label("Rank by / Options"),
                html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr","gap":"6px"}, children=[
                    dcc.Dropdown(id="sl-metric", options=metric_opts, value="total_pts", clearable=False),
                    dcc.Checklist(id="sl-use-share", options=[{"label":"%","value":"share"}], value=[], style={"marginTop":"6px"})
                ]),
            ]),
        ]),

        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(6, minmax(140px,1fr))","gap":"8px","marginTop":"6px"}, children=[
            html.Div(children=[
                html.Label("Top N (chart)"),
                dcc.Slider(id="sl-topn", min=5, max=30, step=1, value=15,
                           marks={5:"5",10:"10",15:"15",20:"20",25:"25",30:"30"})
            ]),
            html.Div(children=[
                html.Label("Min Total Points"),
                dcc.Slider(id="sl-min-pts", min=0, max=200, step=5, value=0,
                           marks={0:"0",25:"25",50:"50",100:"100",150:"150",200:"200"})
            ]),
            html.Div(children=[
                html.Label("Compact Table"),
                dcc.Checklist(id="sl-compact", options=[{"label":" Compact","value":"on"}], value=[])
            ]),
            html.Div(), html.Div(), html.Div(),  # spacers to keep grid even
        ]),

        html.Hr(style={"margin":"8px 0"}),

        # KPI rows: primary + shooting panel (space-efficient mini-cards)
        html.Div(id="sl-kpis", style={"display":"grid","gridTemplateColumns":"repeat(4, minmax(140px, 1fr))","gap":"8px","marginBottom":"6px"}),
        html.Div(id="sl-kpis-shoot", style={"display":"grid","gridTemplateColumns":"repeat(10, minmax(90px, 1fr))","gap":"6px"}),

        html.H3("Leaderboard", style={"marginTop":"10px","marginBottom":"6px"}),

        dash_table.DataTable(
            id="sl-table",
            columns=[], data=[],
            sort_action="native", filter_action="native",
            page_size=20,
            style_table={"overflowX":"auto"},
            style_header={"fontWeight":"600", "padding":"6px 8px"},
            style_cell={"padding":"6px 8px"},
            export_format="csv", export_headers="display",
        ),

        html.Div(style={"marginTop":"10px"}, children=[ dcc.Graph(id="sl-bars") ]),

        html.Hr(style={"margin":"10px 0"}),

        html.Div(style={"display":"flex","gap":"12px","alignItems":"center"}, children=[
            dcc.Checklist(id="sl-show-diag", options=[{"label": " Show diagnostics (flags + team reconciliation)", "value": "on"}], value=[]),
        ]),
        html.Div(id="sl-diag", style={"marginTop":"6px"}),

        html.Div(style={"marginTop":"10px","fontSize":"12px","color":"#666"}, children=[
            html.Span("Categories can overlap (e.g., POT may include FTs). Lenses come only from `qualifier` tokens; FT points from FT makes.")
        ])
    ])

# =============================================================================
# Dash app
# =============================================================================
def create_dash_scoring_leaders(server, base_pathname: str = "/scoring_leaders/") -> Dash:
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
        title="Scoring Leaderboards",
    )

    df = _pbp()
    content = _page_content(df)
    nav_items = server.config.get("NAV", [])
    app.layout = PageShell(content, nav_items, current_endpoint=base_pathname)

    @app.callback(
        Output("sl-table","data"),
        Output("sl-table","columns"),
        Output("sl-kpis","children"),
        Output("sl-kpis-shoot","children"),
        Output("sl-bars","figure"),
        Output("sl-diag","children"),
        Input("sl-mode","value"),
        Input("sl-perspective","value"),
        Input("sl-teams","value"),
        Input("sl-games","value"),
        Input("sl-periods","value"),
        Input("sl-weeks","value"),
        Input("sl-metric","value"),
        Input("sl-use-share","value"),
        Input("sl-topn","value"),
        Input("sl-min-pts","value"),
        Input("sl-compact","value"),
        Input("sl-show-diag","value"),
    )
    def update_view(mode, perspective, teams, games, periods, weeks, metric, use_share, topn, min_pts, compact, show_diag):
        df_local = _pbp()
        players = _DF_PLAYERS
        teams = teams or []
        games = games or []
        periods = periods or [1, int(pd.to_numeric(df_local["period"], errors="coerce").max() or 4)]
        if "week_num" in df_local.columns and not df_local["week_num"].dropna().empty:
            wmin = int(np.nanmin(df_local["week_num"]))
            wmax = int(np.nanmax(df_local["week_num"]))
        else:
            wmin = 1; wmax = 1
        weeks = weeks or [wmin, wmax]

        # Filter by games, periods, weeks (team filter after agg)
        f = _filtered(df_local, games=games, periods=tuple(periods), weeks=tuple(weeks))

        # Choose aggregation path
        if mode == "teams":
            if perspective == "for":
                agg = _agg_team_for(f); label_col = "team_name"; title_suffix = "— Teams (For)"
            elif perspective == "against":
                agg = _agg_team_against(f); label_col = "team_name"; title_suffix = "— Teams (Against)"
            else:
                agg = _agg_team_net(f); label_col = "team_name"; title_suffix = "— Teams (Net F−A)"
            if teams:
                agg = agg[agg["team_name"].isin(teams)]
        else:
            agg = _agg_player(f); label_col = "scorer"; title_suffix = "— Players (For)"
            if teams:
                agg = agg[agg["team_name"].isin(teams)]

        min_pts = int(min_pts or 0)

        # Build columns and sort
        by_share = "share" in (use_share or [])
        if mode == "teams" and perspective == "net":
            columns = [{"name":"Team","id":"team_name"}] + [
                {"name":"Paint Δ","id":"paint_pts_net","type":"numeric"},
                {"name":"2nd-Ch Δ","id":"sc_pts_net","type":"numeric"},
                {"name":"FB Δ","id":"fb_pts_net","type":"numeric"},
                {"name":"POT Δ","id":"pot_pts_net","type":"numeric"},
                {"name":"2PT Δ","id":"p2_pts_net","type":"numeric"},
                {"name":"3PT Δ","id":"p3_pts_net","type":"numeric"},
                {"name":"FT Δ","id":"ft_pts_net","type":"numeric"},
                {"name":"Total Δ","id":"total_pts_net","type":"numeric"},
                {"name":"Paint % Δ","id":"paint_pts_share_net","type":"numeric"},
                {"name":"2nd-Ch % Δ","id":"sc_pts_share_net","type":"numeric"},
                {"name":"FB % Δ","id":"fb_pts_share_net","type":"numeric"},
                {"name":"POT % Δ","id":"pot_pts_share_net","type":"numeric"},
                {"name":"2PT % Δ","id":"p2_pts_share_net","type":"numeric"},
                {"name":"3PT % Δ","id":"p3_pts_share_net","type":"numeric"},
                {"name":"FT % Δ","id":"ft_pts_share_net","type":"numeric"},
            ]
            rank_key = "total_pts_net"
        else:
            common_cols = [
                {"name":"Paint", "id":"paint_pts","type":"numeric"},
                {"name":"2nd-Chance", "id":"sc_pts","type":"numeric"},
                {"name":"Fast-Break", "id":"fb_pts","type":"numeric"},
                {"name":"POT", "id":"pot_pts","type":"numeric"},
                {"name":"2PT", "id":"p2_pts","type":"numeric"},
                {"name":"3PT", "id":"p3_pts","type":"numeric"},
                {"name":"FT", "id":"ft_pts","type":"numeric"},
                {"name":"Total", "id":"total_pts","type":"numeric"},
                {"name":"Paint %", "id":"paint_pts_share","type":"numeric"},
                {"name":"2nd-Chance %", "id":"sc_pts_share","type":"numeric"},
                {"name":"FB %", "id":"fb_pts_share","type":"numeric"},
                {"name":"POT %", "id":"pot_pts_share","type":"numeric"},
                {"name":"2PT %", "id":"p2_pts_share","type":"numeric"},
                {"name":"3PT %", "id":"p3_pts_share","type":"numeric"},
                {"name":"FT %", "id":"ft_pts_share","type":"numeric"},
            ]
            if mode == "teams":
                columns = [{"name":"Team","id":"team_name"}] + common_cols
            else:
                columns = [{"name":"Player","id":"scorer"},{"name":"Team","id":"team_name"}] + common_cols
            rank_key = (metric + "_share") if by_share and (metric+"_share") in agg.columns else metric
            if "total_pts" in agg.columns and min_pts > 0:
                agg = agg[agg["total_pts"] >= min_pts] if not agg.empty else agg

        # Sort
        if rank_key in agg.columns:
            if rank_key.endswith("_share") or rank_key.endswith("_net"):
                agg = agg.sort_values([rank_key], ascending=[False])
            else:
                sec = "total_pts" if "total_pts" in agg.columns else rank_key
                agg = agg.sort_values([rank_key, sec], ascending=[False, False])

        # Apply compact mode to table
        compact_on = "on" in (compact or [])
        table_style_cell = {"padding": ("4px 6px" if compact_on else "6px 8px")}
        table_style_header = {"fontWeight":"600", "padding": ("4px 6px" if compact_on else "6px 8px")}
        dash_table.DataTable.style_cell = table_style_cell  # apply live
        dash_table.DataTable.style_header = table_style_header

        rows = agg[[c["id"] for c in columns if c["id"] in agg.columns]].to_dict("records")

        # KPIs (primary)
        if mode == "teams" and perspective == "against":
            total_val = int(agg["total_pts"].sum()) if ("total_pts" in agg.columns and not agg.empty) else 0
            kpi_title = "Filtered Total Conceded (Teams)"
        elif mode == "teams" and perspective == "net":
            total_val = int(agg["total_pts_net"].sum()) if ("total_pts_net" in agg.columns and not agg.empty) else 0
            kpi_title = "Sum of Team Total Δ (For−Against)"
        else:
            total_val = int(f["total_pts"].sum()) if not f.empty else 0
            kpi_title = "Filtered Total Points"
        total_games = int(f["game_id"].nunique()) if "game_id" in f.columns and not f.empty else 0

        kpis_primary = [
            _kpi(kpi_title, f"{total_val:,}"),
            _kpi("Games Covered", f"{total_games:,}"),
            _kpi("Rows", f"{len(agg):,}"),
            _kpi("Ranking by", ("%" if (("share" in (use_share or [])) and perspective != "net") else "Points")
                 + f" – {metric.replace('_',' ').upper() if perspective!='net' else 'TOTAL Δ'}"),
        ]

        # Shooting panel KPIs (space-efficient)
        if f.empty:
            kpis_shoot = []
        else:
            fga = int(f["fga"].sum()) if "fga" in f.columns else int(((f["ev_code"].isin(["p2","p3"]))).sum())
            fgm = int(f["fgm"].sum()) if "fgm" in f.columns else int((((f["ev_code"].isin(["p2","p3"])) & (f["ev_result"]=="made"))).sum())
            p3m = int(f["p3m"].sum()) if "p3m" in f.columns else int(((f["ev_code"]=="p3") & (f["ev_result"]=="made")).sum())
            p2a = int(f["p2a"].sum()) if "p2a" in f.columns else int((f["ev_code"]=="p2").sum())
            p3a = int(f["p3a"].sum()) if "p3a" in f.columns else int((f["ev_code"]=="p3").sum())
            fg_pct  = (100.0 * fgm / fga) if fga > 0 else 0.0
            efg_pct = (100.0 * (fgm + 0.5 * p3m) / fga) if fga > 0 else 0.0
            pts = int(f["total_pts"].sum())
            two_share = (100.0 * p2a / fga) if fga > 0 else 0.0
            three_share = 100.0 - two_share if fga > 0 else 0.0
            pip = float(f["paint_pts"].sum()) if "paint_pts" in f.columns else 0.0
            scp = float(f["sc_pts"].sum()) if "sc_pts" in f.columns else 0.0
            fbp = float(f["fb_pts"].sum()) if "fb_pts" in f.columns else 0.0

            def pct(x): return f"{x:.1f}"
            def num(x): return f"{x:,.0f}"

            kpis_shoot = [
                _kpi_small("FGA",   num(fga)),
                _kpi_small("FGM",   num(fgm)),
                _kpi_small("FG%",   pct(fg_pct)),
                _kpi_small("eFG%",  pct(efg_pct)),
                _kpi_small("PTS",   num(pts)),
                _kpi_small("2P Share", pct(two_share) + "%"),
                _kpi_small("3P Share", pct(three_share) + "%"),
                _kpi_small("PIP",   num(pip)),
                _kpi_small("2ndCh", num(scp)),
                _kpi_small("FB",    num(fbp)),
            ]

        fig = _stacked_bar(agg, label_col, int(topn or 15), by_share and (perspective != "net"), title_suffix=title_suffix)

        # Diagnostics
        diag_children = []
        if "on" in (show_diag or []):
            diag_children.append(html.H4("Diagnostics"))
            diag_children.append(_diag_summary(f))
            table = _diag_reconcile_team(f, players)
            if table is not None:
                diag_children.append(html.Div(style={"marginTop":"6px"}, children=[
                    html.Div("Team-level reconciliation with players_all.csv (PBP vs players totals)", style={"fontWeight":600, "marginBottom":"4px"}),
                    table
                ]))

        return rows, columns, kpis_primary, kpis_shoot, fig, diag_children

    return app
