# dashapps/scoring_frequency.py
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update

# Optional shell wrapper (renders a common header/nav if present)
try:
    from .common import page_shell as PageShell  # type: ignore
except Exception:
    PageShell = None

# Plotly (bundled with Dash)
import plotly.express as px

# =============================================================================
# Robust IO
# =============================================================================

def _data_root() -> Path:
    """
    Resolve DATA_ROOT in a robust order:
    1) DATA_ROOT env
    2) ./out_data
    """
    return Path(os.environ.get("DATA_ROOT", "out_data"))

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p and p.exists():
                return p
        except Exception:
            pass
    return None

@lru_cache(maxsize=1)
def _read_pbp() -> pd.DataFrame:
    """
    Locate and read a league-wide cumulative PBP table.
    Search order:
      - DATA_ROOT/_cumulative/pbp_all.csv
      - DATA_ROOT/pbp_all.csv
      - /mnt/data/pbp_all.csv (ChatGPT sandbox fallback)
    """
    root = _data_root()
    candidates = [
        root / "_cumulative" / "pbp_all.csv",
        root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),
    ]
    path = _first_existing(candidates)
    if path is None:
        return pd.DataFrame()

    # CSV read with minimal dtype enforcement; let downstream coerce
    try:
        return pd.read_csv(path)
    except Exception:
        # last-ditch: try latin-1 or errors='ignore'
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return pd.read_csv(path, encoding_errors="ignore")


# =============================================================================
# Schema helpers
# =============================================================================

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create harmonized columns:
      - team_col: team_key/team/Team/scoring_team → 'TEAM'
      - player_col: player_key/player/Player/scorer → 'PLAYER'
      - week_col: Week/week → 'WEEK' (optional)
      - action code: ac/AC/action_code → 'AC' (optional)
      - made flag: made/is_made/result → 'MADE' (bool if possible)
      - text: text/desc/play/event_text → 'TEXT' (optional)
      - points: points/pts/score_pts → 'POINTS' (numeric if present)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["TEAM", "PLAYER", "WEEK", "AC", "MADE", "TEXT", "POINTS"])

    out = df.copy()
    # TEAM
    tcol = _first_col(out, ["team_key", "team", "Team", "scoring_team"])
    if tcol is None:
        out["TEAM"] = ""
    else:
        out["TEAM"] = out[tcol].astype(str).fillna("")
    # PLAYER
    pcol = _first_col(out, ["player_key", "player", "Player", "scorer"])
    if pcol is None:
        out["PLAYER"] = ""
    else:
        out["PLAYER"] = out[pcol].astype(str).fillna("")
    # WEEK
    wcol = _first_col(out, ["Week", "week"])
    out["WEEK"] = out[wcol] if wcol else np.nan
    # AC
    acol = _first_col(out, ["ac", "AC", "action_code"])
    out["AC"] = out[acol].astype(str) if acol else ""
    # MADE
    mcol = _first_col(out, ["made", "is_made", "result"])
    if mcol:
        # try to coerce to bool (1/0/True/False/"made"/"miss")
        m = out[mcol]
        if m.dtype == bool:
            out["MADE"] = m
        else:
            # textual normalize
            ms = m.astype(str).str.lower()
            out["MADE"] = ms.isin(["1", "true", "t", "made", "success", "scored", "yes"])
    else:
        out["MADE"] = pd.NA
    # TEXT
    txcol = _first_col(out, ["text", "desc", "play", "event_text"])
    out["TEXT"] = out[txcol].astype(str) if txcol else ""
    # POINTS
    ptscol = _first_col(out, ["points", "pts", "score_pts"])
    if ptscol:
        out["POINTS"] = pd.to_numeric(out[ptscol], errors="coerce")
    else:
        out["POINTS"] = pd.NA

    return out[["TEAM", "PLAYER", "WEEK", "AC", "MADE", "TEXT", "POINTS"]]

def _infer_2p3p(row: pd.Series) -> Tuple[bool, bool, bool]:
    """
    Return tuple: (is_shot_event, is_2pt_made, is_3pt_made)

    Heuristics priority:
      1) If POINTS in {2,3} and looks like a made FG → trust it.
      2) Else use AC codes: 'P2'/'2' vs 'P3'/'3' plus MADE True/“made” in TEXT.
      3) Else parse TEXT.
    """
    # 1) Points heuristic
    pts = row.get("POINTS", pd.NA)
    if pd.notna(pts):
        try:
            ipts = int(pts)
            if ipts == 2:
                return True, True, False
            if ipts == 3:
                return True, False, True
        except Exception:
            pass

    # 2) AC + MADE
    ac = str(row.get("AC", "")).upper()
    made_flag = row.get("MADE", pd.NA)
    text = str(row.get("TEXT", "")).lower()

    text_has_made = any(x in text for x in [" made", "scores", "good", "is good"])
    made = (bool(made_flag) if pd.notna(made_flag) else text_has_made)

    if "P3" in ac or ac.strip() in {"3", "3PT", "3P"}:
        return True, False, bool(made)
    if "P2" in ac or ac.strip() in {"2", "2PT", "2P"}:
        return True, bool(made), False

    # 3) TEXT-only fallback
    tl = text
    if ("made" in tl or "good" in tl or "scores" in tl) and ("3pt" in tl or "3-pt" in tl or "three" in tl):
        return True, False, True
    if ("made" in tl or "good" in tl or "scores" in tl) and ("2pt" in tl or "2-pt" in tl or "two" in tl or "layup" in tl or "dunk" in tl or "hook" in tl or "float" in tl or "fadeaway" in tl or "jump shot" in tl):
        return True, True, False

    return False, False, False


def _prepare_made_shots(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a slim table with only made FGs and normalized fields:
      columns: TEAM, PLAYER, WEEK, KIND in {"2P","3P"}
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["TEAM", "PLAYER", "WEEK", "KIND"])

    df = _normalize_columns(df_raw)

    # Infer events
    flags = df.apply(_infer_2p3p, axis=1, result_type="expand")
    flags.columns = ["is_shot", "is_2pm", "is_3pm"]
    df = pd.concat([df, flags], axis=1)

    made = df["is_shot"] & (df["is_2pm"] | df["is_3pm"])
    slim = df.loc[made, ["TEAM", "PLAYER", "WEEK", "is_2pm", "is_3pm"]].copy()
    slim["KIND"] = np.where(slim["is_3pm"], "3P", "2P")
    return slim[["TEAM", "PLAYER", "WEEK", "KIND"]]


# =============================================================================
# Dash app
# =============================================================================

def _make_layout(app: Dash, has_week: bool, teams: List[str], players: List[str]):
    controls = html.Div([
        html.Div([
            html.Label("Mode"),
            dcc.RadioItems(
                id="sf-mode",
                options=[{"label": "Team", "value": "team"}, {"label": "Player", "value": "player"}],
                value="team",
                inline=True,
                inputClassName="mr-2",
            ),
        ], className="mb-2"),

        html.Div([
            html.Label("Team"),
            dcc.Dropdown(
                id="sf-team",
                options=[{"label": t, "value": t} for t in ["(League)"] + teams],
                value="(League)",
                clearable=False,
            ),
        ], className="mb-2"),

        html.Div([
            html.Label("Player (optional)"),
            dcc.Dropdown(
                id="sf-player",
                options=[{"label": p, "value": p} for p in ["(All)"] + players],
                value="(All)",
                clearable=False,
            ),
        ], className="mb-2"),

        html.Div([
            html.Label("Week range" if has_week else "Week range (not available)"),
            dcc.RangeSlider(
                id="sf-week",
                min=0 if not has_week else  int(pd.to_numeric(app.server.config.get("SF_WEEK_MIN", 1))),
                max=0 if not has_week else  int(pd.to_numeric(app.server.config.get("SF_WEEK_MAX", 1))),
                step=1,
                value=[int(app.server.config.get("SF_WEEK_MIN", 1)), int(app.server.config.get("SF_WEEK_MAX", 1))] if has_week else [0, 0],
                tooltip={"placement": "bottom"},
                marks=None,
                allowCross=False,
                disabled=not has_week,
            ),
        ], className="mb-2"),
    ], className="p-3")

    content = html.Div([
        html.Div([
            dcc.Graph(id="sf-pie"),
        ], className="mb-4"),

        html.Div([
            dcc.Graph(id="sf-bar"),
        ], className="mb-4"),

        html.Div([
            dash_table.DataTable(
                id="sf-table",
                columns=[
                    {"name": "Group", "id": "group"},
                    {"name": "2PM",   "id": "m2"},
                    {"name": "3PM",   "id": "m3"},
                    {"name": "Total", "id": "tot"},
                    {"name": "2P %",  "id": "s2"},
                    {"name": "3P %",  "id": "s3"},
                ],
                data=[],
                sort_action="native",
                filter_action="native",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={"padding": "6px", "fontSize": 14},
                style_header={"fontWeight": "600"},
            )
        ])
    ], className="p-3")

    inner = html.Div([
        html.H2("Scoring Frequency (2P vs 3P)"),
        html.P("Explore the scoring mix by team or player; optionally scope by Week if available."),
        html.Div(className="grid grid-cols-1 md:grid-cols-3 gap-4", children=[
            html.Div(controls, className="col-span-1 card"),
            html.Div(content, className="col-span-2 card"),
        ])
    ], className="container mx-auto")

    if PageShell:
        return PageShell("Scoring Frequency (2P/3P)", inner)
    return inner


def _initial_lists(slim: pd.DataFrame) -> Tuple[List[str], List[str], bool, int, int]:
    teams = sorted([t for t in slim["TEAM"].dropna().astype(str).unique() if t])
    players = sorted([p for p in slim["PLAYER"].dropna().astype(str).unique() if p])
    has_week = "WEEK" in slim.columns and slim["WEEK"].notna().any()
    wmin, wmax = 1, 1
    if has_week:
        w = pd.to_numeric(slim["WEEK"], errors="coerce").dropna()
        if len(w):
            wmin, wmax = int(w.min()), int(w.max())
    return teams, players, has_week, wmin, wmax


def create_dash_scoring_frequency(server=None, base_pathname: str = "/scoring_frequency/"):
    app = Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
    )

    raw = _read_pbp()
    slim = _prepare_made_shots(raw)
    teams, players, has_week, wmin, wmax = _initial_lists(slim)

    # expose week min/max to layout creator
    if server is not None:
        server.config["SF_WEEK_MIN"] = wmin
        server.config["SF_WEEK_MAX"] = wmax

    app.layout = _make_layout(app, has_week, teams, players)

    # ---------------- Callbacks ----------------

    @app.callback(
        Output("sf-player", "options"),
        Output("sf-player", "value"),
        Input("sf-mode", "value"),
        Input("sf-team", "value"),
        prevent_initial_call=False,
    )
    def _update_players(mode: str, team_val: str):
        if mode == "player":
            # In player mode, keep dropdown open for all players (team filter optional)
            if team_val and team_val not in ("(League)", ""):
                opts = ["(All)"] + sorted(slim.loc[slim["TEAM"] == team_val, "PLAYER"].dropna().unique().tolist())
            else:
                opts = ["(All)"] + sorted(slim["PLAYER"].dropna().unique().tolist())
            return [{"label": p, "value": p} for p in opts], "(All)"
        else:
            # In team mode, player selection is ignored
            return [{"label": "(All)", "value": "(All)"}], "(All)"

    @app.callback(
        Output("sf-pie", "figure"),
        Output("sf-bar", "figure"),
        Output("sf-table", "data"),
        Input("sf-mode", "value"),
        Input("sf-team", "value"),
        Input("sf-player", "value"),
        Input("sf-week", "value"),
    )
    def _compute(mode: str, team_val: str, player_val: str, week_rng):
        df = slim.copy()

        # Week slice
        try:
            if "WEEK" in df.columns and week_rng and len(week_rng) == 2:
                lo, hi = week_rng
                mask_w = pd.to_numeric(df["WEEK"], errors="coerce").between(lo, hi, inclusive="both")
                df = df.loc[mask_w]
        except Exception:
            pass

        if mode == "team":
            if team_val and team_val not in ("(League)", ""):
                df = df.loc[df["TEAM"] == team_val]
                group_label = team_val
            else:
                group_label = "League"

            # Summary for chosen scope (2P vs 3P counts)
            m2 = int((df["KIND"] == "2P").sum())
            m3 = int((df["KIND"] == "3P").sum())
            tot = m2 + m3
            s2 = (100.0 * m2 / tot) if tot > 0 else 0.0
            s3 = 100.0 - s2 if tot > 0 else 0.0

            pie_df = pd.DataFrame({"Kind": ["2P", "3P"], "Made": [m2, m3]})
            pie = px.pie(pie_df, names="Kind", values="Made", title=f"{group_label}: 2P vs 3P Share")

            # Player contribution bars inside this scope (top 12)
            if team_val and team_val not in ("(League)", ""):
                by_player = df.groupby(["PLAYER", "KIND"], dropna=False).size().reset_index(name="Made")
                # Pivot to 2P/3P per player
                pivot = by_player.pivot_table(index="PLAYER", columns="KIND", values="Made", fill_value=0).reset_index()
                if "2P" not in pivot.columns: pivot["2P"] = 0
                if "3P" not in pivot.columns: pivot["3P"] = 0
                pivot["Total"] = pivot["2P"] + pivot["3P"]
                pivot = pivot.sort_values("Total", ascending=False).head(12)
                bar = px.bar(pivot, x="PLAYER", y=["2P", "3P"], barmode="stack", title=f"{group_label}: Top Players by 2P/3P Made")
            else:
                # League-wide per-team bars (top 12)
                by_team = df.groupby(["TEAM", "KIND"], dropna=False).size().reset_index(name="Made")
                pivot = by_team.pivot_table(index="TEAM", columns="KIND", values="Made", fill_value=0).reset_index()
                if "2P" not in pivot.columns: pivot["2P"] = 0
                if "3P" not in pivot.columns: pivot["3P"] = 0
                pivot["Total"] = pivot["2P"] + pivot["3P"]
                pivot = pivot.sort_values("Total", ascending=False).head(12)
                bar = px.bar(pivot, x="TEAM", y=["2P", "3P"], barmode="stack", title="League: Top Teams by 2P/3P Made")

            table_data = [{
                "group": group_label,
                "m2": m2,
                "m3": m3,
                "tot": tot,
                "s2": f"{s2:.2f}%",
                "s3": f"{s3:.2f}%"
            }]

            return pie, bar, table_data

        # ----- player mode -----
        if team_val and team_val not in ("(League)", ""):
            df = df.loc[df["TEAM"] == team_val]

        if player_val and player_val not in ("(All)", ""):
            df = df.loc[df["PLAYER"] == player_val]
            label = f"{player_val} ({team_val})" if team_val and team_val not in ("(League)", "") else player_val
        else:
            label = f"Players in {team_val}" if team_val and team_val not in ("(League)", "") else "League Players"

        m2 = int((df["KIND"] == "2P").sum())
        m3 = int((df["KIND"] == "3P").sum())
        tot = m2 + m3
        s2 = (100.0 * m2 / tot) if tot > 0 else 0.0
        s3 = 100.0 - s2 if tot > 0 else 0.0

        pie_df = pd.DataFrame({"Kind": ["2P", "3P"], "Made": [m2, m3]})
        pie = px.pie(pie_df, names="Kind", values="Made", title=f"{label}: 2P vs 3P Share")

        # Bar: top players in current scope (team filter applied if chosen)
        by_player = df.groupby(["PLAYER", "KIND"], dropna=False).size().reset_index(name="Made")
        pivot = by_player.pivot_table(index="PLAYER", columns="KIND", values="Made", fill_value=0).reset_index()
        if "2P" not in pivot.columns: pivot["2P"] = 0
        if "3P" not in pivot.columns: pivot["3P"] = 0
        pivot["Total"] = pivot["2P"] + pivot["3P"]
        pivot = pivot.sort_values("Total", ascending=False).head(12)
        bar = px.bar(pivot, x="PLAYER", y=["2P", "3P"], barmode="stack", title=f"{label}: Top Players by 2P/3P Made")

        table_data = [{
            "group": label,
            "m2": m2,
            "m3": m3,
            "tot": tot,
            "s2": f"{s2:.2f}%",
            "s3": f"{s3:.2f}%"
        }]

        return pie, bar, table_data

    return app