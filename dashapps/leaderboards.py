from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px

from .common import page_shell as PageShell

# ---------------------------------------------------------------------------
# Data discovery / loading (mirrors assist_combos conventions)
# ---------------------------------------------------------------------------
def _find_file(candidates: List[Path]) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None

def _find_pbp_all() -> Optional[Path]:
    env = os.environ.get("PBP_ALL")
    if env and Path(env).exists():
        return Path(env)

    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "_cumulative" / "pbp_all.csv",
        data_root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),  # notebook path
    ]
    f = _find_file(candidates)
    if f:
        return f

    for c in data_root.rglob("pbp_all.csv"):
        return c
    return None

def _find_players_all() -> Optional[Path]:
    env = os.environ.get("PLAYERS_ALL")
    if env and Path(env).exists():
        return Path(env)

    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "_cumulative" / "players_all.csv",
        data_root / "players_all.csv",
        Path("/mnt/data/players_all.csv"),
    ]
    f = _find_file(candidates)
    if f:
        return f

    for c in data_root.rglob("players_all.csv"):
        return c
    return None

def _ensure_cols(df: pd.DataFrame, cols: List[str], fill=None) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def _load_pbp_all() -> pd.DataFrame:
    src = _find_pbp_all()
    if not src or not src.exists():
        raise FileNotFoundError("pbp_all.csv not found. Set PBP_ALL=/path/to/pbp_all.csv or place it under DATA_ROOT/_cumulative/.")

    df = pd.read_csv(src)
    df = _ensure_cols(df, ["game_id", "week", "label"], fill=np.nan)
    df = _ensure_cols(df, ["player", "player_name_from_roster", "assist_player", "team_name",
                           "ev_code", "ev_result", "subType"], fill=np.nan)

    for c in ["period", "actionNumber", "tno", "pno", "assist_tno", "assist_pno", "points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ev_code"] = df["ev_code"].fillna("").astype(str)
    df["ev_result"] = df["ev_result"].fillna("").astype(str)
    df["finisher"] = df["player"].fillna(df["player_name_from_roster"]).fillna("Unknown Player")

    # shot type P2/P3 only for assisted FG logic
    df["shot_type"] = np.where(df["ev_code"] == "P3", "3PT",
                        np.where(df["ev_code"] == "P2", "2PT", None))

    if "points" not in df.columns:
        df["points"] = np.where(df["ev_code"] == "P3", 3,
                         np.where(df["ev_code"] == "P2", 2,
                         np.where(df["ev_code"] == "FT", 1, 0)))
    if "period" not in df.columns:
        df["period"] = 1

    return df

def _load_players_all() -> pd.DataFrame:
    src = _find_players_all()
    if not src or not src.exists():
        # optional – proceed without player meta
        return pd.DataFrame(columns=["team_name", "player_name", "position_group"])
    df = pd.read_csv(src)
    # normalize likely column names to join to PBP names
    # expected columns: team_name, name (or player_name), position_group (optional)
    if "player_name" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "player_name"})
    if "team" in df.columns and "team_name" not in df.columns:
        df = df.rename(columns={"team": "team_name"})
    return df

_DF_PBP: Optional[pd.DataFrame] = None
_DF_PLAYERS: Optional[pd.DataFrame] = None

def _pbp() -> pd.DataFrame:
    global _DF_PBP
    if _DF_PBP is None:
        _DF_PBP = _load_pbp_all()
    return _DF_PBP

def _players() -> pd.DataFrame:
    global _DF_PLAYERS
    if _DF_PLAYERS is None:
        _DF_PLAYERS = _load_players_all()
    return _DF_PLAYERS

# ---------------------------------------------------------------------------
# Filtering and event selection (assisted, made, P2/P3)
# ---------------------------------------------------------------------------
def _filter_events(df: pd.DataFrame,
                   teams: List[str],
                   games: List[str],
                   periods: Tuple[int, int],
                   shot_types: List[str]) -> pd.DataFrame:
    f = df.copy()
    # assisted made field goals only
    mask = (f["assist_player"].notna()) & (f["ev_result"].str.lower() == "made") & (f["ev_code"].isin(["P2", "P3"]))
    f = f.loc[mask]

    if teams:
        f = f[f["team_name"].isin(teams)]
    if games:
        f = f[f["game_id"].astype(str).isin(games)]
    if periods:
        lo, hi = periods
        per = pd.to_numeric(f["period"], errors="coerce")
        f = f[(per >= lo) & (per <= hi)]

    # translate UI shot selection into filter
    if not shot_types:
        shot_types = ["2PT", "3PT"]
    f = f[f["shot_type"].isin(shot_types)]
    return f

# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def _agg_players_passers(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty: return pd.DataFrame()
    g = (f.groupby(["assist_player", "team_name", "shot_type"])
           .agg(makes=("actionNumber","count"),
                points=("points","sum"),
                finishers=("finisher", pd.Series.nunique),
                games=("game_id", pd.Series.nunique))
           .reset_index())
    return g

def _agg_players_finishers(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty: return pd.DataFrame()
    g = (f.groupby(["finisher", "team_name", "shot_type"])
           .agg(makes=("actionNumber","count"),
                points=("points","sum"),
                passers=("assist_player", pd.Series.nunique),
                games=("game_id", pd.Series.nunique))
           .reset_index())
    return g

def _agg_teams(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty: return pd.DataFrame()
    g = (f.groupby(["team_name", "shot_type"])
           .agg(makes=("actionNumber","count"),
                points=("points","sum"),
                passers=("assist_player", pd.Series.nunique),
                finishers=("finisher", pd.Series.nunique),
                games=("game_id", pd.Series.nunique))
           .reset_index())
    return g

# helper to collapse 2PT/3PT to "ALL" (when user wants)
def _collapse_to_all(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if df.empty: return df
    base = df.copy()
    base["shot_type"] = "ALL"
    return (base.groupby(keys + ["shot_type"])
                .agg(makes=("makes","sum"),
                     points=("points","sum"),
                     games=("games","sum"),
                     **({ "finishers": ("finishers","sum")} if "finishers" in base.columns else {}),
                     **({ "passers": ("passers","sum")} if "passers" in base.columns else {}))
                .reset_index())

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
LEAD_MODES = [
    {"label":"Players – Passers", "value":"players_passers"},
    {"label":"Players – Finishers", "value":"players_finishers"},
    {"label":"Teams", "value":"teams"},
]

def _page_content(df: pd.DataFrame) -> html.Div:
    # dropdown options
    team_opts = [{"label": t, "value": t} for t in sorted([t for t in df["team_name"].dropna().unique()])]
    _games_view = df[["week","game_id","label"]].copy()
    _games_view["week"] = _games_view["week"].fillna("")
    _games_view["game_id"] = _games_view["game_id"].astype(str).fillna("")
    _games_view["label"] = _games_view["label"].fillna("")
    game_opts = [{"label": f'{w} • {gid} • {lab}', "value": str(gid)}
                 for w, gid, lab in _games_view.drop_duplicates().sort_values(["week","game_id"]).itertuples(index=False)]

    period_max = int(pd.to_numeric(df["period"], errors="coerce").max() or 4)

    columns_common = [
        {"name":"Shot", "id":"shot_type"},
        {"name":"Makes", "id":"makes", "type":"numeric"},
        {"name":"Points", "id":"points", "type":"numeric"},
        {"name":"Games", "id":"games", "type":"numeric"},
    ]

    # The table will be dynamically reconfigured in callback, but we declare a superset
    return html.Div(children=[
        html.H2("Leaderboards", style={"marginBottom":"8px"}),
        html.P("Rank teams and players on assisted scoring. Toggle modes, filter by shot type (ALL / 2PT / 3PT), teams, games, periods.", style={"color":"#555"}),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr 1fr","gap":"12px"}, children=[
            html.Div(children=[html.Label("Mode"), dcc.RadioItems(id="lb-mode", options=LEAD_MODES, value="players_passers",
                                                                 inputStyle={"marginRight":"6px","marginLeft":"12px"})]),
            html.Div(children=[html.Label("Team(s)"), dcc.Dropdown(id="lb-teams", options=team_opts, multi=True, placeholder="Select teams")]),
            html.Div(children=[html.Label("Game(s)"), dcc.Dropdown(id="lb-games", options=game_opts, multi=True, placeholder="Filter by game")]),
            html.Div(children=[html.Label("Shot Type"),
                dcc.Checklist(id="lb-shot-types",
                    options=[{"label":"All","value":"ALL"},{"label":"2PT","value":"2PT"},{"label":"3PT","value":"3PT"}],
                    value=["ALL"], inputStyle={"marginRight":"6px","marginLeft":"12px"})]),
        ]),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr 1fr","gap":"12px","marginTop":"6px"}, children=[
            html.Div(children=[html.Label("Periods"),
                dcc.RangeSlider(id="lb-periods", min=1, max=period_max, step=1, value=[1, period_max],
                                marks={i:str(i) for i in range(1, period_max+1)})]),
            html.Div(children=[html.Label("Rank by"),
                dcc.RadioItems(id="lb-rank-by",
                    options=[{"label":"Makes","value":"makes"},{"label":"Points (2/3 weighted)","value":"points"}],
                    value="points", inputStyle={"marginRight":"6px","marginLeft":"12px"})]),
            html.Div(children=[html.Label("Min makes"),
                dcc.Slider(id="lb-min-makes", min=1, max=20, step=1, value=3, marks={1:"1",5:"5",10:"10",15:"15",20:"20"})]),
            html.Div(children=[html.Label("Search"),
                dcc.Input(id="lb-search", type="text", placeholder="search player/team", style={"width":"100%"})]),
        ]),

        html.Hr(),

        html.Div(id="lb-kpis", style={"display":"grid","gridTemplateColumns":"repeat(4, minmax(140px, 1fr))","gap":"12px"}),

        html.H3("Leaderboard", style={"marginTop":"16px"}),

        dash_table.DataTable(
            id="lb-table",
            columns=columns_common,
            data=[],
            sort_action="native",
            filter_action="native",
            page_size=20,
            style_table={"overflowX":"auto"},
            style_header={"fontWeight":"600"},
            style_cell={"padding":"6px 8px"},
            export_format="csv",
            export_headers="display",
        ),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr","gap":"16px","marginTop":"18px"}, children=[
            dcc.Graph(id="lb-bars"),
        ]),

        html.Div(style={"marginTop":"12px","fontSize":"12px","color":"#666"}, children=[
            html.Span("Events = assisted, made field goals (P2/P3). 'ALL' collapses 2PT+3PT without double counting.")
        ])
    ])

# ---------------------------------------------------------------------------
# Dash factory
# ---------------------------------------------------------------------------
def create_dash_leaderboards(server, base_pathname: str = "/leaderboards/") -> Dash:
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
        title="Leaderboards",
    )

    df = _pbp()
    players_df = _players()  # optional, for future use (positions, etc.)

    content = _page_content(df)
    nav_items = server.config.get("NAV", [])
    app.layout = PageShell(content, nav_items, current_endpoint=base_pathname)

    # -------------------- Callbacks --------------------
    @app.callback(
        Output("lb-table","data"),
        Output("lb-table","columns"),
        Output("lb-kpis","children"),
        Output("lb-bars","figure"),
        Input("lb-mode","value"),
        Input("lb-teams","value"),
        Input("lb-games","value"),
        Input("lb-shot-types","value"),
        Input("lb-periods","value"),
        Input("lb-rank-by","value"),
        Input("lb-min-makes","value"),
        Input("lb-search","value"),
    )
    def update_view(mode, teams, games, shot_modes, periods, rank_by, min_makes, q):
        df_local = _pbp()
        teams = teams or []
        games = games or []
        periods = periods or [1, int(pd.to_numeric(df_local["period"], errors="coerce").max() or 4)]
        use_points = (rank_by == "points")

        # interpret shot mode into event-level types
        shot_modes = shot_modes or ["ALL"]
        event_types = [s for s in shot_modes if s in ("2PT","3PT")]
        if not event_types:  # only ALL checked
            event_types = ["2PT","3PT"]

        f = _filter_events(df_local, teams=teams, games=games, periods=tuple(periods), shot_types=event_types)

        # search
        if q and q.strip():
            term = q.strip().lower()
            if mode == "players_passers":
                f = f[(f["assist_player"].astype(str).str.lower().str.contains(term, na=False)) |
                      (f["team_name"].astype(str).str.lower().str.contains(term, na=False))]
            elif mode == "players_finishers":
                f = f[(f["finisher"].astype(str).str.lower().str.contains(term, na=False)) |
                      (f["team_name"].astype(str).str.lower().str.contains(term, na=False))]
            else:  # teams
                f = f[f["team_name"].astype(str).str.lower().str.contains(term, na=False)]

        # aggregate by mode
        if mode == "players_passers":
            agg = _agg_players_passers(f)
            key_cols = ["assist_player","team_name"]
            label_col = "assist_player"
            # optionally compute ALL
            frames = [agg]
            if "ALL" in shot_modes:
                frames.append(_collapse_to_all(agg, key_cols))
            agg = pd.concat(frames, ignore_index=True) if frames else agg
            # order & threshold
            if min_makes:
                agg = agg[agg["makes"] >= int(min_makes)]
            agg["value"] = agg["points"] if use_points else agg["makes"]
            agg = agg.sort_values(["value","makes"], ascending=[False,False])

            columns = [
                {"name":"Passer","id":"assist_player"},
                {"name":"Team","id":"team_name"},
                {"name":"Shot","id":"shot_type"},
                {"name":"Makes","id":"makes","type":"numeric"},
                {"name":"Points","id":"points","type":"numeric"},
                {"name":"Unique Finishers","id":"finishers","type":"numeric"},
                {"name":"Games","id":"games","type":"numeric"},
            ]

            # bars
            chart_df = agg.copy().groupby([label_col, "shot_type"], as_index=False).agg(value=("value","sum"))
            chart_df = chart_df.sort_values("value", ascending=False).head(20)
            bars = px.bar(chart_df, x="value", y=label_col, color="shot_type",
                          title=f"Top Passers by {'Points' if use_points else 'Makes'}",
                          orientation="h")

        elif mode == "players_finishers":
            agg = _agg_players_finishers(f)
            key_cols = ["finisher","team_name"]
            label_col = "finisher"
            frames = [agg]
            if "ALL" in shot_modes:
                frames.append(_collapse_to_all(agg, key_cols))
            agg = pd.concat(frames, ignore_index=True) if frames else agg
            if min_makes:
                agg = agg[agg["makes"] >= int(min_makes)]
            agg["value"] = agg["points"] if use_points else agg["makes"]
            agg = agg.sort_values(["value","makes"], ascending=[False,False])

            columns = [
                {"name":"Finisher","id":"finisher"},
                {"name":"Team","id":"team_name"},
                {"name":"Shot","id":"shot_type"},
                {"name":"Makes","id":"makes","type":"numeric"},
                {"name":"Points","id":"points","type":"numeric"},
                {"name":"Unique Passers","id":"passers","type":"numeric"},
                {"name":"Games","id":"games","type":"numeric"},
            ]

            chart_df = agg.copy().groupby([label_col, "shot_type"], as_index=False).agg(value=("value","sum"))
            chart_df = chart_df.sort_values("value", ascending=False).head(20)
            bars = px.bar(chart_df, x="value", y=label_col, color="shot_type",
                          title=f"Top Finishers by {'Points' if use_points else 'Makes'}",
                          orientation="h")

        else:  # teams
            agg = _agg_teams(f)
            key_cols = ["team_name"]
            label_col = "team_name"
            frames = [agg]
            if "ALL" in shot_modes:
                frames.append(_collapse_to_all(agg, key_cols))
            agg = pd.concat(frames, ignore_index=True) if frames else agg
            if min_makes:
                agg = agg[agg["makes"] >= int(min_makes)]
            agg["value"] = agg["points"] if use_points else agg["makes"]
            agg = agg.sort_values(["value","makes"], ascending=[False,False])

            columns = [
                {"name":"Team","id":"team_name"},
                {"name":"Shot","id":"shot_type"},
                {"name":"Makes","id":"makes","type":"numeric"},
                {"name":"Points","id":"points","type":"numeric"},
                {"name":"Unique Passers","id":"passers","type":"numeric"},
                {"name":"Unique Finishers","id":"finishers","type":"numeric"},
                {"name":"Games","id":"games","type":"numeric"},
            ]

            chart_df = agg.copy().groupby([label_col, "shot_type"], as_index=False).agg(value=("value","sum"))
            chart_df = chart_df.sort_values("value", ascending=False).head(20)
            bars = px.bar(chart_df, x="value", y=label_col, color="shot_type",
                          title=f"Top Teams by {'Points' if use_points else 'Makes'}",
                          orientation="h")

        # KPIs from event-level filtered data (no double-count)
        total_makes = int(len(f)) if not f.empty else 0
        total_points = int(f["points"].sum()) if "points" in f.columns and not f.empty else 0
        n_games = int(f["game_id"].nunique()) if not f.empty else 0
        kpis = [
            _kpi("Filtered Makes (events)", f"{total_makes:,}"),
            _kpi("Filtered Points (events)", f"{total_points:,}"),
            _kpi("Games Covered", f"{n_games:,}"),
            _kpi("Rows in Table", f"{len(agg):,}"),
        ]

        # table rows
        rows = agg[ [c["id"] for c in columns] ].to_dict("records")
        return rows, columns, kpis, bars

    return app

def _kpi(title: str, value: str):
    return html.Div(className="kpi", children=[
        html.Div(title, style={"fontSize":"12px","color":"#667"}),
        html.Div(value, style={"fontSize":"20px","fontWeight":700,"marginTop":"2px"}),
    ])
