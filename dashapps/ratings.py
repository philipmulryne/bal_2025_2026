# dashapps/ratings.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import dash
from dash import Dash, html, dcc, Input, Output, State
from dash.dash_table import DataTable
import plotly.graph_objects as go

try:
    from .common import page_shell as PageShell  # optional wrapper used across your apps
except Exception:
    PageShell = None

# =============================================================================
# IO Helpers
# =============================================================================

_DEF_DATA_ROOT = Path(os.environ.get("DATA_ROOT") or "out_data")
_CUM = _DEF_DATA_ROOT / "_cumulative"
_POSSESS_FILE = _CUM / "games_possessions_exact.csv"
_TIMELINE_DIR = _CUM / "possessions_timeline"


def _file_exists(p: Path) -> bool:
    return p.exists() and p.is_file()


def _load_possessions() -> pd.DataFrame:
    """Load cumulative game-level possessions/ratings.

    Expected columns (produced by possessions_exact):
      week, game_id, team_code, team_name, team_id, team_side,
      possessions, points, opp_points, ortg, drtg
    """
    if not _file_exists(_POSSESS_FILE):
        return pd.DataFrame()

    df = pd.read_csv(_POSSESS_FILE)

    # Normalize column names a bit
    cols = {c.lower(): c for c in df.columns}
    # Coerce numeric
    for k in ["possessions", "points", "opp_points", "ortg", "drtg"]:
        col = cols.get(k)
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure text fields
    for k in ["team_code", "team_name", "week", "game_id"]:
        col = cols.get(k)
        if col:
            df[col] = df[col].astype(str)

    # Add NET and a sortable week index (assumes labels like week01)
    df["net"] = df[cols.get("ortg", "ortg")] - df[cols.get("drtg", "drtg")]
    week_col = cols.get("week", "week")
    if week_col not in df.columns:
        df["week"] = "adhoc"
        week_col = "week"

    def _wk_to_int(s: str) -> int:
        # supports "week01", "wk1", or plain integers in the string
        import re
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else 0

    df["week_order"] = df[week_col].map(_wk_to_int).astype(int)
    return df


def _load_timeline_for(game_id: str) -> pd.DataFrame:
    p = _TIMELINE_DIR / f"{game_id}.csv"
    if not _file_exists(p):
        return pd.DataFrame()
    tl = pd.read_csv(p)
    # Coerce types if present
    for c in ["poss_id", "team_points_in_poss", "opp_points_in_poss", "period_start", "period_end"]:
        if c in tl.columns:
            tl[c] = pd.to_numeric(tl[c], errors="coerce")
    return tl


# =============================================================================
# Metrics / Aggregations
# =============================================================================

_DEF_METRIC = "Net"  # choices: Offense, Defense, Net, Possessions


def _aggregate_by_team(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["team_code", "team_name"], dropna=False).agg(
        games=("game_id", "nunique"),
        possessions=("possessions", "sum"),
        points=("points", "sum"),
        opp_points=("opp_points", "sum"),
    ).reset_index()
    g["ortg"] = 100.0 * g["points"] / g["possessions"].replace(0, np.nan)
    g["drtg"] = 100.0 * g["opp_points"] / g["possessions"].replace(0, np.nan)
    g["net"] = g["ortg"] - g["drtg"]
    return g


def _aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["week", "week_order", "team_code", "team_name"], dropna=False).agg(
        possessions=("possessions", "sum"),
        points=("points", "sum"),
        opp_points=("opp_points", "sum"),
        games=("game_id", "nunique"),
    ).reset_index()
    g["ortg"] = 100.0 * g["points"] / g["possessions"].replace(0, np.nan)
    g["drtg"] = 100.0 * g["opp_points"] / g["possessions"].replace(0, np.nan)
    g["net"] = g["ortg"] - g["drtg"]
    return g.sort_values(["week_order", "team_code"])  # stable


# =============================================================================
# Dash App Factory
# =============================================================================


def create_dash_ratings(server, base_pathname: str = "/ratings/") -> Dash:
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
    )

    def _wrap(children):
        if PageShell:
            return PageShell(children, title="Ratings", subtitle="Exact possessions → ORtg / DRtg / Net", icon="📈")
        return html.Div(children)

    # Initial in-memory load (callbacks will reload on demand)
    df0 = _load_possessions()
    teams0 = sorted(df0["team_code"].unique().tolist()) if not df0.empty else []
    weeks0 = sorted(df0["week"].unique().tolist(), key=lambda s: (len(s), s)) if not df0.empty else []

    app.layout = _wrap([
        html.Div(
            style={"maxWidth": "1100px", "margin": "14px auto", "padding": "0 8px"},
            children=[
                html.Div([
                    html.H2("Team Ratings (Exact Possessions)", style={"marginBottom": 0}),
                    html.Div(
                        "Filter by team and week. Explore per-game tables, weekly trends, and game detail timelines.",
                        style={"opacity": 0.7, "marginTop": 4},
                    ),
                ]),

                # Stores
                dcc.Store(id="rat-data-version", data=str(_POSSESS_FILE.stat().st_mtime) if _file_exists(_POSSESS_FILE) else "0"),

                # Filters Row
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "10px", "marginTop": "12px"},
                    children=[
                        html.Div([
                            html.Label("Teams"),
                            dcc.Dropdown(
                                id="rat-teams",
                                options=[{"label": t, "value": t} for t in teams0],
                                value=teams0[:],
                                multi=True,
                                placeholder="All teams",
                            ),
                        ]),
                        html.Div([
                            html.Label("Weeks"),
                            dcc.Dropdown(
                                id="rat-weeks",
                                options=[{"label": w, "value": w} for w in weeks0],
                                value=weeks0[:],
                                multi=True,
                                placeholder="All weeks",
                            ),
                        ]),
                        html.Div([
                            html.Label("Metric"),
                            dcc.Dropdown(
                                id="rat-metric",
                                options=[
                                    {"label": "Net Rating", "value": "Net"},
                                    {"label": "Offensive Rating", "value": "Offense"},
                                    {"label": "Defensive Rating", "value": "Defense"},
                                    {"label": "Possessions", "value": "Possessions"},
                                ],
                                value=_DEF_METRIC,
                                clearable=False,
                            ),
                        ]),
                        html.Div([
                            html.Label("Show"),
                            dcc.Dropdown(
                                id="rat-view",
                                options=[
                                    {"label": "Season Aggregate (by Team)", "value": "season"},
                                    {"label": "Weekly Trend (per Team)", "value": "weekly"},
                                    {"label": "Per-Game Table", "value": "table"},
                                    {"label": "Game Detail (timeline)", "value": "game"},
                                ],
                                value="season",
                                clearable=False,
                            ),
                        ]),
                    ],
                ),

                html.Div(id="rat-kpis", style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "10px", "marginTop": "12px"}),

                html.Div(
                    id="rat-panels",
                    children=[
                        dcc.Graph(id="rat-chart-main"),
                        html.Div(style={"height": "8px"}),
                        dcc.Graph(id="rat-chart-secondary"),
                        html.Div(style={"height": "8px"}),
                        DataTable(
                            id="rat-table",
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_cell={"whiteSpace": "nowrap", "fontSize": 12},
                            sort_action="native",
                            filter_action="native",
                            columns=[],
                            data=[],
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div([
                            html.Label("Select Game (for Game Detail)"),
                            dcc.Dropdown(id="rat-game", options=[], value=None, placeholder="Pick a game_id"),
                        ], id="rat-game-picker", style={"display": "none", "marginTop": "8px"}),
                        html.Div(id="rat-game-summary"),
                        DataTable(
                            id="rat-timeline",
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_cell={"whiteSpace": "nowrap", "fontSize": 12},
                            sort_action="native",
                            filter_action="native",
                            columns=[],
                            data=[],
                        ),
                    ],
                ),

                html.Div(id="rat-status", style={"opacity": 0.75, "marginTop": "10px"}),
            ],
        )
    ])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @app.callback(
        Output("rat-teams", "options"),
        Output("rat-teams", "value"),
        Output("rat-weeks", "options"),
        Output("rat-weeks", "value"),
        Output("rat-status", "children"),
        Input("rat-data-version", "data"),
        prevent_initial_call=False,
    )
    def _refresh_filter_options(_version):
        df = _load_possessions()
        if df.empty:
            return [], [], [], [], f"No possessions file found → {_POSSESS_FILE}"
        teams = sorted(df["team_code"].dropna().unique().tolist())
        weeks = df["week"].dropna().unique().tolist()
        # sort like week01, week02, ... if possible
        weeks = sorted(weeks, key=lambda s: (len(s), s))
        return (
            [{"label": t, "value": t} for t in teams], teams[:],
            [{"label": w, "value": w} for w in weeks], weeks[:],
            f"Loaded {len(df):,} rows from {_POSSESS_FILE}"
        )

    @app.callback(
        Output("rat-kpis", "children"),
        Output("rat-chart-main", "figure"),
        Output("rat-chart-secondary", "figure"),
        Output("rat-table", "columns"),
        Output("rat-table", "data"),
        Output("rat-game-picker", "style"),
        Output("rat-game", "options"),
        Output("rat-game", "value"),
        Input("rat-teams", "value"),
        Input("rat-weeks", "value"),
        Input("rat-metric", "value"),
        Input("rat-view", "value"),
    )
    def _update_main(teams_sel, weeks_sel, metric, view):
        df = _load_possessions()
        if df.empty:
            fig_empty = go.Figure()
            return [], fig_empty, fig_empty, [], [], {"display": "none"}, [], None

        # Filter
        if teams_sel:
            df = df[df["team_code"].isin(teams_sel)]
        if weeks_sel:
            df = df[df["week"].isin(weeks_sel)]

        # KPIs (aggregate over selection)
        agg = _aggregate_by_team(df)
        tot_games = int(agg["games"].sum()) if not agg.empty else 0
        tot_poss = float(agg["possessions"].sum()) if not agg.empty else 0.0
        w_mean_ortg = float(np.average(agg["ortg"], weights=agg["possessions"])) if not agg.empty else np.nan
        w_mean_drtg = float(np.average(agg["drtg"], weights=agg["possessions"])) if not agg.empty else np.nan
        w_mean_net = w_mean_ortg - w_mean_drtg if np.isfinite(w_mean_ortg) and np.isfinite(w_mean_drtg) else np.nan

        def _kpi(label: str, value: str) -> html.Div:
            return html.Div(
                [html.Div(label, style={"fontSize": 12, "opacity": 0.7}), html.Div(value, style={"fontSize": 20, "fontWeight": 700})],
                style={"border": "1px solid var(--border, rgba(255,255,255,0.12))", "borderRadius": "10px", "padding": "10px"},
            )

        kpis = [
            _kpi("Teams", f"{len(agg):,}"),
            _kpi("Games", f"{tot_games:,}"),
            _kpi("Possessions (sum)", f"{tot_poss:,.0f}"),
            _kpi("Weighted ORtg", f"{w_mean_ortg:0.2f}" if np.isfinite(w_mean_ortg) else "–"),
            _kpi("Weighted Net", f"{w_mean_net:0.2f}" if np.isfinite(w_mean_net) else "–"),
        ]

        # View routing
        metric_map = {
            "Offense": "ortg",
            "Defense": "drtg",
            "Net": "net",
            "Possessions": "possessions",
        }
        mcol = metric_map.get(metric, "net")

        # --- Prepare outputs defaults
        fig1 = go.Figure(); fig2 = go.Figure()
        table_cols: List[dict] = []
        table_data: List[dict] = []
        game_picker_style = {"display": "none"}
        game_options: List[dict] = []
        game_value = None

        if view == "season":
            # Rank teams by metric
            if agg.empty:
                return kpis, fig1, fig2, table_cols, table_data, game_picker_style, game_options, game_value

            top = agg.sort_values(mcol, ascending=(mcol == "drtg")).head(20)

            # Bar: chosen metric by team
            fig1 = go.Figure()
            fig1.add_bar(x=top["team_code"], y=top[mcol])
            fig1.update_layout(title=f"Top teams – {metric}", xaxis_title="Team", yaxis_title=metric)

            # Scatter: ORtg vs DRtg (lower-left is elite defense)
            fig2 = go.Figure()
            fig2.add_scatter(x=agg["drtg"], y=agg["ortg"], mode="markers", text=agg["team_code"], hovertemplate="Team=%{text}<br>ORtg=%{y:.1f}<br>DRtg=%{x:.1f}<extra></extra>")
            fig2.update_layout(title="ORtg vs DRtg (season aggregate)", xaxis_title="DRtg (lower is better)", yaxis_title="ORtg (higher is better)")

            # Table: aggregates
            show = agg.copy()
            show = show[["team_code", "team_name", "games", "possessions", "ortg", "drtg", "net"]].sort_values("net", ascending=False)
            show["possessions"] = show["possessions"].round(0).astype(int)
            show["ortg"] = show["ortg"].round(2)
            show["drtg"] = show["drtg"].round(2)
            show["net"] = show["net"].round(2)
            table_cols = [{"name": c.upper(), "id": c} for c in show.columns]
            table_data = show.to_dict("records")

        elif view == "weekly":
            wk = _aggregate_weekly(df)
            if wk.empty:
                return kpis, fig1, fig2, table_cols, table_data, game_picker_style, game_options, game_value

            # Line: metric by week per team (selected teams)
            fig1 = go.Figure()
            for tm, g in wk.groupby("team_code"):
                fig1.add_trace(go.Scatter(x=g["week_order"], y=g[mcol], mode="lines+markers", name=tm))
            fig1.update_layout(title=f"Weekly {metric} (by team)", xaxis_title="Week #", yaxis_title=metric)

            # Secondary: stacked bars of possessions per team per week (sum)
            wk_poss = wk.groupby(["week_order", "team_code"], as_index=False)["possessions"].sum()
            fig2 = go.Figure()
            for tm, g in wk_poss.groupby("team_code"):
                fig2.add_bar(x=g["week_order"], y=g["possessions"], name=tm)
            fig2.update_layout(barmode="stack", title="Weekly Possessions (stacked)", xaxis_title="Week #", yaxis_title="Possessions")

            # Table: weekly
            show = wk[["week", "team_code", "games", "possessions", "ortg", "drtg", "net"]].copy()
            show["possessions"] = show["possessions"].round(0).astype(int)
            for c in ["ortg", "drtg", "net"]:
                show[c] = show[c].round(2)
            table_cols = [{"name": c.upper(), "id": c} for c in show.columns]
            table_data = show.sort_values(["week", "net"], ascending=[True, False]).to_dict("records")

        elif view == "table":
            # Per-game rows already in df
            show = df[["week", "game_id", "team_code", "team_name", "possessions", "points", "opp_points", "ortg", "drtg", "net"]].copy()
            show["possessions"] = show["possessions"].round(0).astype(int)
            for c in ["ortg", "drtg", "net"]:
                show[c] = show[c].round(2)
            table_cols = [{"name": c.upper(), "id": c} for c in show.columns]
            table_data = show.sort_values(["week", "game_id", "team_code"]).to_dict("records")

            # Simple chart: histogram of Net
            fig1 = go.Figure()
            fig1.add_histogram(x=show["net"], nbinsx=30)
            fig1.update_layout(title="Distribution of Net Ratings (per-team per-game)", xaxis_title="Net Rating", yaxis_title="Count")

            fig2 = go.Figure()

        elif view == "game":
            # Provide game selector and summary (2 rows)
            game_picker_style = {"display": "block", "marginTop": "8px"}
            # Only show games present in the filtered set
            game_ids = df["game_id"].dropna().unique().tolist()
            game_ids_sorted = sorted(game_ids, key=lambda s: (len(s), s))
            game_options = [{"label": gid, "value": gid} for gid in game_ids_sorted]
            game_value = game_ids_sorted[0] if game_ids_sorted else None

            fig1 = go.Figure(); fig2 = go.Figure()

        return kpis, fig1, fig2, table_cols, table_data, game_picker_style, game_options, game_value

    @app.callback(
        Output("rat-game-summary", "children"),
        Output("rat-timeline", "columns"),
        Output("rat-timeline", "data"),
        Input("rat-game", "value"),
        Input("rat-teams", "value"),  # to refresh when filters change
        Input("rat-weeks", "value"),
        Input("rat-view", "value"),
        prevent_initial_call=True,
    )
    def _update_game_detail(game_id, _t, _w, view):
        if view != "game" or not game_id:
            return html.Div(), [], []
        df = _load_possessions()
        if df.empty:
            return html.Div("No data"), [], []
        g = df[df["game_id"].astype(str) == str(game_id)].copy()
        if g.empty:
            return html.Div(f"No rows for game_id={game_id}"), [], []

        # Summary (2 rows)
        show = g[["team_code", "team_name", "possessions", "points", "opp_points", "ortg", "drtg", "net"]].copy()
        for c in ["possessions", "points", "opp_points"]:
            show[c] = pd.to_numeric(show[c], errors="coerce")
        for c in ["ortg", "drtg", "net"]:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(2)

        summary_table = DataTable(
            columns=[{"name": c.upper(), "id": c} for c in show.columns],
            data=show.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"whiteSpace": "nowrap", "fontSize": 12},
        )

        # Timeline
        tl = _load_timeline_for(str(game_id))
        if tl.empty:
            tl_cols = []
            tl_data = []
            tl_note = html.Div("No possession timeline file found for this game.", style={"opacity": 0.7, "marginTop": 6})
        else:
            # show a concise set of columns if available
            keep = [
                "poss_id", "poss_team", "period_start", "period_end",
                "start_row", "end_row", "start_clock", "end_clock",
                "start_event_ac", "end_event_ac", "end_reason",
                "team_points_in_poss", "opp_points_in_poss",
            ]
            cols = [c for c in keep if c in tl.columns]
            tl_cols = [{"name": c.upper(), "id": c} for c in cols]
            tl_data = tl[cols].to_dict("records")
            tl_note = html.Div(style={})

        header = html.H4(f"Game {game_id} – Summary & Possession Timeline", style={"marginTop": "10px"})
        return html.Div([header, summary_table, tl_note]), tl_cols, tl_data

    return app
