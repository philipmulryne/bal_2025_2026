# dashapps/possessions.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme

try:
    from .common import page_shell as PageShell  # type: ignore
except Exception:  # pragma: no cover - defensive import for Dash shell
    PageShell = None


# =============================================================================
# Data discovery helpers
# =============================================================================


def _data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "out_data"))


def _extract_week_number(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _load_meta(game_dir: Path) -> Dict[str, Any]:
    meta_path = game_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_rating(points: float, poss: int) -> Optional[float]:
    if poss and poss > 0:
        try:
            return round(100.0 * float(points) / float(poss), 2)
        except Exception:
            return None
    return None


def _safe_net(ortg: Optional[float], drtg: Optional[float]) -> Optional[float]:
    if ortg is None or drtg is None:
        return None
    return round(ortg - drtg, 2)


def _result_text(pf: float, pa: float, margin: int) -> str:
    pf_i = int(round(pf))
    pa_i = int(round(pa))
    if margin > 0:
        return f"W {pf_i}-{pa_i} (+{margin})"
    if margin < 0:
        return f"L {pf_i}-{pa_i} ({margin})"
    return f"T {pf_i}-{pa_i} (0)"


def _week_display(row: pd.Series) -> Any:
    wk = row.get("week_num")
    if pd.notna(wk):
        try:
            return int(wk)
        except Exception:
            return wk
    return row.get("week")


def _clean_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                rec[col] = float(val)
            else:
                rec[col] = val
        records.append(rec)
    return records


@lru_cache(maxsize=1)
def _load_game_log() -> pd.DataFrame:
    root = _data_root()
    if not root.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    def _game_dirs(par: Path) -> Iterable[Path]:
        try:
            for child in sorted(par.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    yield child
        except Exception:
            return []

    for bucket in _game_dirs(root):
        if bucket.name.startswith("_"):
            continue
        for game_dir in _game_dirs(bucket):
            poss_path = game_dir / "game_possessions.csv"
            if not poss_path.exists():
                continue
            meta = _load_meta(game_dir)
            week_label = meta.get("week") or bucket.name
            week_num = _extract_week_number(week_label)
            game_id = str(meta.get("game_id") or game_dir.name)
            game_label = meta.get("label") or ""

            try:
                poss_df = pd.read_csv(poss_path)
            except Exception:
                continue

            if poss_df is None or poss_df.empty or "owner_team_name" not in poss_df.columns:
                continue

            poss_df = poss_df.dropna(subset=["owner_team_name"]).copy()
            if poss_df.empty:
                continue

            poss_df["owner_team_name"] = poss_df["owner_team_name"].astype(str)
            poss_df["pts_owner"] = pd.to_numeric(poss_df.get("pts_owner"), errors="coerce").fillna(0.0)

            grouped = poss_df.groupby("owner_team_name", dropna=False)
            if grouped.ngroups < 2:
                continue

            summary: List[Dict[str, Any]] = []
            for team_name, grp in grouped:
                poss_for = int(grp.shape[0])
                pts_for = float(grp["pts_owner"].sum())
                summary.append(
                    {
                        "team_name": team_name,
                        "poss_for": poss_for,
                        "pts_for": pts_for,
                    }
                )

            if len(summary) < 2:
                continue

            for entry in summary:
                opp = next((o for o in summary if o["team_name"] != entry["team_name"]), None)
                if opp is None:
                    continue
                poss_for = entry["poss_for"]
                poss_against = opp["poss_for"]
                pts_for = entry["pts_for"]
                pts_against = opp["pts_for"]
                ortg = _safe_rating(pts_for, poss_for)
                drtg = _safe_rating(pts_against, poss_for)
                net = _safe_net(ortg, drtg)
                margin = int(round(pts_for - pts_against))

                rows.append(
                    {
                        "week": week_label,
                        "week_num": week_num,
                        "week_display": None,
                        "game_id": game_id,
                        "game_label": game_label,
                        "team_name": entry["team_name"],
                        "opponent": opp["team_name"],
                        "poss_for": poss_for,
                        "poss_against": poss_against,
                        "poss_diff": poss_for - poss_against,
                        "pts_for": int(round(pts_for)),
                        "pts_against": int(round(pts_against)),
                        "ortg": ortg,
                        "drtg": drtg,
                        "net": net,
                        "margin": margin,
                        "result": _result_text(pts_for, pts_against, margin),
                        "score_text": f"{int(round(pts_for))}-{int(round(pts_against))}",
                    }
                )

    games = pd.DataFrame(rows)
    if games.empty:
        return games

    games["week_display"] = games.apply(_week_display, axis=1)
    games["week_sort"] = pd.to_numeric(games["week_num"], errors="coerce").fillna(9999)
    games["game_sort"] = pd.to_numeric(games["game_id"], errors="coerce").fillna(0)
    games = games.sort_values(["team_name", "week_sort", "game_sort", "opponent"], kind="mergesort")
    games["game_number"] = games.groupby("team_name").cumcount() + 1
    games["axis_label"] = games.apply(
        lambda r: (
            f"W{int(r['week_num']):02d} • {r['opponent']}" if pd.notna(r["week_num"]) else f"{r['week']} • {r['opponent']}"
        ),
        axis=1,
    )
    return games.drop(columns=["week_sort", "game_sort"])


def _cumulative_table(games: pd.DataFrame) -> pd.DataFrame:
    if games is None or games.empty:
        return pd.DataFrame(columns=[
            "team_name",
            "games",
            "poss_for",
            "poss_against",
            "poss_diff",
            "pts_for",
            "pts_against",
            "ortg",
            "drtg",
            "net",
        ])

    agg = (
        games.groupby("team_name", dropna=False)
        .agg(
            games=("game_id", "nunique"),
            poss_for=("poss_for", "sum"),
            poss_against=("poss_against", "sum"),
            pts_for=("pts_for", "sum"),
            pts_against=("pts_against", "sum"),
        )
        .reset_index()
    )

    agg["poss_diff"] = agg["poss_for"] - agg["poss_against"]
    agg["ortg"] = [
        _safe_rating(row["pts_for"], row["poss_for"])
        for _, row in agg.iterrows()
    ]
    agg["drtg"] = [
        _safe_rating(row["pts_against"], row["poss_for"])
        for _, row in agg.iterrows()
    ]
    agg["net"] = [
        _safe_net(row["ortg"], row["drtg"])
        for _, row in agg.iterrows()
    ]
    return agg


# =============================================================================
# Layout
# =============================================================================


def _wrap_with_pageshell(app: Dash, base_pathname: str, content):
    if not PageShell:
        return content

    nav_items = []
    try:
        nav_items = list(getattr(app.server, "config", {}).get("NAV", []))
    except Exception:
        nav_items = []

    def _attempt(func):
        try:
            out = func(content, nav_items, current_endpoint=base_pathname)
        except TypeError:
            try:
                out = func(content, nav_items=nav_items, current_endpoint=base_pathname)
            except Exception:
                return None
        except Exception:
            return html.Div([
                html.H2("Possessions"),
                html.P("Page shell failed; falling back to basic layout."),
                content,
            ])
        return out if out is not None else content

    if callable(PageShell):
        out = _attempt(PageShell)
        if out is not None:
            return out

    for attr in ("page_shell", "build"):
        func = getattr(PageShell, attr, None)
        if callable(func):
            out = _attempt(func)
            if out is not None:
                return out

    return content


def _make_layout(app: Dash, base_pathname: str, teams: List[str], games: pd.DataFrame, cumulative: pd.DataFrame):
    games_records = _clean_records(games)
    cumulative_records = _clean_records(cumulative)

    metric_options = [
        {"label": "Net Rating", "value": "net"},
        {"label": "Offensive Rating", "value": "ortg"},
        {"label": "Defensive Rating", "value": "drtg"},
        {"label": "Possessions For", "value": "poss_for"},
        {"label": "Possessions Against", "value": "poss_against"},
    ]

    layout_core = html.Div(
        className="possession-dashboard",
        children=[
            dcc.Store(id="pos-games-store", data=games_records),
            dcc.Store(id="pos-cumulative-store", data=cumulative_records),
            html.H1("Exact Possessions & Efficiency"),
            html.P(
                "Track possessions for/against plus offensive, defensive, and net ratings "
                "using play-by-play derived possessions."
            ),
            html.Div(
                className="pos-controls",
                children=[
                    html.Div(
                        children=[
                            html.Label("Teams"),
                            dcc.Dropdown(
                                id="pos-team-select",
                                options=[{"label": t, "value": t} for t in teams],
                                value=teams[:1],
                                multi=True,
                                placeholder="Select team(s)",
                                clearable=True,
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Metric"),
                            dcc.RadioItems(
                                id="pos-metric",
                                options=metric_options,
                                value="net",
                                labelStyle={"marginRight": "12px"},
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(id="pos-kpis", className="pos-kpis"),
            html.Div(
                className="pos-graph",
                children=[
                    dcc.Graph(id="pos-metric-graph"),
                ],
            ),
            html.Div(
                className="pos-tables",
                children=[
                    html.Div(
                        children=[
                            html.H3("Cumulative Team Totals"),
                            dash_table.DataTable(
                                id="pos-cumulative-table",
                                columns=[
                                    {"name": "Team", "id": "team_name"},
                                    {"name": "Games", "id": "games", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Poss For", "id": "poss_for", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Poss Against", "id": "poss_against", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Poss Diff", "id": "poss_diff", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Points For", "id": "pts_for", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Points Against", "id": "pts_against", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Off Rtg", "id": "ortg", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                    {"name": "Def Rtg", "id": "drtg", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                    {"name": "Net", "id": "net", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                ],
                                data=cumulative_records,
                                sort_action="native",
                                page_size=18,
                                style_table={"overflowX": "auto"},
                                style_cell={"padding": "6px 8px"},
                                style_header={"fontWeight": "600"},
                                export_format="csv",
                                export_headers="display",
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": "{net} > 0", "column_id": "net"},
                                        "color": "#146c43",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"filter_query": "{net} < 0", "column_id": "net"},
                                        "color": "#b02a37",
                                        "fontWeight": "600",
                                    },
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.H3("Game-by-Game Log"),
                            dash_table.DataTable(
                                id="pos-game-table",
                                columns=[
                                    {"name": "Team", "id": "team_name"},
                                    {"name": "Week", "id": "week_display"},
                                    {"name": "Game #", "id": "game_number", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Game ID", "id": "game_id"},
                                    {"name": "Opponent", "id": "opponent"},
                                    {"name": "Poss For", "id": "poss_for", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Poss Against", "id": "poss_against", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Points For", "id": "pts_for", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Points Against", "id": "pts_against", "type": "numeric", "format": Format(group=",")},
                                    {"name": "Off Rtg", "id": "ortg", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                    {"name": "Def Rtg", "id": "drtg", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                    {"name": "Net", "id": "net", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                                    {"name": "Result", "id": "result"},
                                ],
                                data=games_records,
                                sort_action="native",
                                page_size=20,
                                style_table={"overflowX": "auto"},
                                style_cell={"padding": "6px 8px"},
                                style_header={"fontWeight": "600"},
                                export_format="csv",
                                export_headers="display",
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": "{net} > 0", "column_id": "net"},
                                        "color": "#146c43",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"filter_query": "{net} < 0", "column_id": "net"},
                                        "color": "#b02a37",
                                        "fontWeight": "600",
                                    },
                                ],
                            ),
                        ]
                    ),
                ],
            ),
        ],
    )

    return _wrap_with_pageshell(app, base_pathname, layout_core)


# =============================================================================
# Dash app factory
# =============================================================================


def create_dash_possessions(server=None, base_pathname: str = "/possessions/"):
    app = Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div("Initializing Possessions Dashboard…")

    games = _load_game_log()
    cumulative = _cumulative_table(games)
    teams = sorted(games["team_name"].unique().tolist()) if not games.empty else []

    app.layout = _make_layout(app, base_pathname, teams, games, cumulative)

    @app.callback(
        Output("pos-cumulative-table", "data"),
        Output("pos-game-table", "data"),
        Output("pos-metric-graph", "figure"),
        Output("pos-kpis", "children"),
        Input("pos-team-select", "value"),
        Input("pos-metric", "value"),
        State("pos-games-store", "data"),
        State("pos-cumulative-store", "data"),
    )
    def _update_dashboard(team_values, metric_value, games_data, cumulative_data):
        games_df = pd.DataFrame(games_data or [])
        cum_df = pd.DataFrame(cumulative_data or [])

        selected = team_values or []
        if selected:
            games_filtered = games_df[games_df["team_name"].isin(selected)]
            cum_filtered = cum_df[cum_df["team_name"].isin(selected)]
        else:
            games_filtered = games_df
            cum_filtered = cum_df

        def _kpi_block(title: str, value: str) -> html.Div:
            return html.Div(className="pos-kpi", children=[html.Span(title), html.Strong(value)])

        if cum_filtered.empty:
            kpi_children = html.Div("No data for selection", className="pos-kpi-empty")
        else:
            tot_poss_for = float(cum_filtered["poss_for"].sum())
            tot_poss_against = float(cum_filtered["poss_against"].sum())
            tot_pts_for = float(cum_filtered["pts_for"].sum())
            tot_pts_against = float(cum_filtered["pts_against"].sum())
            agg_games = int(cum_filtered["games"].sum())
            ortg = _safe_rating(tot_pts_for, tot_poss_for) if tot_poss_for else None
            drtg = _safe_rating(tot_pts_against, tot_poss_for) if tot_poss_for else None
            net = _safe_net(ortg, drtg)

            def _fmt_int(val: float) -> str:
                try:
                    return f"{int(round(val)):,}"
                except Exception:
                    return "—"

            def _fmt_float(val: Optional[float]) -> str:
                if val is None:
                    return "—"
                return f"{val:.2f}"

            kpi_children = html.Div(
                className="pos-kpi-grid",
                children=[
                    _kpi_block("Games", f"{agg_games:,}"),
                    _kpi_block("Possessions For", _fmt_int(tot_poss_for)),
                    _kpi_block("Possessions Against", _fmt_int(tot_poss_against)),
                    _kpi_block("Off Rtg", _fmt_float(ortg)),
                    _kpi_block("Def Rtg", _fmt_float(drtg)),
                    _kpi_block("Net", _fmt_float(net)),
                ],
            )

        metric_map = {
            "net": ("Net Rating", "net"),
            "ortg": ("Offensive Rating", "ortg"),
            "drtg": ("Defensive Rating", "drtg"),
            "poss_for": ("Possessions For", "poss_for"),
            "poss_against": ("Possessions Against", "poss_against"),
        }
        metric_label, metric_col = metric_map.get(metric_value, ("Net Rating", "net"))

        if games_filtered.empty:
            fig = px.line()
            fig.add_annotation(text="No games available for selection", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            fig.update_layout(title=f"{metric_label} by Game", xaxis={"visible": False}, yaxis={"visible": False})
        else:
            games_filtered = games_filtered.sort_values(["team_name", "game_number"], kind="mergesort")
            fig = px.line(
                games_filtered,
                x="game_number",
                y=metric_col,
                color="team_name",
                markers=True,
                hover_data={
                    "team_name": False,
                    "game_id": True,
                    "opponent": True,
                    "week_display": True,
                    "result": True,
                    metric_col: ":.2f" if metric_col in {"net", "ortg", "drtg"} else ":.0f",
                },
            )
            fig.update_layout(
                title=f"{metric_label} by Game",
                xaxis_title="Game #",
                yaxis_title=metric_label,
                legend_title="Team",
                hovermode="x unified",
            )

        return (
            _clean_records(cum_filtered),
            _clean_records(games_filtered),
            fig,
            kpi_children,
        )

    return app


__all__ = ["create_dash_possessions"]
