# dashapps/possessions.py
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


@lru_cache(maxsize=1)
def _team_name_map() -> Dict[str, str]:
    mapping_path = _data_root() / "_cumulative" / "team_name_map.csv"
    if not mapping_path.exists():
        return {}
    try:
        df = pd.read_csv(mapping_path)
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        old = str(row.get("old_name") or "").strip()
        new = str(row.get("canonical_name") or "").strip()
        if old:
            mapping[old] = new or old
    return mapping


def _canonical_team(name: Optional[str]) -> Optional[str]:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    cleaned = str(name).strip()
    if not cleaned:
        return None
    return _team_name_map().get(cleaned, cleaned)


@lru_cache(maxsize=1)
def _load_games_index() -> pd.DataFrame:
    index_path = _data_root() / "_cumulative" / "games_index.csv"
    if not index_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(index_path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.copy()
    df["week"] = df["week"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    teams_split = df.get("teams").astype(str).str.split(" / ", n=1, expand=True)
    if teams_split is not None and teams_split.shape[1] == 2:
        df["team1_name"] = teams_split[0].apply(_canonical_team)
        df["team2_name"] = teams_split[1].apply(_canonical_team)
    else:
        df["team1_name"] = None
        df["team2_name"] = None

    df["label"] = df.get("label").astype(str).fillna("")
    return df[["week", "game_id", "label", "team1_name", "team2_name"]]


def _split_label(label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(label, str) or not label:
        return None, None
    for delimiter in (" vs ", " v ", " - "):
        if delimiter in label:
            parts = label.split(delimiter, 1)
            return parts[0].strip(), parts[1].strip()
    return label, None


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
    pbp_path = _data_root() / "_cumulative" / "pbp_all.csv"
    if not pbp_path.exists():
        return pd.DataFrame()

    try:
        pbp = pd.read_csv(pbp_path)
    except Exception:
        return pd.DataFrame()

    if pbp.empty:
        return pd.DataFrame()

    pbp = pbp.copy()
    pbp["week"] = pbp.get("week").astype(str)
    pbp["game_id"] = pbp.get("game_id").astype(str)
    pbp["label"] = pbp.get("label").astype(str)

    pbp["possession_id"] = pd.to_numeric(pbp.get("possession_id"), errors="coerce")
    pbp["possession_owner_tno"] = pd.to_numeric(pbp.get("possession_owner_tno"), errors="coerce")
    pbp = pbp.dropna(subset=["possession_id", "possession_owner_tno"])
    if pbp.empty:
        return pd.DataFrame()

    pbp["possession_id"] = pbp["possession_id"].astype(int)
    pbp["possession_owner_tno"] = pbp["possession_owner_tno"].astype(int)
    pbp = pbp[pbp["possession_owner_tno"].isin([1, 2])]
    if pbp.empty:
        return pd.DataFrame()

    for col in ("pts_1", "pts_2"):
        pbp[col] = pd.to_numeric(pbp.get(col), errors="coerce").fillna(0.0)

    group_cols = ["week", "game_id", "possession_id", "possession_owner_tno"]
    possessions = (
        pbp.groupby(group_cols, dropna=False)
        .agg(
            pts_1=("pts_1", "sum"),
            pts_2=("pts_2", "sum"),
        )
        .reset_index()
    )

    if possessions.empty:
        return pd.DataFrame()

    games_index = _load_games_index()
    if games_index.empty:
        games_index = (
            pbp[["week", "game_id", "label"]]
            .dropna(subset=["week", "game_id"])
            .drop_duplicates(subset=["week", "game_id"], keep="first")
        )
        split_values = games_index["label"].apply(_split_label)
        games_index["team1_name"] = [
            _canonical_team(pair[0]) for pair in split_values
        ]
        games_index["team2_name"] = [
            _canonical_team(pair[1]) for pair in split_values
        ]
    else:
        if "label" not in games_index.columns:
            games_index = games_index.copy()
            games_index["label"] = ""

        missing_mask = games_index["team1_name"].isna() | games_index["team2_name"].isna()
        if missing_mask.any():
            pbp_labels = (
                pbp[["week", "game_id", "label"]]
                .dropna(subset=["week", "game_id"])
                .drop_duplicates(subset=["week", "game_id"], keep="first")
            )
            games_index = games_index.merge(
                pbp_labels,
                on=["week", "game_id"],
                how="left",
                suffixes=("", "_pbp"),
            )
            games_index["label"] = games_index["label"].where(
                games_index["label"].astype(bool), games_index["label_pbp"]
            )
            games_index = games_index.drop(columns=[c for c in games_index.columns if c.endswith("_pbp")])
            missing_mask = games_index["team1_name"].isna() | games_index["team2_name"].isna()
            split_values = games_index.loc[missing_mask, "label"].apply(_split_label)
            games_index.loc[missing_mask, "team1_name"] = [
                _canonical_team(pair[0]) for pair in split_values
            ]
            games_index.loc[missing_mask, "team2_name"] = [
                _canonical_team(pair[1]) for pair in split_values
            ]

    games_index["team1_name"] = games_index["team1_name"].apply(_canonical_team)
    games_index["team2_name"] = games_index["team2_name"].apply(_canonical_team)
    games_index = games_index.drop_duplicates(subset=["week", "game_id"], keep="first")

    possessions = possessions.merge(
        games_index,
        on=["week", "game_id"],
        how="left",
    )

    if possessions[["team1_name", "team2_name"]].isna().all(axis=None):
        return pd.DataFrame()

    poss_records = possessions.copy()
    poss_records["team_name"] = np.where(
        poss_records["possession_owner_tno"] == 1,
        poss_records["team1_name"],
        poss_records["team2_name"],
    )
    poss_records["opponent"] = np.where(
        poss_records["possession_owner_tno"] == 1,
        poss_records["team2_name"],
        poss_records["team1_name"],
    )

    poss_records["pts_for"] = np.where(
        poss_records["possession_owner_tno"] == 1,
        poss_records["pts_1"],
        poss_records["pts_2"],
    )
    poss_records["pts_against"] = np.where(
        poss_records["possession_owner_tno"] == 1,
        poss_records["pts_2"],
        poss_records["pts_1"],
    )

    poss_records = poss_records.dropna(subset=["team_name", "opponent"])
    if poss_records.empty:
        return pd.DataFrame()

    poss_records["team_name"] = poss_records["team_name"].apply(_canonical_team)
    poss_records["opponent"] = poss_records["opponent"].apply(_canonical_team)

    poss_records["possessions"] = 1

    offense = (
        poss_records.groupby(["week", "game_id", "label", "team_name", "opponent"], dropna=False)
        .agg(
            poss_for=("possessions", "sum"),
            pts_for=("pts_for", "sum"),
        )
        .reset_index()
    )

    if offense.empty:
        return pd.DataFrame()

    defense = (
        offense[["week", "game_id", "team_name", "poss_for", "pts_for"]]
        .rename(
            columns={
                "team_name": "opponent",
                "poss_for": "poss_against",
                "pts_for": "pts_against",
            }
        )
    )

    games = offense.merge(
        defense,
        on=["week", "game_id", "opponent"],
        how="left",
    )

    games["poss_against"] = games["poss_against"].fillna(0)
    games["pts_against"] = games["pts_against"].fillna(0)

    for col in ("poss_for", "poss_against"):
        games[col] = games[col].astype(float).round().astype(int)
    for col in ("pts_for", "pts_against"):
        games[col] = games[col].astype(float).round().astype(int)

    games["week_num"] = games["week"].apply(_extract_week_number)
    games["poss_diff"] = games["poss_for"] - games["poss_against"]
    games["ortg"] = [
        _safe_rating(row["pts_for"], row["poss_for"])
        for _, row in games.iterrows()
    ]
    games["drtg"] = [
        _safe_rating(row["pts_against"], row["poss_against"])
        for _, row in games.iterrows()
    ]
    games["net"] = [
        _safe_net(row["ortg"], row["drtg"])
        for _, row in games.iterrows()
    ]
    games["margin"] = games["pts_for"] - games["pts_against"]
    games["result"] = [
        _result_text(row["pts_for"], row["pts_against"], int(round(row["margin"])))
        for _, row in games.iterrows()
    ]
    games["score_text"] = games.apply(
        lambda r: f"{int(round(r['pts_for']))}-{int(round(r['pts_against']))}",
        axis=1,
    )

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

    games = games.rename(columns={"label": "game_label"})

    columns = [
        "week",
        "week_num",
        "week_display",
        "game_id",
        "game_label",
        "team_name",
        "opponent",
        "poss_for",
        "poss_against",
        "poss_diff",
        "pts_for",
        "pts_against",
        "ortg",
        "drtg",
        "net",
        "margin",
        "result",
        "score_text",
        "game_number",
        "axis_label",
    ]
    return games[columns]


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
        _safe_rating(row["pts_against"], row["poss_against"])
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
            drtg = _safe_rating(tot_pts_against, tot_poss_against) if tot_poss_against else None
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
