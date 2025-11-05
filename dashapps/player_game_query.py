import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL, no_update
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text as sql_text

# Optional Bootstrap theming (graceful fallback if dbc isn't installed)
try:
    import dash_bootstrap_components as dbc
    BOOTSTRAP = [dbc.themes.FLATLY]  # modern, legible; change theme if you like
    HAVE_DBC = True
except Exception:
    dbc = None
    BOOTSTRAP = []
    HAVE_DBC = False

#########################################################
#                    GLOBALS
#########################################################

DATABASE_URI = "postgresql://philipmulryne@localhost:5432/basketball_data"
engine = create_engine(DATABASE_URI)

def generate_season_table_names(start_season: str, end_season: str, exceptions=None):
    """
    Generates a list of season table names (e.g., processed_games_1985_86, ... )
    from start_season to end_season, skipping any 'exceptions'.
    """
    if exceptions is None:
        exceptions = []
    start_year = int(start_season.split("_")[0])
    end_year = int(end_season.split("_")[0])
    table_names = []
    for year in range(start_year, end_year + 1):
        next_year = str(year + 1)[-2:]
        season_str = f"{year}_{next_year}"
        if season_str not in exceptions:
            table_names.append(f"processed_games_{season_str}")
    return table_names

# All season tables from 1985_86 to 2025_26 (excluding 2019_20, 1986_87)
season_tables = generate_season_table_names(
    "1985_86", "2025_26",
    exceptions=["2019_20", "1986_87"]
)

# --------------------------------------------
# Define filterable columns with metadata
# --------------------------------------------
FILTERABLE_COLUMNS = [
    ("points", "Points", "numeric"),
    ("defensive_rebounds", "Defensive Rebounds", "numeric"),
    ("offensive_rebounds", "Offensive Rebounds", "numeric"),
    ("total_rebounds", "Total Rebounds", "numeric"),
    ("assists", "Assists", "numeric"),
    ("blocks", "Blocks", "numeric"),
    ("turnovers", "Turnovers", "numeric"),
    ("steals", "Steals", "numeric"),
    ("fouls", "Fouls", "numeric"),
    ("efficiency", "Efficiency", "numeric"),
    ("plus_minus", "Plus-Minus", "numeric"),
    ("field_goals_made", "Field Goals Made", "numeric"),
    ("field_goals_attempts", "Field Goals Attempted", "numeric"),
    ("two_points_made", "Two-Point Field Goals Made", "numeric"),
    ("two_points_attempts", "Two-Point Field Goals Attempted", "numeric"),
    ("three_points_made", "Three-Point Field Goals Made", "numeric"),
    ("three_points_attempts", "Three-Point Field Goals Attempted", "numeric"),
    ("free_throws_made", "Free Throws Made", "numeric"),
    ("free_throws_attempts", "Free Throws Attempted", "numeric"),
    ("season", "Season", "text"),
    ("team", "Team", "text"),
    ("game_date", "Game Date", "date"),
    ("player_name", "Player", "text"),
]
COLUMN_OPTIONS = [{"label": label, "value": name} for name, label, ctype in FILTERABLE_COLUMNS]

NUMERIC_OPERATORS = [{"label": s, "value": s} for s in ["=", ">", "<", ">=", "<="]]
TEXT_OPERATORS    = [{"label": "=", "value": "="}, {"label": "LIKE (contains)", "value": "LIKE"}]
DATE_OPERATORS    = [{"label": s, "value": s} for s in ["=", ">", "<", ">=", "<="]]

def _present_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_candidates = [name for (name, _, ctype) in FILTERABLE_COLUMNS if ctype == "numeric"]
    return [c for c in numeric_candidates if c in df.columns]

#########################################################
#                      LAYOUT
#########################################################

def _filter_row(idx: int):
    base_children = [
        dcc.Dropdown(
            id={"type": "pgq-column-dropdown", "index": idx},
            options=COLUMN_OPTIONS,
            placeholder="Column (e.g., Points, Game Date)",
            className="w-100"
        ),
        dcc.Dropdown(
            id={"type": "pgq-operator-dropdown", "index": idx},
            options=[],
            placeholder="Operator",
            className="w-100"
        ),
        dcc.Input(
            id={"type": "pgq-value-input", "index": idx},
            type="text",
            placeholder="Value",
            className="w-100"
        ),
        html.Button(
            "Remove",
            id={"type": "pgq-remove-filter", "index": idx},
            n_clicks=0,
            className="btn btn-outline-danger"
        )
    ]

    if HAVE_DBC:
        return dbc.Row(
            [
                dbc.Col(base_children[0], md=4),
                dbc.Col(base_children[1], md=3),
                dbc.Col(base_children[2], md=4),
                dbc.Col(base_children[3], md=1, className="d-grid"),
            ],
            className="g-2 mb-1"
        )
    else:
        return html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1.2fr 2fr 0.8fr", "gap": "8px", "marginBottom": "6px"},
            children=base_children
        )

def _sidebar():
    controls = [
        html.Div(id="pgq-status", style={"color": "#dc3545", "marginBottom": "6px"}),

        html.Label("Season Range"),
        dcc.RangeSlider(
            id="pgq-season-range-slider",
            min=1985, max=2025, step=1, value=[1985, 2025],
            marks={y: str(y) for y in range(1985, 2026, 5)},
            tooltip={"placement": "bottom"}
        ),
        html.Small(id="pgq-slider-output", className="text-muted"),

        html.Div(className="mt-3"),

        html.Label("Players"),
        dcc.Dropdown(
            id="pgq-player-dropdown",
            options=[], value=[],
            placeholder="Select one or more players",
            multi=True
        ),

        html.Label("Seasons (optional)", className="mt-3"),
        dcc.Dropdown(
            id="pgq-season-dropdown",
            options=[], value=None, multi=True,
            placeholder="Select seasons (default: all)"
        ),

        html.Label("Since Season (optional)", className="mt-3"),
        dcc.Dropdown(
            id="pgq-since-season-dropdown",
            options=[], value=None,
            placeholder="Select a starting season"
        ),

        html.Hr(),

        html.H5("Additional Filters"),
        html.Div(id="pgq-filter-container", children=[_filter_row(0)]),

        html.Div(
            [
                html.Button("Add Filter", id="pgq-add-filter-button", n_clicks=0, className="btn btn-outline-primary"),
                html.Span(" "),
                html.Button("Reset Filters", id="pgq-reset-filters", n_clicks=0, className="btn btn-outline-secondary"),
            ],
            className="mt-1"
        ),

        dcc.Checklist(
            id="pgq-unique-results-checkbox",
            options=[{"label": "Show only unique players", "value": "unique"}],
            value=[],
            className="mt-3"
        ),

        html.Div(
            [
                html.Button("Query", id="pgq-query-button", n_clicks=0, className="btn btn-primary w-100")
            ],
            className="mt-3"
        )
    ]

    if HAVE_DBC:
        return dbc.Card(
            dbc.CardBody(controls),
            className="shadow-sm",
            style={"position": "sticky", "top": "12px"}
        )
    else:
        return html.Div(
            controls,
            style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px", "position": "sticky", "top": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.05)"}
        )

def _main_panel():
    top_controls = [
        html.Div([
            html.Label("Stat for Graph"),
            dcc.Dropdown(
                id="pgq-stat-dropdown",
                options=[{"label": label, "value": name} for (name, label, ctype) in FILTERABLE_COLUMNS if ctype == "numeric"],
                value="assists",
                placeholder="Select a numeric statistic"
            )
        ], className="col-md-4"),
        html.Div([
            html.Label("Display Mode"),
            dcc.RadioItems(
                id="pgq-graph-mode-radio",
                options=[
                    {"label": "Average by Season", "value": "avg"},
                    {"label": "Count of Games", "value": "count"},
                ],
                value="avg",
                labelStyle={"display": "inline-block", "marginRight": "14px"}
            )
        ], className="col-md-4"),
        html.Div([
            html.Label("Grouping"),
            dcc.RadioItems(
                id="pgq-grouping-radio",
                options=[
                    {"label": "Season Only", "value": "season_only"},
                    {"label": "Season + Player", "value": "season_player"},
                ],
                value="season_only",
                labelStyle={"display": "inline-block", "marginRight": "14px"}
            )
        ], className="col-md-4")
    ]

    graph_card = (dbc.Card if HAVE_DBC else html.Div)(
        (dbc.CardBody if HAVE_DBC else html.Div)(
            dcc.Loading(html.Div(id="pgq-output-graph"))
        ),
        **({"className": "shadow-sm mt-3"} if HAVE_DBC else {"style": {"marginTop": "12px"}})
    )

    table_card = (dbc.Card if HAVE_DBC else html.Div)(
        (dbc.CardBody if HAVE_DBC else html.Div)(
            dcc.Loading(html.Div(id="pgq-output-table"))
        ),
        **({"className": "shadow-sm mt-3"} if HAVE_DBC else {"style": {"marginTop": "12px"}})
    )

    # Season averages table
    season_avg_card = (dbc.Card if HAVE_DBC else html.Div)(
        (dbc.CardBody if HAVE_DBC else html.Div)(
            dcc.Loading(html.Div(id="pgq-output-season-avg-table"))
        ),
        **({"className": "shadow-sm mt-3"} if HAVE_DBC else {"style": {"marginTop": "12px"}})
    )

    if HAVE_DBC:
        return dbc.Container(
            [
                dbc.Row(dbc.Col(dbc.Row(top_controls, className="g-3"))),
                graph_card,
                table_card,
                season_avg_card,
            ],
            fluid=True
        )
    else:
        return html.Div(
            [
                html.Div(top_controls, className="row g-3"),
                graph_card,
                table_card,
                season_avg_card,
            ],
            style={"padding": "0 6px"}
        )

# Public layout
layout_tab_player_game_query = (
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H2("Player Game-by-Game Query", className="mb-2"), md=12),
                ],
                className="align-items-center"
            ),
            dbc.Row(
                [
                    dbc.Col(_sidebar(), md=4, lg=3),
                    dbc.Col(_main_panel(), md=8, lg=9),
                ],
                className="g-3"
            ),
        ],
        fluid=True
    )
    if HAVE_DBC else
    html.Div(
        [
            html.H2("Player Game-by-Game Query", style={"marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(_sidebar(), style={"flex": "0 0 300px"}),
                    html.Div(_main_panel(), style={"flex": "1", "marginLeft": "16px"})
                ],
                style={"display": "flex", "gap": "16px"}
            )
        ],
        style={"padding": "10px"}
    )
)

#########################################################
#                CALLBACK REGISTRATION
#########################################################

def register_callbacks_tab_player_game_query(app: dash.Dash):
    """
    Register all callbacks for the 'Player Game by Game Query' tab.
    """

    def get_column_type(col_name):
        for (name, _, ctype) in FILTERABLE_COLUMNS:
            if name == col_name:
                return ctype
        return None

    @app.callback(
        [
            Output("pgq-player-dropdown", "options"),
            Output("pgq-season-dropdown", "options"),
            Output("pgq-since-season-dropdown", "options"),
            Output("pgq-status", "children")
        ],
        [Input("pgq-query-button", "n_clicks"), Input("pgq-season-range-slider", "value")],
        prevent_initial_call=False
    )
    def populate_dropdowns(_n_clicks, _slider_value):
        player_query = " UNION ".join([f"SELECT DISTINCT player_name FROM {tbl}" for tbl in season_tables])
        season_query = " UNION ".join([f"SELECT DISTINCT season FROM {tbl}" for tbl in season_tables])
        try:
            players_df = pd.read_sql(sql_text(player_query), engine)
            seasons_df = pd.read_sql(sql_text(season_query), engine)

            player_options = [{"label": n, "value": n}
                              for n in sorted(players_df["player_name"].dropna().unique())]
            season_values = sorted(seasons_df["season"].dropna().unique())
            season_options = [{"label": s, "value": s} for s in season_values]

            return player_options, season_options, season_options, ""
        except Exception as e:
            return [], [], [], f"Dropdown load error: {e}"

    @app.callback(
        Output({"type": "pgq-operator-dropdown", "index": MATCH}, "options"),
        Input({"type": "pgq-column-dropdown", "index": MATCH}, "value")
    )
    def set_operator_options(selected_column):
        if not selected_column:
            return []
        col_type = get_column_type(selected_column)
        if col_type == "numeric":
            return NUMERIC_OPERATORS
        elif col_type == "date":
            return DATE_OPERATORS
        return TEXT_OPERATORS

    @app.callback(
        Output("pgq-filter-container", "children"),
        Input("pgq-add-filter-button", "n_clicks"),
        Input("pgq-reset-filters", "n_clicks"),
        Input({"type": "pgq-remove-filter", "index": ALL}, "n_clicks"),
        State("pgq-filter-container", "children"),
        prevent_initial_call=True
    )
    def modify_filter_rows(add_clicks, reset_clicks, remove_clicks, existing_rows):
        ctx = dash.callback_context
        if not ctx.triggered:
            return existing_rows

        trig = ctx.triggered_id
        if trig == "pgq-reset-filters":
            return [_filter_row(0)]
        if trig == "pgq-add-filter-button":
            new_index = len(existing_rows)
            existing_rows.append(_filter_row(new_index))
            return existing_rows
        if isinstance(trig, dict) and trig.get("type") == "pgq-remove-filter":
            idx = trig.get("index", -1)
            kept = [row for i, row in enumerate(existing_rows) if i != idx]
            rebuilt = []
            for i, _ in enumerate(kept):
                rebuilt.append(_filter_row(i))
            return rebuilt
        return existing_rows

    @app.callback(
        Output("pgq-slider-output", "children"),
        Input("pgq-season-range-slider", "value")
    )
    def update_slider_output(slider_range):
        if not slider_range:
            return "No season range selected."
        min_year, max_year = slider_range
        return f"Selected season range: {min_year} to {max_year}"

    @app.callback(
        [
            Output("pgq-output-table", "children"),
            Output("pgq-output-season-avg-table", "children"),
            Output("pgq-output-graph", "children"),
        ],
        Input("pgq-query-button", "n_clicks"),
        State("pgq-player-dropdown", "value"),
        State("pgq-season-dropdown", "value"),
        State("pgq-since-season-dropdown", "value"),
        State({"type": "pgq-column-dropdown", "index": ALL}, "value"),
        State({"type": "pgq-operator-dropdown", "index": ALL}, "value"),
        State({"type": "pgq-value-input", "index": ALL}, "value"),
        State("pgq-unique-results-checkbox", "value"),
        State("pgq-season-range-slider", "value"),
        State("pgq-stat-dropdown", "value"),
        State("pgq-graph-mode-radio", "value"),
        State("pgq-grouping-radio", "value"),
        prevent_initial_call=True
    )
    def query_with_multiple_filters(
        _n_clicks,
        selected_players,
        selected_seasons,
        since_season,
        columns, operators, values,
        unique_results,
        slider_range,
        selected_stat,
        graph_mode,
        grouping
    ):
        # UNION all candidate tables
        union_query = " UNION ALL ".join([f"SELECT * FROM {tbl}" for tbl in season_tables])

        # WHERE building with params
        where_clauses = ["1=1"]
        params = {}
        pid = 0
        def add_param(v):
            nonlocal pid
            key = f"p{pid}"
            params[key] = v
            pid += 1
            return f":{key}"

        if selected_players:
            phs = [add_param(p) for p in selected_players]
            where_clauses.append(f"player_name IN ({', '.join(phs)})")

        if selected_seasons:
            phs = [add_param(s) for s in selected_seasons]
            where_clauses.append(f"season IN ({', '.join(phs)})")

        if since_season:
            where_clauses.append(f"season >= {add_param(since_season)}")

        if slider_range and len(slider_range) == 2:
            min_y, max_y = slider_range
            where_clauses.append(
                f"CAST(LEFT(season, 4) AS INT) BETWEEN {add_param(int(min_y))} AND {add_param(int(max_y))}"
            )

        columns = columns or []
        operators = operators or []
        values = values or []
        for col, op, val in zip(columns, operators, values):
            if not (col and op and val not in (None, "")):
                continue
            ctype = get_column_type(col)
            if ctype == "numeric":
                try:
                    nval = float(val)
                except Exception:
                    continue
                where_clauses.append(f"{col} {op} {add_param(nval)}")
            elif ctype == "text":
                if op == "LIKE":
                    where_clauses.append(f"{col} LIKE {add_param('%' + str(val) + '%')}")
                else:
                    where_clauses.append(f"{col} = {add_param(str(val))}")
            elif ctype == "date":
                where_clauses.append(f"{col} {op} {add_param(str(val))}")

        final_query = f"""
            SELECT *
            FROM (
                {union_query}
            ) AS all_seasons
            WHERE {' AND '.join(where_clauses)}
            ORDER BY game_date DESC
        """

        try:
            df = pd.read_sql(sql_text(final_query), engine, params=params)
        except Exception as e:
            err_div = html.Div(f"Error fetching data: {e}", style={"color": "#dc3545"})
            return (err_div, html.Div(), html.Div())

        if df.empty:
            empty_div = html.Div("No results found for the selected filters.")
            return (empty_div, html.Div(), html.Div())

        if "game_date" in df.columns:
            df.sort_values("game_date", ascending=False, inplace=True)

        unique_players = df["player_name"].nunique() if "player_name" in df.columns else 0
        df_gbg = df.copy()
        if "unique" in (unique_results or []):
            if "player_name" in df_gbg.columns:
                df_gbg = df_gbg.drop_duplicates(subset="player_name")
        num_results = len(df_gbg)

        # -------------------------
        # GAME-BY-GAME TABLE
        # -------------------------
        summary = html.Div(
            [
                html.Strong(f"Rows: {num_results}"),
                html.Span("  •  "),
                html.Strong(f"Unique players: {unique_players}")
            ],
            className="mb-2"
        )

        table = DataTable(
            id="pgq-result-table",
            columns=[{"name": c, "id": c} for c in df_gbg.columns],
            data=df_gbg.to_dict("records"),
            sort_action="native",
            filter_action="native",
            page_size=50,
            style_table={"overflowX": "auto", "maxHeight": "70vh", "overflowY": "auto"},
            style_header={"fontWeight": "bold", "whiteSpace": "normal", "height": "auto"},
            style_cell={
                "textAlign": "left",
                "minWidth": "90px", "width": "90px", "maxWidth": "240px",
                "whiteSpace": "normal"
            },
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "right"}
                for c, _, t in FILTERABLE_COLUMNS if t == "numeric"
            ] + [
                {"if": {"column_id": "player_name"}, "textAlign": "left", "fontWeight": "600"},
                {"if": {"column_id": "team"}, "textAlign": "left"},
                {"if": {"column_id": "season"}, "textAlign": "center"},
            ],
        )
        table_div = html.Div([summary, table])

        # -------------------------
        # SEASON AVERAGES TABLE (with %s, AST/TO, 2-dec rounding, ordering)
        # -------------------------
        def season_to_int(s):
            try:
                return int(str(s).split("_")[0])
            except Exception:
                return 0

        # If no player selected, force per-player lines (season + player).
        if selected_players:
            group_cols_sa = ["season"] if grouping == "season_only" else ["season", "player_name"]
        else:
            group_cols_sa = ["season", "player_name"]

        # Per-game means for all numeric columns (rounded to 2 decimals)
        num_cols = _present_numeric_columns(df)
        agg_means = {c: "mean" for c in num_cols}
        gavg = df.groupby(group_cols_sa, dropna=False).agg(agg_means)

        # Round all per-game numeric means to 2 decimals
        for c in num_cols:
            if c in gavg.columns:
                gavg[c] = gavg[c].round(2)

        # Games count
        gavg["Games"] = df.groupby(group_cols_sa, dropna=False).size().astype(int)

        # Season totals for derived percentages / ratios
        sum_candidates = [
            "field_goals_made", "field_goals_attempts",
            "two_points_made", "two_points_attempts",
            "three_points_made", "three_points_attempts",
            "free_throws_made", "free_throws_attempts",
            "assists", "turnovers",
        ]
        sum_cols = [c for c in sum_candidates if c in df.columns]
        gsum = pd.DataFrame(index=gavg.index)
        if sum_cols:
            gsum = df.groupby(group_cols_sa, dropna=False)[sum_cols].sum()

        def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
            num, den = num.align(den, join="left")
            return num.astype(float) / den.replace(0, np.nan).astype(float)

        derived = pd.DataFrame(index=gavg.index)

        if {"two_points_made", "two_points_attempts"}.issubset(gsum.columns):
            derived["pct_2fg"] = (_safe_div(gsum["two_points_made"], gsum["two_points_attempts"]) * 100).round(1)

        if {"three_points_made", "three_points_attempts"}.issubset(gsum.columns):
            derived["pct_3fg"] = (_safe_div(gsum["three_points_made"], gsum["three_points_attempts"]) * 100).round(1)

        if {"free_throws_made", "free_throws_attempts"}.issubset(gsum.columns):
            derived["pct_ft"] = (_safe_div(gsum["free_throws_made"], gsum["free_throws_attempts"]) * 100).round(1)

        if {"field_goals_made", "three_points_made", "field_goals_attempts"}.issubset(gsum.columns):
            num_efg = gsum["field_goals_made"] + 0.5 * gsum["three_points_made"]
            den_efg = gsum["field_goals_attempts"]
            derived["pct_efg"] = (_safe_div(num_efg, den_efg) * 100).round(1)

        if {"assists", "turnovers"}.issubset(gsum.columns):
            derived["ast_to"] = _safe_div(gsum["assists"], gsum["turnovers"]).round(2)

        gout = pd.concat([gavg, derived], axis=1).reset_index()
        gout["season_year"] = gout["season"].apply(season_to_int)

        sort_keys = ["season_year"]
        if "player_name" in gout.columns and "player_name" in group_cols_sa:
            sort_keys.append("player_name")
        gout.sort_values(sort_keys, inplace=True)

        # Column ordering: keys, Games, Points, Assists, derived %, then other stats
        lead_cols = ["season"] if group_cols_sa == ["season"] else ["player_name", "season"]
        derived_order = [c for c in ["pct_2fg", "pct_3fg", "pct_ft", "pct_efg", "ast_to"] if c in gout.columns]
        pa_cols = [c for c in ["points", "assists"] if c in gout.columns]
        num_cols_remainder = [c for c in num_cols if c not in pa_cols]

        visible_cols = lead_cols + ["Games"] + pa_cols + derived_order + num_cols_remainder
        visible_cols = [c for c in visible_cols if c in gout.columns]

        pretty_names = {
            "player_name": "Player",
            "season": "Season",
            "Games": "Games",
            "points": "Points",
            "assists": "Assists",
            "pct_2fg": "2FG%",
            "pct_3fg": "3FG%",
            "pct_ft": "FT%",
            "pct_efg": "eFG%",
            "ast_to": "AST/TO",
        }

        # Per-column formats
        int_cols = {"Games"}
        one_dec_cols = set(derived_order)    # percentage columns
        two_dec_cols = set(pa_cols + num_cols_remainder + ["ast_to"])  # per-game means + AST/TO

        def _mk_col(c):
            col = {"name": pretty_names.get(c, c.replace("_", " ").title()), "id": c}
            if c in int_cols:
                col.update({"type": "numeric", "format": Format(precision=0, scheme=Scheme.fixed)})
            elif c in one_dec_cols:
                col.update({"type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)})
            elif c in two_dec_cols:
                col.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
            return col

        columns = [_mk_col(c) for c in visible_cols]

        season_avg_table = DataTable(
            id="pgq-season-avg-table",
            columns=columns,
            data=gout[visible_cols].to_dict("records"),
            sort_action="native",
            filter_action="native",
            page_size=25,
            style_table={"overflowX": "auto", "maxHeight": "70vh", "overflowY": "auto"},
            style_header={"fontWeight": "bold", "whiteSpace": "normal", "height": "auto"},
            style_cell={
                "textAlign": "left",
                "minWidth": "90px", "width": "90px", "maxWidth": "240px",
                "whiteSpace": "normal"
            },
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "right"}
                for c in (list(one_dec_cols) + list(two_dec_cols) + ["Games"])
            ] + [
                {"if": {"column_id": "player_name"}, "textAlign": "left", "fontWeight": "600"},
                {"if": {"column_id": "season"}, "textAlign": "center"},
            ],
        )
        season_avg_div = html.Div(
            [
                html.H5("Season Averages (Totals-derived %s, 2-dec stats)", className="mb-2"),
                html.Small("If no player is selected, this table lists all players matching the filters."),
                season_avg_table
            ]
        )

        # -------------------------
        # GRAPH (unchanged; respects user 'Grouping')
        # -------------------------
        if "season" not in df.columns:
            graph_div = html.Div("No 'season' column present for grouping.")
            return (table_div, season_avg_div, graph_div)

        group_cols_graph = ["season"] if grouping == "season_only" else ["season", "player_name"]

        if graph_mode == "avg":
            if selected_stat not in df.columns:
                graph_div = html.Div(f"Selected stat '{selected_stat}' not present in the result.")
            else:
                g = df.groupby(group_cols_graph, as_index=False).agg({selected_stat: "mean"})
                g["season_year"] = g["season"].apply(lambda s: int(str(s).split("_")[0]) if isinstance(s, str) and "_" in s else 0)
                g.sort_values("season_year", inplace=True)
                if group_cols_graph == ["season"]:
                    fig = px.bar(
                        g, x="season", y=selected_stat,
                        title=f"Average {selected_stat} by Season",
                        labels={selected_stat: selected_stat.capitalize(), "season": "Season"}
                    )
                else:
                    fig = px.bar(
                        g, x="season", y=selected_stat, color="player_name", barmode="group",
                        title=f"Average {selected_stat} by Season & Player",
                        labels={selected_stat: selected_stat.capitalize(), "season": "Season", "player_name": "Player"}
                    )
                fig.update_layout(xaxis={"type": "category"}, margin=dict(l=10, r=10, t=50, b=10))
                graph_div = dcc.Graph(figure=fig, config={"displayModeBar": True})
        else:
            g = (
                df.groupby(group_cols_graph, as_index=False)
                .size()
                .rename(columns={"size": "count_of_games"})
            )
            g["season_year"] = g["season"].apply(lambda s: int(str(s).split("_")[0]) if isinstance(s, str) and "_" in s else 0)
            g.sort_values("season_year", inplace=True)
            if group_cols_graph == ["season"]:
                fig = px.bar(
                    g, x="season", y="count_of_games",
                    title="Count of Games per Season",
                    labels={"count_of_games": "Count of Games", "season": "Season"}
                )
            else:
                fig = px.bar(
                    g, x="season", y="count_of_games", color="player_name", barmode="group",
                    title="Count of Games per Season & Player",
                    labels={"count_of_games": "Count of Games", "season": "Season", "player_name": "Player"}
                )
            fig.update_layout(xaxis={"type": "category"}, margin=dict(l=10, r=10, t=50, b=10))
            graph_div = dcc.Graph(figure=fig, config={"displayModeBar": True})

        return (table_div, season_avg_div, graph_div)

#########################################################
#               FACTORY (for Flask mount)
#########################################################
from dash import Dash  # keep after potential dbc import

def create_dash_player_game_query(server, base_pathname="/player_game_query/", **kwargs):
    """
    Create and mount the Player Game Query Dash app inside Flask.
    """
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        external_stylesheets=BOOTSTRAP,
        suppress_callback_exceptions=True,
        title="Player Game Query",
    )

    dash_app.layout = layout_tab_player_game_query
    register_callbacks_tab_player_game_query(dash_app)
    return dash_app
