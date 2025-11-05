# dashapps/all_time.py
# (UNIFIED, with Min Attempts Filters + High-Contrast Dropdowns + Row "Details" -> Season Breakdown)
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

# Optional DB
from sqlalchemy import create_engine

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
# CSV option (if files exist, they’ll be used)
DATA_PATHS = {
    "BSL": "data/all_time_bsl.csv",
    "EL":  "data/all_time_el.csv",
    "EC":  "data/all_time_ec.csv",
}
LEAGUE_OPTIONS = [{"label": k, "value": k} for k in DATA_PATHS.keys()]

# Postgres option (fallback if CSV not present)
DATABASE_URI = os.getenv("BASKETBALL_DB_URI", "postgresql://philipmulryne@localhost:5432/basketball_data")
DB_ENGINE = None
try:
    if DATABASE_URI and "://" in DATABASE_URI:
        DB_ENGINE = create_engine(DATABASE_URI)
except Exception:
    DB_ENGINE = None

# The DB loader expects seasonal tables like processed_games_1985_86 … processed_games_2024_25
DB_SEASON_START = int(os.getenv("ALLTIME_DB_START_YEAR", "1985"))
DB_SEASON_END   = int(os.getenv("ALLTIME_DB_END_YEAR",   "2025"))
DB_EXCEPTIONS   = set(os.getenv("ALLTIME_DB_EXCEPTIONS", "2019_20").split(",")) if os.getenv("ALLTIME_DB_EXCEPTIONS") else {"2019_20"}

# Metrics exposed to the user
METRIC_OPTIONS = [
    ("Points", "PTS"),
    ("Assists", "AST"),
    ("Rebounds", "REB"),
    ("Steals", "STL"),
    ("Blocks", "BLK"),
    ("3P Made", "3PM"),
    ("3P Attempted", "3PA"),
    ("FG%", "FG_PCT"),
    ("3P%", "TP_PCT"),
    ("FT%", "FT_PCT"),
    ("Turnovers", "TOV"),
]
NORMALIZATION_OPTIONS = [
    ("Per Game", "per_game"),
    ("Per 36 Min", "per_36"),
    ("Totals (career)", "totals"),
]
DEFAULT_TOP_N = 15


# --------------------------------------------------------------------------------------
# DATA LOADING / PREP
# --------------------------------------------------------------------------------------
def _safe_int_season(season_val) -> int:
    """Convert '2024-25' or '2017' to 2024 or 2017."""
    if pd.isna(season_val):
        return 0
    s = str(season_val)
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else 0

def _pct(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce").replace({0: np.nan})
    return (numer / denom) * 100.0

def _season_label(y: int) -> str:
    return f"{y}-{str(y+1)[-2:]}"

def _season_table_name(y: int) -> str:
    return f"{y}_{str(y+1)[-2:]}"

def _season_tables_in_range(start_year: int, end_year: int, exceptions: set[str]) -> List[str]:
    out = []
    for y in range(start_year, end_year + 1):
        ss = _season_table_name(y)
        if ss not in exceptions:
            out.append(ss)
    return out

def _load_from_csv(league: str) -> Optional[pd.DataFrame]:
    path = Path(DATA_PATHS.get(league, ""))
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Coerce numeric
    for c in ["Games","Minutes","PTS","AST","REB","STL","BLK","FGM","FGA","3PM","3PA","FTM","FTA","TOV"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Percentages
    df["FG_PCT"] = _pct(df.get("FGM", 0), df.get("FGA", 0)).round(2)
    df["TP_PCT"] = _pct(df.get("3PM", 0), df.get("3PA", 0)).round(2)
    df["FT_PCT"] = _pct(df.get("FTM", 0), df.get("FTA", 0)).round(2)
    # Season numeric
    df["SeasonStart"] = df["Season"].apply(_safe_int_season)
    if "POS" not in df.columns:
        df["POS"] = ""
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).fillna("—")
    return df

def _try_query(engine, query: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_sql(query, engine)
    except Exception:
        return None

def _load_from_db(start_year: int, end_year: int, exceptions: set[str]) -> Optional[pd.DataFrame]:
    """Build a long 'season rows' dataframe from per-season tables, then we aggregate to careers later."""
    if DB_ENGINE is None:
        return None

    dfs = []
    for y in range(start_year, end_year + 1):
        season_token = _season_table_name(y)   # e.g., 1985_86
        if season_token in exceptions:
            continue
        table_name = f"processed_games_{season_token}"

        # Try with Minutes if present (column name varies across repos — attempt both)
        q_with_minutes = f"""
            SELECT 
                player_name AS "Player",
                team        AS "Team",
                game_type   AS game_type, 
                COUNT(DISTINCT game_code) AS "Games",
                COALESCE(SUM(minutes), SUM(minutes_played)) AS "Minutes",
                SUM(points)              AS "PTS",
                SUM(assists)             AS "AST",
                SUM(total_rebounds)      AS "REB",
                SUM(steals)              AS "STL",
                SUM(blocks)              AS "BLK",
                SUM(field_goals_made)    AS "FGM",
                SUM(field_goals_attempts)AS "FGA",
                SUM(three_points_made)   AS "3PM",
                SUM(three_points_attempts) AS "3PA",
                SUM(free_throws_made)    AS "FTM",
                SUM(free_throws_attempts)AS "FTA",
                SUM(turnovers)           AS "TOV"
            FROM {table_name}
            WHERE player_name != 'TAKIM'
            GROUP BY player_name, team, game_type
        """
        df = _try_query(DB_ENGINE, q_with_minutes)

        if df is None:
            # Retry without Minutes column
            q_no_minutes = f"""
                SELECT 
                    player_name AS "Player",
                    team        AS "Team",
                    game_type   AS game_type,   
                    COUNT(DISTINCT game_code) AS "Games",
                    SUM(points)              AS "PTS",
                    SUM(assists)             AS "AST",
                    SUM(total_rebounds)      AS "REB",
                    SUM(steals)              AS "STL",
                    SUM(blocks)              AS "BLK",
                    SUM(field_goals_made)    AS "FGM",
                    SUM(field_goals_attempts)AS "FGA",
                    SUM(three_points_made)   AS "3PM",
                    SUM(three_points_attempts) AS "3PA",
                    SUM(free_throws_made)    AS "FTM",
                    SUM(free_throws_attempts)AS "FTA",
                    SUM(turnovers)           AS "TOV"
                FROM {table_name}
                WHERE player_name != 'TAKIM'
                GROUP BY player_name, team, game_type
            """
            df = _try_query(DB_ENGINE, q_no_minutes)
            if df is not None:
                df["Minutes"] = 0  # if backend lacks minutes, keep zeros so per-36 just shows 0

        if df is None or df.empty:
            # silent skip of missing/old seasons
            continue

        df["Season"] = _season_label(y)
        dfs.append(df)

    if not dfs:
        return None

    df_long = pd.concat(dfs, ignore_index=True)

    # --- Normalize GameType to canonical lower-case tokens -----------------------
    if "game_type" in df_long.columns:
        df_long["game_type"] = (
            df_long["game_type"].astype(str).str.strip().str.lower().replace({
                "regular season": "regular",
                "rs": "regular",
                "playoffs": "playoff",
                "postseason": "playoff",
                "po": "playoff",
            })
        )
    else:
        df_long["game_type"] = "regular"

    # Coerce types (safety)
    for c in ["Games","Minutes","PTS","AST","REB","STL","BLK","FGM","FGA","3PM","3PA","FTM","FTA","TOV"]:
        if c in df_long.columns:
            df_long[c] = pd.to_numeric(df_long[c], errors="coerce").fillna(0)

    # Shooting %
    df_long["FG_PCT"] = _pct(df_long["FGM"], df_long["FGA"]).round(2)
    df_long["TP_PCT"] = _pct(df_long["3PM"], df_long["3PA"]).round(2)
    df_long["FT_PCT"] = _pct(df_long["FTM"], df_long["FTA"]).round(2)

    # Season numeric
    df_long["SeasonStart"] = df_long["Season"].apply(_safe_int_season)

    # POS may be absent at source; keep missing-friendly
    if "POS" not in df_long.columns:
        df_long["POS"] = ""

    # Normalize Team to string
    df_long["Team"] = df_long["Team"].astype(str).fillna("—")
    return df_long

def _load_dataframe(league: str) -> pd.DataFrame:
    """
    Load the all-time dataset for the selected league from CSV if present.
    If no CSV, try DB. If neither available, return a small demo dataset.
    """
    # Try CSV
    df = _load_from_csv(league)
    if df is not None:
        return df

    # Try DB (season-long build)
    df = _load_from_db(DB_SEASON_START, DB_SEASON_END, DB_EXCEPTIONS)
    if df is not None:
        return df

    # --- Demo fallback (keeps the app functional) ---
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Season": ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"] * 6,
        "Player": (["Player A"]*5 + ["Player B"]*5 + ["Player C"]*5 +
                   ["Player D"]*5 + ["Player E"]*5 + ["Player F"]*5),
        "Team":   (["Team X"]*10 + ["Team Y"]*10 + ["Team Z"]*10),
        "POS":    (["G","G","G","G","G"] + ["F","F","F","F","F"] + ["C"]*5 +
                   ["G"]*5 + ["F"]*5 + ["C"]*5),
        "Games":  rng.integers(10, 34, size=30),
        "Minutes": rng.integers(300, 950, size=30),
        "PTS":    rng.integers(80, 700, size=30),
        "AST":    rng.integers(20, 180, size=30),
        "REB":    rng.integers(30, 220, size=30),
        "STL":    rng.integers(5, 60, size=30),
        "BLK":    rng.integers(3, 40, size=30),
        "FGM":    rng.integers(30, 250, size=30),
        "FGA":    rng.integers(80, 550, size=30),
        "3PM":    rng.integers(5, 120, size=30),
        "3PA":    rng.integers(20, 300, size=30),
        "FTM":    rng.integers(10, 150, size=30),
        "FTA":    rng.integers(15, 200, size=30),
        "TOV":    rng.integers(10, 120, size=30),
    })
    for c in ["Games","Minutes","PTS","AST","REB","STL","BLK","FGM","FGA","3PM","3PA","FTM","FTA","TOV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["FG_PCT"] = _pct(df["FGM"], df["FGA"]).round(2)
    df["TP_PCT"] = _pct(df["3PM"], df["3PA"]).round(2)
    df["FT_PCT"] = _pct(df["FTM"], df["FTA"]).round(2)
    df["SeasonStart"] = df["Season"].apply(_safe_int_season)
    return df

def _aggregate_career(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per player across selected seasons: compute career totals & per-game/per-36."""
    last_team = df.sort_values(["Player", "SeasonStart"]).groupby("Player")["Team"].last()

    grp = df.groupby("Player", as_index=False).agg({
        "Games": "sum",
        "Minutes": "sum",
        "PTS": "sum",
        "AST": "sum",
        "REB": "sum",
        "STL": "sum",
        "BLK": "sum",
        "FGM": "sum",
        "FGA": "sum",
        "3PM": "sum",
        "3PA": "sum",
        "FTM": "sum",
        "FTA": "sum",
        "TOV": "sum",
    })
    grp["Team"] = grp["Player"].map(last_team).fillna("—")

    # Derivable split: 2PT made/attempts
    grp["2PM"] = (grp["FGM"] - grp["3PM"]).clip(lower=0)
    grp["2PA"] = (grp["FGA"] - grp["3PA"]).clip(lower=0)

    # Percentages (career)
    grp["FG_PCT"] = _pct(grp["FGM"], grp["FGA"]).round(2)
    grp["TP_PCT"] = _pct(grp["3PM"], grp["3PA"]).round(2)
    grp["FT_PCT"] = _pct(grp["FTM"], grp["FTA"]).round(2)

    # Per game & per 36
    g = grp["Games"].replace({0: np.nan})
    m = grp["Minutes"].replace({0: np.nan})

    for raw, per, per36 in [
        ("PTS","PTS_PG","PTS_36"),
        ("AST","AST_PG","AST_36"),
        ("REB","REB_PG","REB_36"),
        ("STL","STL_PG","STL_36"),
        ("BLK","BLK_PG","BLK_36"),
        ("TOV","TOV_PG","TOV_36"),
        ("3PM","3PM_PG","3PM_36"),
        ("3PA","3PA_PG","3PA_36"),
    ]:
        grp[per] = (grp[raw] / g).round(2)
        grp[per36] = ((grp[raw] / m) * 36).round(2)

    # Advanced
    grp["eFG%"] = ((grp["FGM"] + 0.5 * grp["3PM"]) / grp["FGA"].replace({0: np.nan}) * 100).round(2)
    denom = (2 * (grp["FGA"] + 0.44 * grp["FTA"])).replace({0: np.nan})
    grp["TS%"] = (grp["PTS"] / denom * 100).round(2)

    return grp.fillna(0)

def _apply_filters(
    df_all: pd.DataFrame,
    season_range: Tuple[int, int],
    teams: List[str],
    positions: List[str],
    min_games: int,
    player_substr: str,
    min_fga: int,
    min_3pa: int,
    min_2pa: int,
    min_fta: int,
    game_type: str = "all",
) -> pd.DataFrame:
    df = df_all.copy()

    # Season range filter (pre-aggregation)
    s0, s1 = season_range
    df = df[(df["SeasonStart"] >= s0) & (df["SeasonStart"] <= s1)]

    if teams:
        df = df[df["Team"].isin(teams)]

    if "POS" in df.columns and positions:
        df = df[df["POS"].isin(positions)]
    
    # Game type filter BEFORE aggregation
    if game_type and game_type != "all" and "game_type" in df.columns:
        df = df[df["game_type"] == str(game_type).lower()]

    career = _aggregate_career(df)

    # Threshold filters (attempts + games)
    career = career[
        (career["Games"] >= int(min_games)) &
        (career["FGA"]   >= int(min_fga))   &
        (career["3PA"]   >= int(min_3pa))   &
        (career["2PA"]   >= int(min_2pa))   &
        (career["FTA"]   >= int(min_fta))
    ]

    # Player name filter
    if player_substr:
        ss = player_substr.lower()
        career = career[career["Player"].str.lower().str.contains(ss)]

    # Stable default ordering
    return career.sort_values(["PTS", "AST", "REB"], ascending=False)

def _compute_display_metric(df_career: pd.DataFrame, metric_key: str, norm: str) -> pd.Series:
    pct_like = {"FG_PCT", "TP_PCT", "FT_PCT", "eFG%", "TS%"}
    key = metric_key
    if key in pct_like:
        return df_career[key].fillna(0)
    if norm == "totals":
        return df_career[key].fillna(0)
    if norm == "per_game":
        table = {"PTS":"PTS_PG","AST":"AST_PG","REB":"REB_PG","STL":"STL_PG","BLK":"BLK_PG","TOV":"TOV_PG","3PM":"3PM_PG","3PA":"3PA_PG"}
        return df_career[table.get(key, key)].fillna(0)
    if norm == "per_36":
        table = {"PTS":"PTS_36","AST":"AST_36","REB":"REB_36","STL":"STL_36","BLK":"BLK_36","TOV":"TOV_36","3PM":"3PM_36","3PA":"3PA_36"}
        return df_career[table.get(key, key)].fillna(0)
    return df_career[key].fillna(0)

def _season_breakdown_for_player(
    df_all: pd.DataFrame,
    player: str,
    season_range: Tuple[int, int],
    teams_filter: List[str],
    positions_filter: List[str],
    game_type: str,
) -> pd.DataFrame:
    """
    Build a Season x Team breakdown for a selected player, respecting season range,
    game_type, and optional team/position filters.
    """
    if df_all.empty or not player:
        return pd.DataFrame()

    df = df_all.copy()

    # Pre-aggregation filters consistent with top table
    s0, s1 = season_range
    df = df[(df["SeasonStart"] >= s0) & (df["SeasonStart"] <= s1)]

    if teams_filter:
        df = df[df["Team"].isin(teams_filter)]

    if "POS" in df.columns and positions_filter:
        df = df[df["POS"].isin(positions_filter)]

    if game_type and game_type != "all" and "game_type" in df.columns:
        df = df[df["game_type"] == str(game_type).lower()]

    # Keep only selected player
    df = df[df["Player"] == player].copy()
    if df.empty:
        return df

    # Aggregate per Season x Team (sums)
    cols_sum = ["Games","Minutes","PTS","AST","REB","STL","BLK","FGM","FGA","3PM","3PA","FTM","FTA","TOV"]
    agg_map = {c: "sum" for c in cols_sum if c in df.columns}
    gb = df.groupby(["Season","Team","SeasonStart"], as_index=False).agg(agg_map)

    # Derivations
    if "FGM" in gb and "FGA" in gb:
        gb["FG_PCT"] = _pct(gb["FGM"], gb["FGA"]).round(2)
    else:
        gb["FG_PCT"] = 0.0
    if "3PM" in gb and "3PA" in gb:
        gb["TP_PCT"] = _pct(gb["3PM"], gb["3PA"]).round(2)
    else:
        gb["TP_PCT"] = 0.0
    if "FTM" in gb and "FTA" in gb:
        gb["FT_PCT"] = _pct(gb["FTM"], gb["FTA"]).round(2)
    else:
        gb["FT_PCT"] = 0.0

    # Per-game & per-36
    g = gb["Games"].replace({0: np.nan})
    m = gb.get("Minutes", pd.Series([np.nan] * len(gb))).replace({0: np.nan})

    for raw, per, per36 in [
        ("PTS","PTS_PG","PTS_36"),
        ("AST","AST_PG","AST_36"),
        ("REB","REB_PG","REB_36"),
        ("STL","STL_PG","STL_36"),
        ("BLK","BLK_PG","BLK_36"),
        ("3PM","3PM_PG","3PM_36"),
        ("3PA","3PA_PG","3PA_36"),
        ("TOV","TOV_PG","TOV_36"),
    ]:
        if raw in gb.columns:
            gb[per]   = (gb[raw] / g).round(2)
            gb[per36] = ((gb[raw] / m) * 36).round(2)

    # Advanced
    if "FGA" in gb.columns and "3PM" in gb.columns:
        gb["eFG%"] = ((gb.get("FGM", 0) + 0.5 * gb["3PM"]) / gb["FGA"].replace({0: np.nan}) * 100).round(2)
    else:
        gb["eFG%"] = 0.0
    denom = (2 * (gb.get("FGA", 0) + 0.44 * gb.get("FTA", 0))).replace({0: np.nan})
    if "PTS" in gb.columns:
        gb["TS%"] = (gb["PTS"] / denom * 100).round(2)
    else:
        gb["TS%"] = 0.0

    # Order by season ascending
    gb = gb.sort_values(["SeasonStart","Team"]).reset_index(drop=True)
    return gb.fillna(0)


# --------------------------------------------------------------------------------------
# APP FACTORY (assets folder; safe across Dash versions)
# --------------------------------------------------------------------------------------
def create_dash_all_time(server, base_pathname="/all_time/", **kwargs):
    """
    Uses a per-app assets folder:
      dashapps/all_time/assets/all_time/overrides.css
    This avoids html.Style and prevents CSS leakage to other dashboards.
    """
    # Per-app assets directory (namespaced)
    assets_dir = Path(__file__).parent / "assets" / "all_time"

    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,     # mounted by the Flask host
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder=str(assets_dir),       # <— point to our namespaced assets
        assets_url_path="all_time_assets",   # <— avoids collisions with other apps
        title="All-Time Leaderboards",
    )

    # Stores
    store_raw = dcc.Store(id="alltime-store-raw")
    store_filtered = dcc.Store(id="alltime-store-filtered")
    store_teardown = dcc.Store(id="alltime-store-teardown")
    store_selected_player = dcc.Store(id="alltime-selected-player")  # ← NEW

    # Controls
    controls = dbc.Card(
        [
            html.Div([
                html.Div([
                    html.Label("League", className="form-label"),
                    dcc.Dropdown(
                        id="alltime-league",
                        options=LEAGUE_OPTIONS,
                        value=(LEAGUE_OPTIONS[0]["value"] if LEAGUE_OPTIONS else None),
                        clearable=False,
                        className="dd-dark",
                    ),
                ], className="col-12 col-md-3 mb-2"),

                html.Div([
                    html.Div([
                        html.Label("Game Type", className="form-label"),
                        dcc.Dropdown(
                            id="alltime-game-type",
                            options=[
                                {"label": "All Games", "value": "all"},
                                {"label": "Regular Season", "value": "regular"},
                                {"label": "Playoffs", "value": "playoff"},
                            ],
                            value="all",
                            clearable=False,
                            className="dd-dark",
                        ),
                    ], className="col-12 col-md-3 mb-2"),
                ], className="row"),

                html.Div([
                    html.Label("Metric", className="form-label"),
                    dcc.Dropdown(
                        id="alltime-metric",
                        options=[{"label": a, "value": b} for a,b in METRIC_OPTIONS] +
                                [{"label":"eFG%", "value":"eFG%"}, {"label":"TS%", "value":"TS%"}],
                        value="PTS",
                        clearable=False,
                        className="dd-dark",
                    ),
                ], className="col-12 col-md-3 mb-2"),

                html.Div([
                    html.Label("Normalization", className="form-label"),
                    dcc.Dropdown(
                        id="alltime-norm",
                        options=[{"label": a, "value": b} for a,b in NORMALIZATION_OPTIONS],
                        value="per_game",
                        clearable=False,
                        className="dd-dark",
                    ),
                ], className="col-12 col-md-3 mb-2"),

                html.Div([
                    html.Label("Min Games", className="form-label"),
                    dcc.Slider(id="alltime-min-games", min=0, max=200, step=1, value=30,
                               marks={0:"0", 50:"50", 100:"100", 150:"150", 200:"200"}),
                ], className="col-12 col-md-3 mb-2"),
            ], className="row"),

            html.Div([
                html.Div([
                    html.Label("Season Range", className="form-label"),
                    dcc.RangeSlider(
                        id="alltime-season-range",
                        min=2000,
                        max=2026,  # includes 2025–26 season
                        step=1,
                        value=[2015, 2026],
                        marks={y: f"{y}-{str(y+1)[-2:]}" for y in range(2000, 2027)},
                        tooltip={"always_visible": True, "placement": "bottom"},
                        updatemode="mouseup",
                        allowCross=False,
                    ),
                ], className="col-12 col-lg-5 mb-2"),

                html.Div([
                    html.Label("Teams", className="form-label"),
                    dcc.Dropdown(id="alltime-teams", options=[], value=[], multi=True, placeholder="All teams",
                                 className="dd-dark"),
                ], className="col-12 col-lg-3 mb-2"),

                html.Div([
                    html.Label("Positions", className="form-label"),
                    dcc.Dropdown(
                        id="alltime-positions",
                        options=[{"label": p, "value": p} for p in ["G","F","C","G/F","F/C"]],
                        value=[], multi=True, placeholder="All positions",
                        className="dd-dark",
                    ),
                ], className="col-12 col-lg-2 mb-2"),

                html.Div([
                    html.Label("Player search", className="form-label"),
                    dcc.Input(
                        id="alltime-player-substr", type="text", placeholder="Type a name...",
                        debounce=True, className="form-control"
                    ),
                ], className="col-12 col-lg-2 mb-2"),
            ], className="row"),

            # ---------- Attempts thresholds ----------
            html.Div([
                html.Div([
                    html.Label("Min FGA (career)", className="form-label"),
                    dcc.Input(id="alltime-min-fga", type="number", min=0, step=1, value=0,
                              className="form-control", placeholder="0"),
                ], className="col-6 col-md-3 mb-2"),

                html.Div([
                    html.Label("Min 3PA (career)", className="form-label"),
                    dcc.Input(id="alltime-min-3pa", type="number", min=0, step=1, value=10,
                              className="form-control", placeholder="10"),
                ], className="col-6 col-md-3 mb-2"),

                html.Div([
                    html.Label("Min 2PA (career)", className="form-label"),
                    dcc.Input(id="alltime-min-2pa", type="number", min=0, step=1, value=10,
                              className="form-control", placeholder="10"),
                ], className="col-6 col-md-3 mb-2"),

                html.Div([
                    html.Label("Min FTA (career)", className="form-label"),
                    dcc.Input(id="alltime-min-fta", type="number", min=0, step=1, value=10,
                              className="form-control", placeholder="10"),
                ], className="col-6 col-md-3 mb-2"),
            ], className="row"),

            html.Div([
                dbc.Button("Reset Filters", id="alltime-btn-reset", color="secondary", className="me-2"),
                dbc.Button("Download CSV", id="alltime-btn-download", color="primary"),
                dcc.Download(id="alltime-download"),
            ], className="mt-2"),
        ],
        className="p-3 mb-3",
        style={"background": "#171b27", "border":"1px solid #243047", "borderRadius":"14px"}
    )

    # KPI strip
    kpi_row = html.Div(id="alltime-kpis", className="row g-3 mb-3")

    # Leader chart
    chart = dbc.Card(
        dcc.Graph(id="alltime-chart", config={"displayModeBar": False}),
        className="p-3 mb-3",
        style={"background": "#171b27", "border":"1px solid #243047", "borderRadius":"14px"}
    )

    # Table (add a clickable Details column)
    table = dash_table.DataTable(
        id="alltime-table",
        columns=[
            {"name":"", "id":"__DETAILS__", "presentation":"markdown"},
            {"name":"Player", "id":"Player"},
            {"name":"Team", "id":"Team"},
            {"name":"Games", "id":"Games", "type":"numeric"},
            {"name":"Minutes", "id":"Minutes", "type":"numeric"},
            {"name":"PTS", "id":"PTS", "type":"numeric"},
            {"name":"AST", "id":"AST", "type":"numeric"},
            {"name":"REB", "id":"REB", "type":"numeric"},
            {"name":"STL", "id":"STL", "type":"numeric"},
            {"name":"BLK", "id":"BLK", "type":"numeric"},
            {"name":"FGA", "id":"FGA", "type":"numeric"},
            {"name":"3PA", "id":"3PA", "type":"numeric"},
            {"name":"2PA", "id":"2PA", "type":"numeric"},
            {"name":"FTA", "id":"FTA", "type":"numeric"},
            {"name":"FG%", "id":"FG_PCT", "type":"numeric"},
            {"name":"3P%", "id":"TP_PCT", "type":"numeric"},
            {"name":"FT%", "id":"FT_PCT", "type":"numeric"},
            {"name":"eFG%", "id":"eFG%", "type":"numeric"},
            {"name":"TS%", "id":"TS%"},
            {"name":"3PM", "id":"3PM", "type":"numeric"},
        ],
        data=[],
        page_size=25,
        sort_action="native",
        filter_action="native",
        style_as_list_view=True,
        style_table={"overflowX":"auto"},
        style_header={
            "backgroundColor":"#0f1530","color":"#e2e8f0","border":"1px solid #243047","fontWeight":"700"
        },
        style_cell={
            "backgroundColor":"#0f1424","color":"#f1f5f9","border":"1px solid #243047",
            "padding":"8px","fontFamily":"system-ui, -apple-system, Segoe UI, Roboto, Arial","fontSize":"14px"
        },
        style_data_conditional=[
            {"if":{"row_index":"odd"}, "backgroundColor":"#111831"},
            {"if":{"column_id":"__DETAILS__"}, "textAlign":"center", "width":"36px", "minWidth":"36px", "maxWidth":"36px"},
        ],
    )
    table_card = dbc.Card(table, className="p-3", style={"background": "#171b27", "border":"1px solid #243047", "borderRadius":"14px"})

    # Player breakdown area
    details_card = dbc.Card(
        [
            html.Div(id="alltime-player-header", className="mb-2"),
            dash_table.DataTable(
                id="alltime-player-season-table",
                columns=[
                    {"name":"Season", "id":"Season"},
                    {"name":"Team", "id":"Team"},
                    {"name":"Games", "id":"Games", "type":"numeric"},
                    {"name":"Minutes", "id":"Minutes", "type":"numeric"},
                    {"name":"PTS", "id":"PTS", "type":"numeric"},
                    {"name":"PTS/G", "id":"PTS_PG", "type":"numeric"},
                    {"name":"PTS/36", "id":"PTS_36", "type":"numeric"},
                    {"name":"AST", "id":"AST", "type":"numeric"},
                    {"name":"AST/G", "id":"AST_PG", "type":"numeric"},
                    {"name":"AST/36", "id":"AST_36", "type":"numeric"},
                    {"name":"REB", "id":"REB", "type":"numeric"},
                    {"name":"REB/G", "id":"REB_PG", "type":"numeric"},
                    {"name":"REB/36", "id":"REB_36", "type":"numeric"},
                    {"name":"STL", "id":"STL", "type":"numeric"},
                    {"name":"BLK", "id":"BLK", "type":"numeric"},
                    {"name":"FGA", "id":"FGA", "type":"numeric"},
                    {"name":"3PA", "id":"3PA", "type":"numeric"},
                    {"name":"FTA", "id":"FTA", "type":"numeric"},
                    {"name":"FG%", "id":"FG_PCT", "type":"numeric"},
                    {"name":"3P%", "id":"TP_PCT", "type":"numeric"},
                    {"name":"FT%", "id":"FT_PCT", "type":"numeric"},
                    {"name":"eFG%", "id":"eFG%", "type":"numeric"},
                    {"name":"TS%", "id":"TS%", "type":"numeric"},
                ],
                data=[],
                page_size=50,
                sort_action="native",
                style_as_list_view=True,
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":"#0f1530","color":"#e2e8f0","border":"1px solid #243047","fontWeight":"700"},
                style_cell={"backgroundColor":"#0f1424","color":"#f1f5f9","border":"1px solid #243047","padding":"6px","fontSize":"13px"},
            ),
        ],
        className="p-3",
        style={"background": "#171b27", "border":"1px solid #243047", "borderRadius":"14px", "display":"none"},
        id="alltime-player-details-card",
    )

    app.layout = dbc.Container(
        [
            store_raw, store_filtered, store_teardown, store_selected_player,
            html.H2("All-Time Leaderboards", className="mb-3"),
            controls,
            kpi_row,
            chart,
            table_card,
            details_card,
        ],
        fluid=True,
        className="pt-2 pb-4 alltime-scope",  # scope for CSS overrides
    )

    # ----------------------------------------------------------------------------------
    # CALLBACKS
    # ----------------------------------------------------------------------------------
    @app.callback(
        Output("alltime-store-raw", "data"),
        Output("alltime-season-range", "min"),
        Output("alltime-season-range", "max"),
        Output("alltime-season-range", "value"),
        Output("alltime-teams", "options"),
        Input("alltime-league", "value"),
    )
    def load_league(league):
        df = _load_dataframe(league or "BSL")

        # Season slider bounds
        if df["SeasonStart"].notna().any():
            mn = int(df["SeasonStart"].min())
            mx = int(df["SeasonStart"].max())
        else:
            mn, mx = 2000, 2025
        val = [max(mn, mx - 10), mx]

        # Team options from dataset
        teams = sorted([str(t) for t in df["Team"].dropna().unique().tolist()])
        team_opts = [{"label": t, "value": t} for t in teams]

        return df.to_dict("records"), mn, mx, val, team_opts

    @app.callback(
        Output("alltime-store-filtered", "data"),
        Input("alltime-store-raw", "data"),
        Input("alltime-season-range", "value"),
        Input("alltime-teams", "value"),
        Input("alltime-positions", "value"),
        Input("alltime-min-games", "value"),
        Input("alltime-player-substr", "value"),
        Input("alltime-min-fga", "value"),
        Input("alltime-min-3pa", "value"),
        Input("alltime-min-2pa", "value"),
        Input("alltime-min-fta", "value"),
        Input("alltime-game-type", "value"),
    )
    def filter_data(raw, season_range, teams, positions, min_games, q, min_fga, min_3pa, min_2pa, min_fta, game_type):
        if not raw:
            return []
        df_all = pd.DataFrame(raw)
        teams = teams or []
        positions = positions or []
        q = (q or "").strip()
        season_range = season_range or [2000, 2025]

        # Normalize threshold inputs
        min_fga = int(min_fga or 0)
        min_3pa = int(min_3pa or 0)
        min_2pa = int(min_2pa or 0)
        min_fta = int(min_fta or 0)

        out = _apply_filters(
            df_all,
            season_range=(int(season_range[0]), int(season_range[1])),
            teams=teams,
            positions=positions,
            min_games=int(min_games or 0),
            player_substr=q,
            min_fga=min_fga,
            min_3pa=min_3pa,
            min_2pa=min_2pa,
            min_fta=min_fta,
            game_type=game_type or "all",
        )
        return out.to_dict("records")

    @app.callback(
        Output("alltime-kpis", "children"),
        Output("alltime-chart", "figure"),
        Output("alltime-table", "data"),
        Input("alltime-store-filtered", "data"),
        Input("alltime-metric", "value"),
        Input("alltime-norm", "value"),
    )
    def update_outputs(filtered, metric_key, norm):
        dfc = pd.DataFrame(filtered or [])
        if dfc.empty:
            empty_fig = px.bar(title="No data for current filters")
            empty_fig.update_layout(template="plotly_dark", paper_bgcolor="#171b27", plot_bgcolor="#171b27")
            return _kpi_cards(0, 0, 0, 0), empty_fig, []

        # Rank series
        rank_series = _compute_display_metric(dfc, metric_key, norm)
        dfc = dfc.assign(__METRIC__=rank_series)

        # KPIs
        top_val = float(dfc["__METRIC__"].max()) if not dfc["__METRIC__"].empty else 0
        n_players = int(dfc.shape[0])
        avg_games = float(dfc["Games"].mean()) if "Games" in dfc else 0
        avg_minutes = float(dfc["Minutes"].mean()) if "Minutes" in dfc else 0
        kpis = _kpi_cards(n_players, avg_games, avg_minutes, top_val, metric_key, norm)

        # Chart (Top N)
        topn = dfc.sort_values("__METRIC__", ascending=False).head(DEFAULT_TOP_N)
        fig = px.bar(
            topn,
            x="__METRIC__", y="Player",
            orientation="h",
            hover_data=["Team", "Games", "Minutes"],
            labels={"__METRIC__": _metric_label(metric_key, norm)},
            title=f"Top {DEFAULT_TOP_N} — {_metric_label(metric_key, norm)}",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171b27",
            plot_bgcolor="#171b27",
            xaxis_title=_metric_label(metric_key, norm),
            yaxis_title="",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_traces(marker_line_color="#243047", marker_line_width=1.2)

        # Table data (with normalized cols)
        df_display = dfc.copy()
        for src, per, per36 in [
            ("PTS","PTS_PG","PTS_36"),
            ("AST","AST_PG","AST_36"),
            ("REB","REB_PG","REB_36"),
            ("STL","STL_PG","STL_36"),
            ("BLK","BLK_PG","BLK_36"),
            ("3PM","3PM_PG","3PM_36"),
            ("3PA","3PA_PG","3PA_36"),
            ("TOV","TOV_PG","TOV_36"),
        ]:
            for col in (per, per36):
                if col not in df_display:
                    if col.endswith("_PG") and "Games" in df_display and src in df_display:
                        df_display[col] = (df_display[src] / df_display["Games"].replace({0: np.nan})).round(2).fillna(0)
                    elif col.endswith("_36") and "Minutes" in df_display and src in df_display:
                        df_display[col] = ((df_display[src] / df_display["Minutes"].replace({0: np.nan})) * 36).round(2).fillna(0)

        # Ensure attempts columns visible (helpful since we filter on them)
        if "2PA" not in df_display.columns and "FGA" in df_display and "3PA" in df_display:
            df_display["2PA"] = (df_display["FGA"] - df_display["3PA"]).clip(lower=0)

        # Add clickable "Details" glyph per row
        df_display["__DETAILS__"] = "➕"

        keep_cols = ["__DETAILS__","Player","Team","Games","Minutes",
                     "PTS","PTS_PG","PTS_36",
                     "AST","AST_PG","AST_36",
                     "REB","REB_PG","REB_36",
                     "STL","STL_PG","STL_36",
                     "BLK","BLK_PG","BLK_36",
                     "FGA","3PA","2PA","FTA","3PM",
                     "FG_PCT","TP_PCT","FT_PCT","eFG%","TS%"]
        cols = [c for c in keep_cols if c in df_display.columns]
        df_display = df_display[cols]

        # Render ➕ as markdown so it's centered nicely
        df_display["__DETAILS__"] = df_display["__DETAILS__"].apply(lambda _: "**➕**")

        return kpis, fig, df_display.to_dict("records")

    # Capture clicks on the Details glyph or on the Player cell to set selected player
    @app.callback(
        Output("alltime-selected-player", "data"),
        Input("alltime-table", "active_cell"),
        State("alltime-table", "data"),
        prevent_initial_call=True,
    )
    def set_selected_player(active_cell, table_rows):
        if not active_cell or not table_rows:
            return no_update
        row_idx = active_cell.get("row")
        col_id = active_cell.get("column_id")
        if row_idx is None or row_idx < 0 or row_idx >= len(table_rows):
            return no_update
        row = table_rows[row_idx]
        if col_id in ("__DETAILS__", "Player"):
            return {"player": row.get("Player", "")}
        return no_update

    # Render the Player Season Breakdown card
    @app.callback(
        Output("alltime-player-details-card", "style"),
        Output("alltime-player-header", "children"),
        Output("alltime-player-season-table", "data"),
        Input("alltime-selected-player", "data"),
        State("alltime-store-raw", "data"),
        State("alltime-season-range", "value"),
        State("alltime-teams", "value"),
        State("alltime-positions", "value"),
        State("alltime-game-type", "value"),
        prevent_initial_call=True,
    )
    def render_player_breakdown(sel, raw, season_range, teams, positions, game_type):
        if not sel or not sel.get("player") or not raw:
            return {"display":"none"}, "", []
        player = sel["player"]
        df_all = pd.DataFrame(raw)
        teams = teams or []
        positions = positions or []
        season_range = season_range or [2000, 2025]

        gb = _season_breakdown_for_player(
            df_all=df_all,
            player=player,
            season_range=(int(season_range[0]), int(season_range[1])),
            teams_filter=teams,
            positions_filter=positions,
            game_type=game_type or "all",
        )

        if gb.empty:
            header = html.H4([player, html.Small(" — no rows for current filters", className="ms-2 text-muted")])
            return {"display":"block"}, header, []

        # Build header with player & team years summary
        team_years = gb.groupby("Team")["Season"].apply(lambda s: ", ".join(s)).reset_index()
        team_badges = [
            dbc.Badge(f"{r.Team}: {r.Season}", className="me-1", color="primary", pill=True)
            for r in team_years.itertuples(index=False)
        ]
        header = html.Div([
            html.H4(player, className="mb-1"),
            html.Div(team_badges),
        ])

        # Columns already defined in DataTable; just pass rows
        rows = gb[[
            c for c in [
                "Season","Team","Games","Minutes",
                "PTS","PTS_PG","PTS_36",
                "AST","AST_PG","AST_36",
                "REB","REB_PG","REB_36",
                "STL","BLK",
                "FGA","3PA","FTA",
                "FG_PCT","TP_PCT","FT_PCT","eFG%","TS%",
            ] if c in gb.columns
        ]].to_dict("records")

        return {"display":"block"}, header, rows

    @app.callback(
        Output("alltime-store-teardown", "data"),
        Input("alltime-btn-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_filters(n):
        # You can wire JS to clear component values if desired; here it's a no-op.
        return no_update

    @app.callback(
        Output("alltime-download", "data"),
        Input("alltime-btn-download", "n_clicks"),
        State("alltime-table", "data"),
        prevent_initial_call=True,
    )
    def download_csv(n, table_rows):
        df = pd.DataFrame(table_rows or [])
        if df.empty:
            return no_update
        # Remove the glyph column on export
        if "__DETAILS__" in df.columns:
            df = df.drop(columns=["__DETAILS__"])
        return dcc.send_data_frame(df.to_csv, "all_time_leaders.csv", index=False)

    return app


# --------------------------------------------------------------------------------------
# HELPERS (KPI Cards & Labels)
# --------------------------------------------------------------------------------------
def _metric_label(metric_key: str, norm: str) -> str:
    label_map = {"per_game": "per game", "per_36": "per 36", "totals": "totals"}
    if metric_key in {"FG_PCT","TP_PCT","FT_PCT","eFG%","TS%"}:
        return metric_key
    return f"{metric_key} ({label_map.get(norm, norm)})"

def _kpi_card(title: str, value: str, sub: str = "") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(title, className="text-muted", style={"fontSize":"0.9rem"}),
                html.H4(value, className="mb-0"),
                html.Small(sub, className="text-secondary")
            ]),
            style={"background":"#171b27","border":"1px solid #243047","borderRadius":"14px"}
        ),
        xs=6, md=3
    )

def _kpi_cards(n_players: int, avg_games: float, avg_minutes: float, top_val: float,
               metric_key: str="PTS", norm: str="per_game"):
    return [
        _kpi_card("Players", f"{n_players:,}"),
        _kpi_card("Avg Games", f"{avg_games:.1f}"),
        _kpi_card("Avg Minutes", f"{avg_minutes:.1f}"),
        _kpi_card("Top Value", f"{top_val:.2f}", _metric_label(metric_key, norm)),
    ]
