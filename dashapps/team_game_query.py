# dashapps/team_game_query.py
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Dict, Any

import dash
from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL, no_update
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# -----------------------------------------------------------------------------
# DB ENGINE
# -----------------------------------------------------------------------------
DATABASE_URI = os.environ.get(
    "DATABASE_URI",
    "postgresql://philipmulryne@localhost:5432/basketball_data",
)
engine = create_engine(DATABASE_URI, pool_pre_ping=True)

# -----------------------------------------------------------------------------
# Season helpers
# -----------------------------------------------------------------------------
def generate_season_table_names(start_season: str, end_season: str, exceptions: Optional[List[str]] = None) -> List[str]:
    exceptions = exceptions or []
    start_year = int(start_season.split("_")[0])
    end_year = int(end_season.split("_")[0])
    names: List[str] = []
    for year in range(start_year, end_year + 1):
        next_year = str(year + 1)[-2:]
        season_str = f"{year}_{next_year}"
        if season_str not in exceptions:
            names.append(f"processed_games_{season_str}")
    return names

# Extended through 2025_26
CANDIDATE_SEASON_TABLES = generate_season_table_names(
    "1985_86", "2025_26", exceptions=["2019_20", "1986_87"]
)

def _existing_tables(engine, candidates: List[str]) -> List[str]:
    if not candidates:
        return []
    placeholders = ", ".join([f":t{i}" for i in range(len(candidates))])
    sql = text(
        f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ({placeholders})
        """
    )
    params = {f"t{i}": t for i, t in enumerate(candidates)}
    try:
        df = pd.read_sql(sql, engine, params=params)
        present = set(df["table_name"].str.lower())
        return [t for t in candidates if t.lower() in present]
    except Exception:
        return candidates

# -----------------------------------------------------------------------------
# Column normalization & filter metadata
# -----------------------------------------------------------------------------
FILTERABLE_COLUMNS: List[Tuple[str, str, str]] = [
    # Box-score staples
    ("points", "Points", "numeric"),
    ("defensive_rebounds", "Defensive Rebounds", "numeric"),
    ("offensive_rebounds", "Offensive Rebounds", "numeric"),
    ("total_rebounds", "Total Rebounds", "numeric"),
    ("assists", "Assists", "numeric"),
    ("blocks", "Blocks", "numeric"),

    # Turnovers / Steals
    ("turnovers", "Turnovers (player rows)", "numeric"),
    ("turnovers_teamonly", "Turnovers – Team only", "numeric"),
    ("turnovers_total", "Turnovers – Total (player+team)", "numeric"),
    ("steals", "Steals", "numeric"),

    # Fouls & misc
    ("fouls", "Fouls", "numeric"),
    ("efficiency", "Efficiency", "numeric"),
    ("team_plus_minus", "Plus/Minus (team from score)", "numeric"),
    ("points_allowed", "Points Allowed", "numeric"),

    # Shooting – raw counts
    ("fgm", "FGM (Field Goals Made)", "numeric"),
    ("fga", "FGA (Field Goals Attempted)", "numeric"),
    ("fg3m", "3PM (Threes Made)", "numeric"),
    ("fg3a", "3PA (Threes Attempted)", "numeric"),
    ("fg2m", "2PM (Twos Made)", "numeric"),
    ("fg2a", "2PA (Twos Attempted)", "numeric"),
    ("ftm", "FTM (Free Throws Made)", "numeric"),
    ("fta", "FTA (Free Throws Attempted)", "numeric"),

    # Shooting – % and advanced (derived after aggregation)
    ("fg_pct", "FG%", "numeric"),
    ("fg3_pct", "3P%", "numeric"),
    ("fg2_pct", "2P%", "numeric"),
    ("ft_pct", "FT%", "numeric"),
    ("efg_pct", "eFG%", "numeric"),
    ("ts_pct", "TS%", "numeric"),
    ("ftr", "FTr (FTA/FGA)", "numeric"),

    # text keys
    ("season", "Season", "text"),
    ("team", "Team", "text"),
    ("opponent", "Opponent", "text"),
    ("home_away", "Home/Away (H/A)", "text"),
    ("game_type", "Game Type (regular/playoff)", "text"),
    ("week", "Week", "text"),
]

# Columns that must NOT be summed; recompute after aggregation
DERIVED_RATE_COLUMNS = {
    "fg_pct", "fg3_pct", "fg2_pct", "ft_pct", "efg_pct", "ts_pct", "ftr"
}

# For convenience later when formatting
PERCENT_COLUMNS = ["fg_pct", "fg3_pct", "fg2_pct", "ft_pct", "efg_pct", "ts_pct", "ftr"]

COLUMN_OPTIONS = [{"label": label, "value": name} for name, label, _ in FILTERABLE_COLUMNS]
NUMERIC_OPERATORS = [{"label": op, "value": op} for op in ("=", ">", "<", ">=", "<=")]
TEXT_OPERATORS = [{"label": "=", "value": "="}, {"label": "LIKE (contains)", "value": "LIKE"}]

def _col_type(col: str) -> Optional[str]:
    for name, _label, ctype in FILTERABLE_COLUMNS:
        if name == col:
            return ctype
    return None

def _present_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [
        name for name, _label, ctype in FILTERABLE_COLUMNS
        if ctype == "numeric" and name in df.columns
    ]

GAME_KEY_CANDIDATES = [
    "game_code", "game_id", "gid", "match_id", "gameid", "gc", "game_code_int", "date"
]

def _pick_game_key(df: pd.DataFrame) -> Optional[str]:
    for k in GAME_KEY_CANDIDATES:
        if k in df.columns:
            return k
    return None

# -----------------------------------------------------------------------------
# Opponent / HomeAway / Score helpers
# -----------------------------------------------------------------------------
TEAM_ROW_TOKENS = {
    "TEAM", "TEAM TOTAL", "TEAM TOTALS", "TAKIM", "TAKIMI", "TAKIM TOPLAM", "TOPLAM",
    "—", "-", "", "NAN"
}

STEAL_ALIASES = ["steals", "stl", "top_calma", "top çalma", "top_çalma", "çalma", "topcalma", "top_calma"]
TURNOVER_ALIASES = ["turnovers", "tov", "to", "top_kaybi", "top kaybı", "kayip", "kaybı", "topkaybi", "top_kaybı"]

# ---- UPDATED alias map (matches your actual data names) ----
SHOOTING_ALIASES: Dict[str, List[str]] = {
    # field goals overall
    "fgm": [
        "fgm", "field_goals_made", "fieldgoals_made", "fg_made", "fg_m",
        "isabet", "sahaici_isabet", "saha içi isabet", "saha ici isabet"
    ],
    "fga": [
        "fga", "field_goals_attempts", "fieldgoals_attempts", "fg_att", "fg_a",
        "deneme", "sahaici_deneme", "saha içi deneme", "saha ici deneme"
    ],

    # 3-pointers
    "fg3m": [
        "fg3m", "3pm", "three_points_made", "three_pointers_made", "3pt_made",
        "uc_sayi_isabet", "üç sayı isabet", "ucluk_isabet", "üçlük isabet", "ucluk isabet"
    ],
    "fg3a": [
        "fg3a", "3pa", "three_points_attempts", "three_pointers_attempted", "3pt_att",
        "uc_sayi_deneme", "üç sayı deneme", "ucluk_deneme", "üçlük deneme", "ucluk deneme"
    ],

    # 2-pointers
    "fg2m": [
        "fg2m", "2pm", "two_points_made", "two_pointers_made", "2pt_made",
        "iki_sayi_isabet", "iki sayı isabet", "ikilik_isabet", "ikilik isabet"
    ],
    "fg2a": [
        "fg2a", "2pa", "two_points_attempts", "two_pointers_attempted", "2pt_att",
        "iki_sayi_deneme", "iki sayı deneme", "ikilik_deneme", "ikilik deneme"
    ],

    # free throws
    "ftm": [
        "ftm", "free_throws_made", "ft_made",
        "serbest_atış_isabet", "serbest atış isabet", "faul_atışı_isabet"
    ],
    "fta": [
        "fta", "free_throws_attempts", "ft_att",
        "serbest_atış_deneme", "serbest atış deneme", "faul_atışı_deneme"
    ],

    # percentages (if present at row level; recomputed post-agg anyway)
    "fg_pct": ["fg%", "fg_pct", "field_goals_pct", "sahaici_yuzde", "saha içi yüzde", "saha ici yuzde"],
    "fg3_pct": ["3p%", "fg3_pct", "three_pointers_pct", "ucluk_yuzde", "üçlük yüzde", "ucluk yuzde"],
    "fg2_pct": ["2p%", "fg2_pct", "two_pointers_pct", "ikilik_yuzde", "ikilik yüzde", "ikilik yuzde"],
    "ft_pct": ["ft%", "ft_pct", "free_throws_pct", "serbest_atış_yuzde", "serbest atış yüzde", "serbest atis yuzde"],
}

def _apply_alias_renames(df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> pd.DataFrame:
    cols_low = {c.lower().strip(): c for c in df.columns}
    rename_map: Dict[str, str] = {}
    for canon, aliases in alias_map.items():
        for a in aliases:
            key = a.lower().strip()
            if key in cols_low:
                rename_map[cols_low[key]] = canon
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _safe_div(n, d):
    n = pd.to_numeric(n, errors="coerce")
    d = pd.to_numeric(d, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d == 0, np.nan, n / d)
    return pd.Series(out)

def _normalize_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common Turkish/English aliases to canonical names (steals, turnovers, shooting)."""
    # Steals / TOs
    cols_low = {c.lower(): c for c in df.columns}
    rename_map = {}
    for a in STEAL_ALIASES:
        if a in cols_low:
            rename_map[cols_low[a]] = "steals"
            break
    for a in TURNOVER_ALIASES:
        if a in cols_low:
            rename_map[cols_low[a]] = "turnovers"
            break
    if rename_map:
        df = df.rename(columns=rename_map)

    # Shooting aliases
    df = _apply_alias_renames(df, SHOOTING_ALIASES)

    # Ensure existence & numeric types
    df = _ensure_numeric(df, ["steals", "turnovers"])
    shoot_counts = ["fgm", "fga", "fg3m", "fg3a", "fg2m", "fg2a", "ftm", "fta"]
    df = _ensure_numeric(df, shoot_counts)

    # Optional row-level % (we will recompute post-aggregation anyway)
    for pct in ["fg_pct", "fg3_pct", "fg2_pct", "ft_pct"]:
        if pct in df.columns:
            df[pct] = pd.to_numeric(df[pct], errors="coerce")

    # Synthesize points if absent
    if "points" not in df.columns:
        for c in ["fg2m", "fg3m", "ftm"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["points"] = 2.0 * df["fg2m"] + 3.0 * df["fg3m"] + 1.0 * df["ftm"]

    return df

def _add_opponent_and_ha(df: pd.DataFrame) -> pd.DataFrame:
    if "team" in df.columns and "home_team" in df.columns and "away_team" in df.columns:
        df = df.copy()
        df["home_away"] = np.where(df["team"] == df["home_team"], "H",
                            np.where(df["team"] == df["away_team"], "A", None))
        df["opponent"] = np.where(df["home_away"] == "H", df["away_team"],
                           np.where(df["home_away"] == "A", df["home_team"], None))
    else:
        if "home_away" not in df.columns:
            df["home_away"] = None
        if "opponent" not in df.columns:
            df["opponent"] = None
    return df

def _parse_score_pair(score: str) -> Optional[Tuple[int, int]]:
    if not isinstance(score, str):
        return None
    try:
        parts = score.replace(" ", "").split("-")
        if len(parts) != 2:
            return None
        a, b = int(parts[0]), int(parts[1])
        return a, b
    except Exception:
        return None

def _inject_team_pm_and_points_allowed(df: pd.DataFrame) -> pd.DataFrame:
    required = {"score", "home_team", "away_team", "team"}
    if not required.issubset(df.columns):
        if "team_plus_minus" not in df.columns:
            df["team_plus_minus"] = np.nan
        if "points_allowed" not in df.columns:
            df["points_allowed"] = np.nan
        return df
    df = df.copy()
    team_pm = []
    pts_allowed = []
    for _, r in df.iterrows():
        pair = _parse_score_pair(r.get("score"))
        if not pair:
            team_pm.append(np.nan)
            pts_allowed.append(np.nan)
            continue
        home_pts, away_pts = pair
        if r.get("team") == r.get("home_team"):
            team_pm.append(home_pts - away_pts)
            pts_allowed.append(away_pts)
        elif r.get("team") == r.get("away_team"):
            team_pm.append(away_pts - home_pts)
            pts_allowed.append(home_pts)
        else:
            team_pm.append(np.nan)
            pts_allowed.append(np.nan)
    df["team_plus_minus"] = team_pm
    df["points_allowed"] = pts_allowed
    return df

def _mark_team_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "player_name" not in df.columns:
        df["is_team_row"] = False
        return df
    s = df["player_name"].astype(str).str.strip().str.upper().fillna("NAN")
    df["is_team_row"] = s.eq("") | s.isin(TEAM_ROW_TOKENS)
    return df

# ---------- Detect and fix steals/turnovers inversion ----------
def _detect_and_fix_stl_tov_inversion(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Detects if 'steals' and 'turnovers' are swapped in the dataset.
    Heuristic:
      - Look at per-team-game sums; if median(steals) >> median(turnovers)
        or a large share of games have steals>=15 and turnovers<=6, assume swapped.
    Manual overrides:
      - TGQ_FORCE_SWAP_STL_TOV=1 forces swap for all data.
      - TGQ_SWAP_STL_TOV_SEASONS="2025_26;2024_25" forces swap only for those seasons.
    """
    force_all = os.environ.get("TGQ_FORCE_SWAP_STL_TOV", "0") == "1"
    swap_seasons_env = os.environ.get("TGQ_SWAP_STL_TOV_SEASONS", "").strip()
    swap_seasons = set([s.strip() for s in swap_seasons_env.split(";") if s.strip()])

    if "steals" not in df.columns or "turnovers" not in df.columns:
        return df, False

    if not df.empty and "season" in df.columns and swap_seasons:
        seasons_in_df = set(df["season"].dropna().unique().tolist())
        if seasons_in_df.issubset(swap_seasons):
            s = df["steals"].copy()
            df["steals"] = df["turnovers"]
            df["turnovers"] = s
            return df, True

    if force_all:
        s = df["steals"].copy()
        df["steals"] = df["turnovers"]
        df["turnovers"] = s
        return df, True

    keys_pref = ["season", "team", "opponent", "home_team", "away_team", "score", "game_date"]
    gkeys = [k for k in keys_pref if k in df.columns]
    if not gkeys:
        med_stl = float(pd.to_numeric(df["steals"], errors="coerce").median())
        med_tov = float(pd.to_numeric(df["turnovers"], errors="coerce").median())
        if med_stl - med_tov >= 4.0:
            s = df["steals"].copy()
            df["steals"] = df["turnovers"]
            df["turnovers"] = s
            return df, True
        return df, False

    tmp = df.copy()
    pg = tmp.groupby(gkeys, as_index=False)[["steals", "turnovers"]].sum(min_count=1)
    if pg.empty:
        return df, False

    med_stl = float(pg["steals"].median())
    med_tov = float(pg["turnovers"].median())
    suspicious = ((pg["steals"] >= 15) & (pg["turnovers"] <= 6)).mean()
    if (med_stl - med_tov >= 4.0) or (suspicious >= 0.40):
        s = df["steals"].copy()
        df["steals"] = df["turnovers"]
        df["turnovers"] = s
        return df, True
    return df, False

# ---------------------------------------------------------------------
# Turnover variants
def _add_turnover_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "turnovers" not in df.columns:
        df["turnovers"] = 0.0
    if "steals" not in df.columns:
        df["steals"] = 0.0
    if "is_team_row" not in df.columns:
        df = _mark_team_only_rows(df)

    df["turnovers_player"] = np.where(df["is_team_row"], 0.0, df["turnovers"]).astype(float)
    df["turnovers_teamonly"] = np.where(df["is_team_row"], df["turnovers"], 0.0).astype(float)

    team_comp = 0.0
    for cand in ("team_turnovers", "turnovers_team", "tov_team", "team_to"):
        if cand in df.columns:
            team_comp = pd.to_numeric(df[cand], errors="coerce").fillna(0.0)
            break
    if isinstance(team_comp, float):
        df["turnovers_team_component"] = 0.0
    else:
        df["turnovers_team_component"] = team_comp

    df["turnovers_total"] = (df["turnovers_player"] + df["turnovers_teamonly"]).astype(float)
    return df

# ---------------------------------------------------------------------
# Shooting variants & derived
def _add_shooting_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create safe, player-only shooting counters to avoid double counting 'TEAM TOTAL' rows.
    Also standardize raw columns presence.
    """
    df = df.copy()
    df = _ensure_numeric(df, ["fgm", "fga", "fg3m", "fg3a", "fg2m", "fg2a", "ftm", "fta"])
    if "is_team_row" not in df.columns:
        df = _mark_team_only_rows(df)

    # Player-only versions (team rows zeroed)
    for base in ["fgm", "fga", "fg3m", "fg3a", "fg2m", "fg2a", "ftm", "fta"]:
        df[f"{base}_player"] = np.where(df["is_team_row"], 0.0, df[base]).astype(float)

    return df

def _recompute_shooting_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute percentages & advanced rates from SUMS (post-aggregation).
    Assumes df has fgm/fga, fg3m/fg3a, fg2m/fg2a, ftm/fta, and points.
    """
    df = df.copy()
    if "fga" in df.columns and "fgm" in df.columns:
        df["fg_pct"] = _safe_div(df["fgm"], df["fga"])
    if "fg3a" in df.columns and "fg3m" in df.columns:
        df["fg3_pct"] = _safe_div(df["fg3m"], df["fg3a"])
    if "fg2a" in df.columns and "fg2m" in df.columns:
        df["fg2_pct"] = _safe_div(df["fg2m"], df["fg2a"])
    if "fta" in df.columns and "ftm" in df.columns:
        df["ft_pct"] = _safe_div(df["ftm"], df["fta"])

    # eFG% = (FGM + 0.5*3PM) / FGA
    if set(["fgm", "fg3m", "fga"]).issubset(df.columns):
        df["efg_pct"] = _safe_div(df["fgm"] + 0.5 * df["fg3m"], df["fga"])

    # TS% = Points / (2*(FGA + 0.44*FTA))
    if set(["points", "fga", "fta"]).issubset(df.columns):
        denom = 2.0 * (pd.to_numeric(df["fga"], errors="coerce").fillna(0.0) +
                       0.44 * pd.to_numeric(df["fta"], errors="coerce").fillna(0.0))
        df["ts_pct"] = _safe_div(df["points"], denom)

    # Free Throw Rate (FTr) = FTA / FGA
    if set(["fta", "fga"]).issubset(df.columns):
        df["ftr"] = _safe_div(df["fta"], df["fga"])

    return df

# -----------------------------------------------------------------------------
# Optional PageShell wrapper
# -----------------------------------------------------------------------------
try:
    from .common.page_shell import page_shell as PageShell
except Exception:
    try:
        from .common import page_shell as PageShell  # pragma: no cover
    except Exception:
        PageShell = None  # type: ignore

if PageShell is not None and not callable(PageShell):
    PageShell = getattr(PageShell, "page_shell", None)  # type: ignore

def _wrap_with_pageshell(content, title: str):
    if callable(PageShell):
        try:
            return PageShell(content=content, title=title)
        except TypeError:
            try:
                return PageShell(content, title)
            except Exception:
                return None
    return content

# -----------------------------------------------------------------------------
# Fallback CSV loader (optional)
# -----------------------------------------------------------------------------
def _load_csv_fallback() -> Optional[pd.DataFrame]:
    """
    If DB tables aren't available, you can set TGQ_CSV to a file path to test.
    """
    path = os.environ.get("TGQ_CSV")
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[tgq] CSV fallback failed: {e}")
            return None
    # Look for the user's uploaded sample (newer id first)
    for cand in (
        "/mnt/data/data-1761721762871.csv",
        "/mnt/data/data-1761690566618.csv",
    ):
        try:
            if os.path.exists(cand):
                return pd.read_csv(cand)
        except Exception as e:
            print(f"[tgq] CSV fallback {cand} failed: {e}")
    return None

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def create_dash_team_game_query(
    server,
    base_pathname: str = "/team_game_query/",
    title: str = "Team Game Stats Query",
) -> Dash:
    season_tables = _existing_tables(engine, CANDIDATE_SEASON_TABLES)

    app = Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
    )

    # ---------------- Layout ----------------
    layout_core = html.Div([
        html.H1("Team Game Stats Query"),

        dcc.Interval(id="tgq-boot", interval=250, n_intervals=0, max_intervals=1),

        html.Label("Season Range (slider):"),
        dcc.RangeSlider(
            id="tgq-season-range-slider",
            # Selecting 2025 includes 2025_26
            min=1985, max=2025, step=1, value=[1985, 2025],
            marks={y: str(y) for y in range(1985, 2026, 5)},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(id="tgq-slider-output", style={"marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("Select Team:"),
                dcc.Dropdown(id="tgq-team-dropdown", options=[], placeholder="Select a team"),
            ], style={"flex": 1, "marginRight": "12px"}),

            html.Div([
                html.Label("Opponent (optional):"),
                dcc.Dropdown(id="tgq-opponent-dropdown", options=[], placeholder="Filter by opponent", multi=True),
            ], style={"flex": 1, "marginRight": "12px"}),

            html.Div([
                html.Label("Home/Away:"),
                dcc.RadioItems(
                    id="tgq-home-away",
                    options=[
                        {"label": "All", "value": "all"},
                        {"label": "Home", "value": "H"},
                        {"label": "Away", "value": "A"},
                    ],
                    value="all",
                    inline=True,
                ),
            ], style={"flex": 1}),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "10px"}),

        html.Div([
            html.Div([
                html.Label("Game Type:"),
                dcc.RadioItems(
                    id="tgq-game-type",
                    options=[
                        {"label": "All", "value": "all"},
                        {"label": "Regular", "value": "regular"},
                        {"label": "Playoff", "value": "playoff"},
                    ],
                    value="all",
                    inline=True,
                ),
            ], style={"flex": 1, "marginRight": "12px"}),

            html.Div([
                html.Label("Select Seasons (optional):"),
                dcc.Dropdown(id="tgq-season-dropdown", options=[], multi=True,
                             placeholder="Select seasons (default: all)"),
            ], style={"flex": 1, "marginRight": "12px"}),

            html.Div([
                html.Label("Since Season (optional):"),
                dcc.Dropdown(id="tgq-since-season-dropdown", options=[], placeholder="Select a starting season"),
            ], style={"flex": 1}),
        ], style={"display": "flex", "gap": "12px"}),

        html.Label("Aggregation:"),
        dcc.RadioItems(
            id="tgq-agg-mode",
            options=[
                {"label": "Team per game (with opponent & H/A)", "value": "per_game"},
                {"label": "Season totals (by team)", "value": "season_total"},
                {"label": "None (show raw rows)", "value": "none"},
            ],
            value="per_game",
            inline=True,
        ),

        html.Hr(),
        html.H2("Additional Column Filters"),
        html.Div(
            id="tgq-filter-container",
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
                    children=[
                        dcc.Dropdown(
                            id={"type": "tgq-column-dropdown", "index": 0},
                            options=COLUMN_OPTIONS,
                            placeholder="Select a column (e.g., Points)",
                            style={"width": "260px", "marginRight": "10px"},
                        ),
                        dcc.Dropdown(
                            id={"type": "tgq-operator-dropdown", "index": 0},
                            options=[],
                            placeholder="Select operator",
                            style={"width": "160px", "marginRight": "10px"},
                        ),
                        dcc.Input(
                            id={"type": "tgq-value-input", "index": 0},
                            type="text",
                            placeholder="Enter filter value",
                            style={"width": "180px", "marginRight": "16px"},
                        ),
                        html.Button("Remove", id={"type": "tgq-remove-filter", "index": 0}, n_clicks=0),
                    ],
                )
            ],
        ),
        html.Button("Add Filter", id="tgq-add-filter-button", n_clicks=0, style={"marginTop": "10px"}),

        html.Div([
            dcc.Checklist(
                id="tgq-unique-results-checkbox",
                options=[{"label": "Show only unique teams (post-aggregation)", "value": "unique"}],
                value=[],
            )
        ], style={"marginTop": "16px"}),

        html.Button("Query", id="tgq-query-button", n_clicks=0, style={"marginTop": "16px"}),

        html.Div(id="tgq-output-table", style={"marginTop": "20px"}),
    ])

    wrapped = _wrap_with_pageshell(layout_core, title)
    app.layout = wrapped if wrapped is not None else layout_core

    # ---------------- Callbacks ----------------
    @app.callback(
        outputs=(
            Output("tgq-team-dropdown", "options"),
            Output("tgq-season-dropdown", "options"),
            Output("tgq-since-season-dropdown", "options"),
            Output("tgq-opponent-dropdown", "options"),
        ),
        inputs=(Input("tgq-boot", "n_intervals"), Input("tgq-query-button", "n_clicks")),
        prevent_initial_call=False,
    )
    def populate_dropdowns(_boot, _clicks):
        season_tables = _existing_tables(engine, CANDIDATE_SEASON_TABLES)
        if not season_tables:
            df = _load_csv_fallback()
            if df is None or df.empty:
                return [], [], [], []
            df = _normalize_stats(df)
            df = _add_opponent_and_ha(df)
            df, _ = _detect_and_fix_stl_tov_inversion(df)
            team_opts = [{"label": t, "value": t} for t in sorted(pd.Series(df.get("team", pd.Series([]))).dropna().unique())]
            season_vals = sorted(pd.Series(df.get("season", pd.Series([]))).dropna().unique())
            season_opts = [{"label": s, "value": s} for s in season_vals]
            opp_vals = sorted(pd.Series(df.get("opponent", pd.Series([]))).dropna().unique())
            opp_opts = [{"label": t, "value": t} for t in opp_vals]
            return team_opts, season_opts, season_opts, opp_opts

        try:
            team_sql = " UNION ".join([f"SELECT DISTINCT team FROM {tbl}" for tbl in season_tables])
            season_sql = " UNION ".join([f"SELECT DISTINCT season FROM {tbl}" for tbl in season_tables])
            teams_df = pd.read_sql(team_sql, engine)
            seasons_df = pd.read_sql(season_sql, engine)

            try:
                ha_sql = " UNION ".join([
                    f"SELECT DISTINCT home_team AS t FROM {tbl} WHERE home_team IS NOT NULL"
                    for tbl in season_tables
                ]) + " UNION " + " UNION ".join([
                    f"SELECT DISTINCT away_team AS t FROM {tbl} WHERE away_team IS NOT NULL"
                    for tbl in season_tables
                ])
                ha_df = pd.read_sql(ha_sql, engine)
                opp_vals = sorted(pd.Series(pd.concat([ha_df["t"], teams_df["team"]], ignore_index=True)).dropna().unique())
            except Exception:
                opp_vals = sorted(teams_df["team"].dropna().unique())

            team_opts = [{"label": t, "value": t} for t in sorted(teams_df["team"].dropna().unique())]
            season_vals = sorted(seasons_df["season"].dropna().unique())
            season_opts = [{"label": s, "value": s} for s in season_vals]
            opp_opts = [{"label": t, "value": t} for t in opp_vals]
            return team_opts, season_opts, season_opts, opp_opts
        except Exception as e:
            print(f"[tgq] dropdown fetch error: {e}")
            return [], [], [], []

    @app.callback(
        Output({"type": "tgq-operator-dropdown", "index": MATCH}, "options"),
        Input({"type": "tgq-column-dropdown", "index": MATCH}, "value"),
    )
    def set_operator_options(selected_column):
        if not selected_column:
            return []
        return NUMERIC_OPERATORS if _col_type(selected_column) == "numeric" else TEXT_OPERATORS

    @app.callback(
        Output("tgq-filter-container", "children"),
        inputs=(Input("tgq-add-filter-button", "n_clicks"),
                Input({"type": "tgq-remove-filter", "index": ALL}, "n_clicks")),
        state=(State("tgq-filter-container", "children"),),
        prevent_initial_call=True,
    )
    def modify_filter_rows(add_clicks, remove_clicks, existing_rows):
        ctx = dash.callback_context
        existing_rows = list(existing_rows or [])
        if not ctx.triggered:
            return existing_rows
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        if trig == "tgq-add-filter-button":
            new_index = len(existing_rows)
            existing_rows.append(
                html.Div(
                    style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
                    children=[
                        dcc.Dropdown(
                            id={"type": "tgq-column-dropdown", "index": new_index},
                            options=COLUMN_OPTIONS,
                            placeholder="Select a column (e.g., Points)",
                            style={"width": "260px", "marginRight": "10px"},
                        ),
                        dcc.Dropdown(
                            id={"type": "tgq-operator-dropdown", "index": new_index},
                            options=[],
                            placeholder="Select operator",
                            style={"width": "160px", "marginRight": "10px"},
                        ),
                        dcc.Input(
                            id={"type": "tgq-value-input", "index": new_index},
                            type="text",
                            placeholder="Enter filter value",
                            style={"width": "180px", "marginRight": "16px"},
                        ),
                        html.Button("Remove", id={"type": "tgq-remove-filter", "index": new_index}, n_clicks=0),
                    ],
                )
            )
            return existing_rows
        try:
            fired = eval(trig)  # {'type': 'tgq-remove-filter', 'index': k}
            if isinstance(fired, dict) and fired.get("type") == "tgq-remove-filter":
                idx = fired.get("index")
                if isinstance(idx, int) and 0 <= idx < len(existing_rows):
                    existing_rows.pop(idx)
        except Exception:
            pass
        return existing_rows

    @app.callback(
        Output("tgq-slider-output", "children"),
        Input("tgq-season-range-slider", "value"),
    )
    def update_slider_output(slider_range):
        if not slider_range:
            return "No season range selected."
        a, b = slider_range
        return f"Selected season range: {a} to {b} (selecting 2025 includes 2025_26)."

    @app.callback(
        Output("tgq-output-table", "children"),
        Input("tgq-query-button", "n_clicks"),
        State("tgq-team-dropdown", "value"),
        State("tgq-opponent-dropdown", "value"),
        State("tgq-home-away", "value"),
        State("tgq-game-type", "value"),
        State("tgq-season-dropdown", "value"),
        State("tgq-since-season-dropdown", "value"),
        State({"type": "tgq-column-dropdown", "index": ALL}, "value"),
        State({"type": "tgq-operator-dropdown", "index": ALL}, "value"),
        State({"type": "tgq-value-input", "index": ALL}, "value"),
        State("tgq-unique-results-checkbox", "value"),
        State("tgq-season-range-slider", "value"),
        State("tgq-agg-mode", "value"),
        prevent_initial_call=True,
    )
    def query_with_multiple_filters(
        _n_clicks,
        team_name,
        opponents,
        home_away_sel,
        game_type_sel,
        selected_seasons,
        since_season,
        columns,
        operators,
        values,
        unique_results,
        slider_range,
        agg_mode,
    ):
        season_tables = _existing_tables(engine, CANDIDATE_SEASON_TABLES)

        df: Optional[pd.DataFrame] = None
        params: Dict[str, Any] = {}
        where_clauses: List[str] = ["1=1"]

        # Slider range on first 4 digits of season
        if slider_range and len(slider_range) == 2:
            params["min_year"], params["max_year"] = int(slider_range[0]), int(slider_range[1])
            where_clauses.append(
                "CAST(SUBSTRING(season FROM 1 FOR 4) AS INT) BETWEEN :min_year AND :max_year"
            )

        if season_tables:
            if team_name:
                where_clauses.append("team = :team_name")
                params["team_name"] = team_name
            if selected_seasons:
                phs = []
                for i, s in enumerate(selected_seasons):
                    key = f"season_{i}"
                    params[key] = s
                    phs.append(f":{key}")
                where_clauses.append(f"season IN ({', '.join(phs)})")
            if since_season:
                where_clauses.append("season >= :since_season")
                params["since_season"] = since_season
            if game_type_sel and game_type_sel != "all":
                where_clauses.append("LOWER(game_type) = :game_type_sel")
                params["game_type_sel"] = game_type_sel.lower()

            union_sql = " UNION ALL ".join([f"SELECT * FROM {tbl}" for tbl in season_tables])
            sql = text(
                f"""
                SELECT *
                FROM (
                    {union_sql}
                ) AS all_seasons
                WHERE {' AND '.join(where_clauses)}
                """
            )
            try:
                df = pd.read_sql(sql, engine, params=params)
            except Exception as e:
                return html.Div(f"Error fetching data: {e}")
        else:
            df = _load_csv_fallback()
            if df is None or df.empty:
                return html.Div("No season tables found and no CSV fallback available.")

            # pandas-side filters for CSV fallback
            if "season" in df.columns and slider_range and len(slider_range) == 2:
                a, b = slider_range
                df = df[df["season"].astype(str).str[:4].astype(int).between(a, b)]
            if team_name and "team" in df.columns:
                df = df[df["team"] == team_name]
            if selected_seasons and "season" in df.columns:
                df = df[df["season"].isin(selected_seasons)]
            if since_season and "season" in df.columns:
                df = df[df["season"] >= since_season]
            if game_type_sel and "game_type" in df.columns and game_type_sel != "all":
                df = df[df["game_type"].str.lower() == game_type_sel.lower()]

        if df is None or df.empty:
            return html.Div("No results found for the selected filters.")

        # ---- Enrich & normalize
        df = _normalize_stats(df)
        df = _add_opponent_and_ha(df)

        # Detect & fix steals/turnovers inversion
        df, swapped = _detect_and_fix_stl_tov_inversion(df)

        df = _inject_team_pm_and_points_allowed(df)
        df = _mark_team_only_rows(df)

        # Variants
        df = _add_turnover_variants(df)
        df = _add_shooting_variants(df)

        # Opponent filter (derived)
        if opponents:
            opp_set = set(opponents if isinstance(opponents, list) else [opponents])
            df = df[df["opponent"].isin(opp_set)]

        # H/A filter (derived)
        if home_away_sel in ("H", "A"):
            df = df[df["home_away"] == home_away_sel]

        # ---------------- TEAM AGGREGATION ----------------
        if agg_mode in ("per_game", "season_total"):
            # Drop strictly player-identity columns to group cleanly
            for col in ("player", "player_id", "player_name"):
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Build list of numeric columns present
            num_cols_all = _present_numeric_cols(df)

            # We never sum %/rate columns; compute after grouping
            num_cols_to_sum = [c for c in num_cols_all if c not in DERIVED_RATE_COLUMNS]

            # Always include the player-only shooting counters and TO variants if present
            for extra in [
                "fgm_player", "fga_player", "fg3m_player", "fg3a_player", "fg2m_player", "fg2a_player",
                "ftm_player", "fta_player",
                "turnovers_player", "turnovers_teamonly", "turnovers_team_component"
            ]:
                if extra in df.columns and extra not in num_cols_to_sum:
                    num_cols_to_sum.append(extra)

            if agg_mode == "per_game":
                game_key = _pick_game_key(df)
                key_candidates = [
                    "season", "team", "opponent", "home_away", "game_type",
                    "game_date", "game_time", "week", "home_team", "away_team", "score", game_key
                ]
                group_keys = [k for k in key_candidates if k and k in df.columns]
            else:
                group_keys = [c for c in ["season", "team", "game_type"] if c in df.columns]

            if group_keys:
                agg_map: Dict[str, Any] = {c: "sum" for c in num_cols_to_sum}
                if "turnovers_team_component" in agg_map:
                    agg_map["turnovers_team_component"] = "max"  # per-game component, avoid multiplying
                g = df.groupby(group_keys, as_index=False).agg(agg_map)
                df = g
            else:
                if "team" in df.columns and num_cols_to_sum:
                    df = df.groupby(["team"], as_index=False)[num_cols_to_sum].sum()

            # Recompose canonical totals from player-only counters
            def _maybe_copy(dst, src):
                if src in df.columns:
                    df[dst] = df[src]

            _maybe_copy("fgm", "fgm_player")
            _maybe_copy("fga", "fga_player")
            _maybe_copy("fg3m", "fg3m_player")
            _maybe_copy("fg3a", "fg3a_player")
            _maybe_copy("fg2m", "fg2m_player")
            _maybe_copy("fg2a", "fg2a_player")
            _maybe_copy("ftm", "ftm_player")
            _maybe_copy("fta", "fta_player")

            # Recompute turnover totals (player + teamonly + component)
            if "turnovers_player" in df.columns and "turnovers_teamonly" in df.columns:
                df["turnovers_total"] = df["turnovers_player"] + df["turnovers_teamonly"]
                if "turnovers_team_component" in df.columns:
                    df["turnovers_total"] = df["turnovers_total"] + df["turnovers_team_component"]

            # Drop raw 'turnovers' if present to avoid confusion
            df = df.drop(columns=["turnovers"], errors="ignore")

            # Recompute shooting rates from sums
            df = _recompute_shooting_rates(df)

            # Minor cleanup
            if "team_plus_minus" in df.columns:
                df = df.drop(columns=["plus_minus"], errors="ignore")

        # Unique teams checkbox
        if unique_results and "unique" in (unique_results or []) and "team" in df.columns:
            df = df.drop_duplicates(subset=["team"])

        if df.empty:
            return html.Div("No results after aggregation.")

        # ---------------- Summary + Output ----------------
        num_results = len(df)
        unique_teams = df["team"].nunique() if "team" in df.columns else "—"

        notes = []
        if swapped:
            notes.append("Applied auto-fix: swapped mislabeled Steals/Turnovers.")
        notes_text = " ".join(notes)

        summary = html.Div([
            html.Div(f"Rows: {num_results}", style={"marginBottom": "6px", "fontWeight": "bold"}),
            html.Div(f"Unique teams: {unique_teams}", style={"marginBottom": "10px", "fontWeight": "bold"}),
            html.Div(f"Aggregation: {agg_mode.replace('_',' ')}", style={"color": "#9aa3b2"}),
            html.Div(notes_text, style={"color": "#d08400"}) if notes_text else html.Div(),
        ])

        # Column ordering: keys -> other -> numeric (counts first, then %)
        preferred_keys = []
        for k in [
            _pick_game_key(df),
            "season", "team", "opponent", "home_away", "game_type",
            "game_date", "game_time", "week", "home_team", "away_team", "score"
        ]:
            if k and k in df.columns:
                preferred_keys.append(k)

        counts_priority = ["fgm", "fga", "fg3m", "fg3a", "fg2m", "fg2a", "ftm", "fta",
                           "points", "offensive_rebounds", "defensive_rebounds", "total_rebounds",
                           "assists", "steals", "blocks", "fouls",
                           "turnovers_player", "turnovers_teamonly", "turnovers_total"]
        rates_priority = ["fg_pct", "fg3_pct", "fg2_pct", "ft_pct", "efg_pct", "ts_pct", "ftr"]

        numeric_all = [c for c in _present_numeric_cols(df) if c in df.columns]
        numeric_counts = [c for c in counts_priority if c in numeric_all]
        numeric_rates = [c for c in rates_priority if c in df.columns]  # ensure presence
        numeric_rest = [c for c in numeric_all if c not in set(numeric_counts + numeric_rates)]

        other_cols = [c for c in df.columns if c not in set(preferred_keys + numeric_all)]
        ordered_cols = [c for c in preferred_keys + other_cols + numeric_counts + numeric_rates + numeric_rest if c in df.columns]

        # ----- Percentage formatting (two decimals with % sign) -----
        pct_set = set([c for c in PERCENT_COLUMNS if c in df.columns])
        columns_def = []
        for c in ordered_cols:
            col_def = {"name": c, "id": c}
            if c in pct_set:
                col_def.update({
                    "type": "numeric",
                    "format": Format(precision=2, scheme=Scheme.percentage)  # 0.4532 -> 45.32%
                })
            columns_def.append(col_def)

        return html.Div([
            summary,
            DataTable(
                id="tgq-results-table",
                columns=columns_def,
                data=df[ordered_cols].to_dict("records"),
                sort_action="native",
                page_size=50,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "80px", "maxWidth": "320px"},
                style_header={"fontWeight": "bold"},
                tooltip_header={
                    "efg_pct": "Effective FG% = (FGM + 0.5*3PM) / FGA",
                    "ts_pct": "True Shooting% = PTS / [2*(FGA + 0.44*FTA)]",
                    "ftr": "Free Throw Rate = FTA / FGA",
                },
            ),
        ])

    return app
    