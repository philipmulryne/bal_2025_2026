# dashapps/pos_ratings.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, dash_table

# Optional shell wrapper (renders a common header/nav if present)
try:
    from .common import page_shell as PageShell  # type: ignore
except Exception:
    PageShell = None


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _data_root() -> Path:
    """
    Resolve the data root in a robust order:
    1) DATA_ROOT env (respects your launcher)
    2) Your absolute Mac path (current BSL season)
    3) ./out_data (repo default)
    """
    env = os.environ.get("DATA_ROOT")
    if env:
        return Path(env).expanduser()
    mac_abs = Path("/Users/uygarkaraca/JSON/BSL_2025_26/out_data")
    if mac_abs.exists():
        return mac_abs
    return Path("out_data")


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p and p.exists():
                return p
        except Exception:
            pass
    return None


def _find_csv(base_names: List[str]) -> Optional[Path]:
    """
    Look for files in several common locations.
    """
    root = _data_root()
    candidates: List[Path] = []
    for bn in base_names:
        candidates += [
            root / "_cumulative" / bn,                           # preferred
            root / bn,                                           # direct
            Path("/mnt/data") / bn,                              # sandbox
            Path("/Users/uygarkaraca/JSON/BSL_2025_26/out_data/_cumulative") / bn,  # explicit
        ]
    return _first_existing(candidates)


def _read_csv_any(base_names: List[str]) -> pd.DataFrame:
    """
    Best-effort CSV/CSV.GZ/Parquet read for any of the provided base names.
    """
    p = _find_csv(base_names)
    if p is None:
        return pd.DataFrame()

    # Try plain CSV
    try:
        return pd.read_csv(p)
    except Exception:
        pass

    # Try .csv.gz (if p is *.csv -> turn into *.csv.gz)
    try:
        gz = p.with_suffix(p.suffix + ".gz") if p.suffix else p.with_suffix(".csv.gz")
        if gz.exists():
            return pd.read_csv(gz)
    except Exception:
        pass

    # Try Parquet sibling
    try:
        pq = p.with_suffix(".parquet")
        if pq.exists():
            return pd.read_parquet(pq)
    except Exception:
        pass

    return pd.DataFrame()


def _to_week_int(val) -> float:
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except Exception:
        pass
    import re
    m = re.search(r"(\d+)", str(val).strip().lower())
    if m:
        try:
            return float(int(m.group(1)))
        except Exception:
            return np.nan
    return np.nan


def _load_games_index() -> pd.DataFrame:
    df = _read_csv_any(["games_index.csv"])
    if df.empty:
        return df

    ren: Dict[str, str] = {}
    if "gid" in df.columns and "game_id" not in df.columns:
        ren["gid"] = "game_id"
    if "Week" not in df.columns:
        for alt in ["week", "GameWeek", "round", "Round", "GW"]:
            if alt in df.columns:
                ren[alt] = "Week"
                break
    if ren:
        df = df.rename(columns=ren)

    if "game_id" in df.columns:
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    if "Week" in df.columns:
        df["Week"] = df["Week"].apply(_to_week_int)
    return df


def _load_pbp() -> pd.DataFrame:
    df = _read_csv_any(["pbp_all.csv", "pop_all.csv"])
    if df.empty:
        return df

    ren: Dict[str, str] = {}
    if "game_id" not in df.columns:
        for c in ["gid", "GameId", "gameId", "game_code", "match_id", "id_game"]:
            if c in df.columns:
                ren[c] = "game_id"
                break
    if "team" not in df.columns:
        for c in ["Team", "team_name", "team_key", "club", "club_name", "HomeAwayTeam"]:
            if c in df.columns:
                ren[c] = "team"
                break
    if "opponent" not in df.columns:
        for c in ["Opp", "opponent_name", "opp_name", "OpponentTeam", "opponentTeam"]:
            if c in df.columns:
                ren[c] = "opponent"
                break
    if "points" not in df.columns:
        for c in ["pts", "delta", "score_points", "scored", "Points"]:
            if c in df.columns:
                ren[c] = "points"
                break
    if "possession_id" not in df.columns:
        for c in ["poss_id", "possession", "possessionId"]:
            if c in df.columns:
                ren[c] = "possession_id"
                break

    if ren:
        df = df.rename(columns=ren)

    if "points" in df.columns:
        df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
    if "game_id" in df.columns:
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Possession rollups
# ---------------------------------------------------------------------
@dataclass
class PossessionConfig:
    retain_after_tech_unsports: bool = True
    lookahead_inbound_rows: int = 6
    treb_is_def_reb: bool = True
    lane_violation_turnover: bool = False


CFG = PossessionConfig()


def _build_possessions_table(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp.empty:
        return pd.DataFrame(columns=["game_id", "team", "possessions", "pts_for"])

    need = {"game_id", "team", "possession_id", "points"}
    if need <= set(pbp.columns):
        g = (
            pbp.dropna(subset=["game_id", "team"])
            .groupby(["game_id", "team", "possession_id"], as_index=False)["points"]
            .sum()
            .rename(columns={"points": "poss_points"})
        )
        roll = (
            g.groupby(["game_id", "team"], as_index=False)
            .agg(
                possessions=("possession_id", "nunique"),
                pts_for=("poss_points", "sum"),
            )
        )
        return roll

    return pd.DataFrame(columns=["game_id", "team", "possessions", "pts_for"])


def _enrich_with_opponent(roll: pd.DataFrame, games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    if roll.empty:
        return roll

    if "opponent" in pbp.columns:
        opp = (
            pbp.groupby(["game_id", "team"])["opponent"]
            .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else None)
            .reset_index()
        )
    else:
        teams = pbp[["game_id", "team"]].dropna().drop_duplicates()
        pair = teams.merge(teams, on="game_id", suffixes=("", "_opp"))
        opp = pair[pair["team"] != pair["team_opp"]][["game_id", "team", "team_opp"]].drop_duplicates()
        opp = opp.rename(columns={"team_opp": "opponent"})

    df = roll.merge(opp, on=["game_id", "team"], how="left")

    opp_pts = roll.rename(
        columns={
            "team": "opponent",
            "pts_for": "pts_against",
            "possessions": "opp_possessions",
        }
    )
    df = df.merge(opp_pts[["game_id", "opponent", "pts_against"]], on=["game_id", "opponent"], how="left")

    if not games.empty:
        g2 = games.copy()
        if "Week" in g2.columns:
            g2["Week"] = g2["Week"].apply(_to_week_int)
        keep = ["game_id", "Week"] if "Week" in g2.columns else ["game_id"]
        df = df.merge(g2[keep], on="game_id", how="left")

    df["ORtg"] = np.where(df["possessions"] > 0, 100.0 * df["pts_for"] / df["possessions"], np.nan)
    df["DRtg"] = np.where(df["possessions"] > 0, 100.0 * df["pts_against"] / df["possessions"], np.nan)
    df["NetRtg"] = df["ORtg"] - df["DRtg"]

    cols = ["game_id", "Week", "team", "opponent", "possessions", "pts_for", "pts_against", "ORtg", "DRtg", "NetRtg"]
    return df[[c for c in cols if c in df.columns]]


# ---------------------------------------------------------------------
# Dash factory
# ---------------------------------------------------------------------
def _is_flask_app(obj) -> bool:
    try:
        return hasattr(obj, "wsgi_app") and hasattr(obj, "add_url_rule")
    except Exception:
        return False


def _unique_slug(flask_app=None, prefix: str = "posrat") -> str:
    """
    Generate a slug that won't collide with existing blueprints.
    Dash registers two blueprints per app; we avoid collisions here.
    """
    taken = set(getattr(flask_app, "blueprints", {}).keys()) if flask_app else set()
    for _ in range(24):
        slug = f"{prefix}_{uuid4().hex[:8]}"
        if (
            slug not in taken
            and f"_{slug}_dash_assets" not in taken
            and f"__{slug}_dash_assets" not in taken
        ):
            return slug
    return f"{prefix}_{uuid4().hex}"


def create_dash_pos_ratings(server=None, base_pathname: str = "/pos_ratings/", **_):
    embedded = _is_flask_app(server)
    dash_server = server if embedded else True
    url_prefix = base_pathname if embedded else None

    # === CRUCIAL: unique name + assets URL per instance ===
    slug = _unique_slug(server if embedded else None)            # e.g. "posrat_a1b2c3d4"
    assets_dir = os.path.join(tempfile.gettempdir(), f"dash_assets_{slug}")
    os.makedirs(assets_dir, exist_ok=True)

    app = Dash(
        name=slug,                                               # avoids "_pos_ratings_dash_assets"
        server=dash_server,
        url_base_pathname=url_prefix,
        assets_folder=assets_dir,                                # never None
        assets_url_path=f"/assets-{slug}",                       # unique assets blueprint URL
        suppress_callback_exceptions=True,
        title="Possessions & Ratings",
    )

    # ---------------------- Data ----------------------
    GAMES = _load_games_index()
    PBP = _load_pbp()

    if PBP.empty:
        app.layout = html.Div(
            [
                html.H2("Possessions & Ratings"),
                html.P(
                    "No play-by-play found (looked for pbp_all.csv / pop_all.csv in DATA_ROOT)."
                ),
            ]
        )
        return app

    ROLL = _build_possessions_table(PBP)
    METRICS = _enrich_with_opponent(ROLL, GAMES, PBP)

    if "Week" in METRICS.columns and METRICS["Week"].notna().any():
        wk = pd.to_numeric(METRICS["Week"], errors="coerce").dropna()
        week_min = int(wk.min()) if not wk.empty else 1
        week_max = int(wk.max()) if not wk.empty else 1
    else:
        week_min = week_max = 1

    teams_list = sorted(
        [t for t in METRICS.get("team", pd.Series([], dtype=object)).dropna().unique().tolist()]
    )

    # ---------------------- UI ----------------------
    controls = html.Div(
        [
            html.Div(
                [
                    html.Div("Week range", className="muted"),
                    dcc.RangeSlider(
                        id="pos-weeks",
                        min=week_min,
                        max=week_max,
                        step=1,
                        value=[week_min, week_max],
                        allowCross=False,
                        tooltip={"placement": "bottom"},
                        marks={
                            int(k): str(k)
                            for k in range(
                                week_min,
                                week_max + 1,
                                max(1, (week_max - week_min) // 8 or 1),
                            )
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Div("Team filter (optional)", className="muted"),
                    dcc.Dropdown(
                        id="pos-team",
                        options=[{"label": t, "value": t} for t in teams_list],
                        multi=True,
                        placeholder="All teams",
                        value=[],
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    dcc.RadioItems(
                        id="pos-view-mode",
                        options=[
                            {"label": " Per game", "value": "per_game"},
                            {
                                "label": " Cumulative (selected weeks)",
                                "value": "cumulative",
                            },
                        ],
                        value="per_game",
                        labelStyle={"marginRight": "16px"},
                    ),
                ]
            ),
        ],
        className="controls",
    )

    table_cols_pref = [
        "Week",
        "game_id",
        "team",
        "opponent",
        "possessions",
        "pts_for",
        "pts_against",
        "ORtg",
        "DRtg",
        "NetRtg",
    ]
    table = dash_table.DataTable(
        id="pos-table",
        columns=[
            {"name": c, "id": c}
            for c in [c for c in table_cols_pref if c in METRICS.columns]
        ],
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_cell={"padding": "6px 8px", "fontSize": 13},
        style_as_list_view=True,
        style_header={"fontWeight": "600"},
        style_data_conditional=[{"if": {"column_id": "NetRtg"}, "fontWeight": "600"}],
    )
    chart = dcc.Graph(id="pos-chart", config={"displayModeBar": False})

    core = html.Div(
        [
            html.H2("Possessions & Ratings"),
            html.P(
                "Event-true possessions with ORtg/DRtg/NetRtg per game and cumulatively across week windows."
            ),
            controls,
            html.Hr(),
            html.Div([html.H4("Summary"), table]),
            html.Hr(),
            html.Div([html.H4("ORtg / DRtg"), chart]),
        ],
        className="wrap",
    )

    app.layout = PageShell(core) if PageShell else html.Div(core, className="wrap")

    # ---------------------- Callbacks ----------------------
    @app.callback(
        Output("pos-table", "data"),
        Output("pos-table", "columns"),
        Output("pos-chart", "figure"),
        Input("pos-weeks", "value"),
        Input("pos-team", "value"),
        Input("pos-view-mode", "value"),
    )
    def _update_table(weeks, teams, mode):
        df = METRICS.copy()

        if "Week" in df.columns and weeks and len(weeks) == 2:
            lo, hi = int(weeks[0]), int(weeks[1])
            df["_WeekNum"] = pd.to_numeric(df["Week"], errors="coerce")
            df = df[df["_WeekNum"].between(lo, hi, inclusive="both")].drop(
                columns=["_WeekNum"]
            )

        if teams:
            df = df[df["team"].isin(teams)]

        cols_order_pg = [
            c
            for c in [
                "Week",
                "game_id",
                "team",
                "opponent",
                "possessions",
                "pts_for",
                "pts_against",
                "ORtg",
                "DRtg",
                "NetRtg",
            ]
            if c in df.columns
        ]

        if df.empty:
            cols = [{"name": c, "id": c} for c in cols_order_pg]
            return [], cols, {
                "data": [],
                "layout": {"height": 280, "margin": dict(l=30, r=10, t=20, b=40)},
            }

        if mode == "cumulative":
            grp = ["team"] if "team" in df.columns else []
            out = (
                df.groupby(grp, as_index=False)
                .agg(
                    games=("game_id", "nunique")
                    if "game_id" in df.columns
                    else ("team", "size"),
                    possessions=("possessions", "sum"),
                    pts_for=("pts_for", "sum"),
                    pts_against=("pts_against", "sum"),
                )
            )
            out["ORtg"] = np.where(
                out["possessions"] > 0, 100.0 * out["pts_for"] / out["possessions"], np.nan
            )
            out["DRtg"] = np.where(
                out["possessions"] > 0,
                100.0 * out["pts_against"] / out["possessions"],
                np.nan,
            )
            out["NetRtg"] = out["ORtg"] - out["DRtg"]
            show = out[
                ["team", "games", "possessions", "pts_for", "pts_against", "ORtg", "DRtg", "NetRtg"]
            ].sort_values("NetRtg", ascending=False)

            fig = {
                "data": [
                    {"type": "bar", "x": show["team"], "y": show["ORtg"], "name": "ORtg"},
                    {"type": "bar", "x": show["team"], "y": show["DRtg"], "name": "DRtg"},
                ],
                "layout": {
                    "barmode": "group",
                    "height": 360,
                    "margin": dict(l=50, r=10, t=30, b=120),
                    "xaxis": {"tickangle": -30},
                },
            }
            cols = [{"name": c, "id": c} for c in show.columns]
            return show.to_dict("records"), cols, fig

        sort_keys = [c for c in ["Week", "game_id", "team"] if c in df.columns] or [
            c for c in ["game_id", "team"] if c in df.columns
        ]
        show = df.sort_values(sort_keys, kind="mergesort", na_position="last").reset_index(drop=True)

        cols = [{"name": c, "id": c} for c in cols_order_pg]
        lbl = (
            show["team"].astype(str) + " — " + show["game_id"].astype(str)
            if {"team", "game_id"} <= set(show.columns)
            else pd.Series(range(len(show)))
        )
        fig = {
            "data": [
                {"type": "bar", "x": lbl, "y": show.get("ORtg"), "name": "ORtg"}
                if "ORtg" in show.columns
                else {},
                {"type": "bar", "x": lbl, "y": show.get("DRtg"), "name": "DRtg"}
                if "DRtg" in show.columns
                else {},
            ],
            "layout": {
                "barmode": "group",
                "height": 360,
                "margin": dict(l=50, r=10, t=30, b=120),
                "xaxis": {"tickangle": -60},
            },
        }
        return show.to_dict("records"), cols, fig

    return app
