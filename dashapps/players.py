# dashapps/players.py
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update

# Optional shell wrapper (renders a common header/nav if present)
try:
    from .common import page_shell as PageShell  # type: ignore
except Exception:
    PageShell = None


# =============================================================================
# ---- Canonical column candidates (define ONCE, at the top) ------------------
# =============================================================================
_DATE_CANDIDATES = [
    "game_date", "date", "startDate", "matchDate", "gmtDate", "playedAt", "utcDate",
    "gameDate", "match_date", "GameDate", "eventDate", "tipoff", "Tipoff", "StartDateTime"
]
_WEEK_CANDIDATES = ["week", "Week", "round", "Round", "gameweek", "matchday", "matchWeek", "gw", "GW", "RoundNo"]

_HOME_NAME_CANDS = [
    "home", "home_team", "homeTeam", "homeName", "HomeTeam", "Home",
    "homeTeamName", "home_team_name", "HomeTeamName", "home_name"
]
_AWAY_NAME_CANDS = [
    "away", "away_team", "awayTeam", "awayName", "AwayTeam", "Away",
    "awayTeamName", "away_team_name", "AwayTeamName", "away_name"
]
_HOME_CODE_CANDS = [
    "home_code", "homeTeamCode", "HomeTeamCode", "homeShort", "homeCode",
    "home_team_code", "homeTricode", "homeAbbr", "homeTeamShort"
]
_AWAY_CODE_CANDS = [
    "away_code", "awayTeamCode", "AwayTeamCode", "awayShort", "awayCode",
    "away_team_code", "awayTricode", "awayAbbr", "awayTeamShort"
]

_OPP_NAME_CANDIDATES = [
    "opponent", "opponent_name", "opponentTeam", "opp_name", "opponent_short", "Opp", "OppName",
    "opponentTeamName", "OpponentTeam", "Opponent", "Opp_Name", "oppTeam", "oppTeamName"
]
_OPP_CODE_CANDIDATES = [
    "opponent_code", "opp_code", "opponentTeamCode", "opp",
    "OppCode", "opponentShort", "oppShort", "opponentAbbr", "oppAbbr", "opponentTricode", "oppTricode"
]

# Shared game ID keys (must be defined BEFORE any reuse)
_GAME_ID_KEYS = [
    "game_code", "GameCode", "match_id", "gameId", "GameID", "game_id",
    "MatchId", "matchId", "fixture_id", "fixtureId", "FixtureID"
]

# ---- PBP-specific candidates ------------------------------------------------
_PBP_GAME_ID_KEYS = _GAME_ID_KEYS  # reuse
_PBP_PLAYER_CANDS = [
    "player", "player_name", "name", "scorer", "shooter", "athlete", "athleteName",
    "PlayerName", "ShooterName", "scorerName", "eventPlayer"
]
_PBP_TEAM_CANDS = [
    "team", "team_name", "teamName", "Team", "team_short", "team_code", "teamCode", "teamTricode"
]
_PBP_TEXT_CANDS = ["text", "desc", "description", "play", "playText", "eventText", "details"]
_PBP_EVENT_CANDS = ["ac", "action", "eventType", "event_type", "actionType", "type", "play_type", "code"]
_PBP_RESULT_CANDS = ["result", "resultType", "isMade", "made", "outcome", "shotResult"]
_PBP_ASSIST_CANDS = ["assist", "assist_player", "assistName", "assister", "assist_by", "assistBy", "asst"]
_PBP_FASTBREAK_FLAGS = ["fast_break", "fastbreak", "fb", "isFastBreak", "fastBreak"]
_PBP_ZONE_CANDS = ["shot_zone", "zone", "area", "shotArea", "shotZone", "location", "locZone"]
_PBP_PERIOD_CANDS = ["period", "q", "quarter", "Quarter"]
_PBP_CLOCK_CANDS = ["clock", "time", "timeRemaining", "game_clock", "GameClock"]


# =============================================================================
# ---- Light utils -------------------------------------------------------------
# =============================================================================
def _col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _clean_text(s: object) -> str:
    if s is None:
        return ""
    t = str(s).replace("\u00A0", " ")
    t = unicodedata.normalize("NFKC", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def _normalize_code(s: object) -> str:
    t = _clean_text(s).upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t


def _upper_or_blank(s: object) -> str:
    return _clean_text(s).upper()


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_slug(x: object) -> str:
    return _slugify(_clean_text(x))


def _has_kw(s: str, *kws: str) -> bool:
    if not s:
        return False
    s2 = s.lower()
    return any(k in s2 for k in kws)


# ---- Team alias table (extend as needed) ------------------------------------
TEAM_ALIASES: Dict[str, str] = {
    _slugify("MANISA BASKET"): "GLİNT MANİSA BASKET",
    _slugify("GLİNT MANİSA BASKET"): "GLİNT MANİSA BASKET",
    _slugify("BURSASPOR BASKET"): "BURSASPOR BASKET",
    _slugify("BURSASPOR BSKETBOL"): "BURSASPOR BASKET",
}


def _canonical_team_display(raw_team_name: str, raw_team_code: str) -> str:
    name_clean = _clean_text(raw_team_name)
    code_clean = _normalize_code(raw_team_code)

    if name_clean:
        name_slug = _slugify(name_clean)
        return TEAM_ALIASES.get(name_slug, name_clean)

    if code_clean:
        return code_clean
    return ""


def _team_key(display_name: str) -> str:
    return _slugify(display_name)


def _stable_player_key(row: pd.Series) -> str:
    # 1) Prefer a persistent ID if present
    for cand in ["player_id", "playerId", "personId", "player_code", "id", "externalId"]:
        if cand in row and pd.notna(row[cand]) and str(row[cand]).strip():
            return f"id-{_slugify(str(row[cand]))}"
    # 2) Fallback: NAME-ONLY
    name = _clean_text(row.get("name", ""))
    slug = _slugify(name)
    if slug:
        return f"name-{slug}"
    # 3) Last resort (jersey-only)
    num = _clean_text(row.get("player_no", "")) or "na"
    return f"noonly-{_slugify(num)}"


def _parse_minutes_to_seconds(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    parts = s.split(":")
    try:
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return 0.0
    return 0.0


# =============================================================================
# ---- IO / discovery ----------------------------------------------------------
# =============================================================================
def _find_csv(env_key: str, candidates: Iterable[Path]) -> Optional[Path]:
    env = os.environ.get(env_key)
    if env and Path(env).exists():
        return Path(env)
    for c in candidates:
        if c and c.exists():
            return c
    return None


def _find_players_all() -> Optional[Path]:
    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "out_data" / "_cumulative" / "players_all.csv",   # <- corrected (no double "out_data")
        data_root / "players_all.csv",
        Path("/mnt/data/players_all.csv"),
    ]
    return _find_csv("PLAYERS_ALL", candidates)


def _find_pbp_all() -> Optional[Path]:
    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "_cumulative" / "pbp_all.csv",
        data_root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),
    ]
    return _find_csv("PBP_ALL", candidates)


# =============================================================================
# ---- Loading players ---------------------------------------------------------
# =============================================================================
def _load_players() -> pd.DataFrame:
    p = _find_players_all()
    if not p:
        cols = [
            "team", "team_key", "team_code", "player_no", "name", "position_group", "sMinutes",
            "sPoints", "sAssists", "sReboundsTotal", "sReboundsOffensive", "sReboundsDefensive",
            "sSteals", "sBlocks", "sTurnovers",
            "sFieldGoalsMade", "sFieldGoalsAttempted",
            "sThreePointersMade", "sThreePointersAttempted",
            "sFreeThrowsMade", "sFreeThrowsAttempted",
            "sPlusMinusPoints",
        ]
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(p, low_memory=False)

    # Ensure expected columns exist
    for need in [
        "team_name", "team_code", "player_no", "name", "position_group", "sMinutes",
        "sPoints", "sAssists", "sReboundsTotal", "sReboundsOffensive", "sReboundsDefensive",
        "sSteals", "sBlocks", "sTurnovers",
        "sFieldGoalsMade", "sFieldGoalsAttempted",
        "sThreePointersMade", "sThreePointersAttempted",
        "sFreeThrowsMade", "sFreeThrowsAttempted",
        "sPlusMinusPoints",
    ]:
        if need not in df.columns:
            if need in {"team_name", "team_code", "player_no", "name", "position_group", "sMinutes"}:
                df[need] = ""
            else:
                df[need] = 0

    # Clean text fields
    df["team_name"] = df["team_name"].map(_clean_text)
    df["team_code"] = df["team_code"].map(_normalize_code)
    df["name"] = df["name"].map(_clean_text)
    df["player_no"] = df["player_no"].map(lambda v: re.sub(r"\.0$", "", _clean_text(v)))
    df["position_group"] = df["position_group"].map(_upper_or_blank)

    # Numeric casts
    for c in [
        "sPoints", "sAssists", "sReboundsTotal", "sReboundsOffensive", "sReboundsDefensive",
        "sSteals", "sBlocks", "sTurnovers",
        "sFieldGoalsMade", "sFieldGoalsAttempted",
        "sThreePointersMade", "sThreePointersAttempted",
        "sFreeThrowsMade", "sFreeThrowsAttempted",
        "sPlusMinusPoints",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Minutes -> seconds and GP flag
    df["_seconds"] = df["sMinutes"].apply(_parse_minutes_to_seconds)
    df["_gp_flag"] = (df["_seconds"] > 0).astype(int)

    # Canonical team (display + key)
    df["team_display"] = df.apply(
        lambda r: _canonical_team_display(r.get("team_name", ""), r.get("team_code", "")), axis=1
    )
    df["team_key"] = df["team_display"].map(_team_key)

    # Stable player key
    df["player_key"] = df.apply(_stable_player_key, axis=1)

    # Friendly UI columns
    df["player"] = df["name"]
    df["team"] = df["team_display"]

    # Guard against NaNs in keys used later
    for c in ["team", "team_key", "player", "position_group"]:
        df[c] = df[c].fillna("")

    return df


# =============================================================================
# ---- PBP meta map (1 row per game_id) ---------------------------------------
# =============================================================================
_LABEL_SPLITS = [
    r"\s+vs\.?\s+",
    r"\s+@\s+",
    r"\s*[–—-]\s*",
    r"\s+v\.?\s+",
]


def _split_label_teams(label: object) -> tuple[str, str]:
    s = _clean_text(label)
    if not s:
        return "", ""
    for pat in _LABEL_SPLITS:
        parts = re.split(pat, s, maxsplit=1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return "", ""


def _norm_contains(a: str, b: str) -> bool:
    """Return True if slug a≈b (contains either way)."""
    if not a or not b:
        return False
    return (a == b) or (a in b) or (b in a)


def _load_pbp_meta_map() -> Optional[pd.DataFrame]:
    """
    Build (game_id -> date/week/home-away names & codes) from pbp_all.csv.
    Uses 'label' to derive _home_name/_away_name when explicit columns are absent.
    Returns: columns [join_key,_date,_week,_home_name,_away_name,_home_code,_away_code]
    """
    p = _find_pbp_all()
    if not p:
        return None
    try:
        pbp = pd.read_csv(p, low_memory=False)
    except Exception:
        return None

    pbp_key = _pick_first(pbp, _GAME_ID_KEYS) or "game_id"
    if pbp_key not in pbp.columns:
        return None

    out = pd.DataFrame({"join_key": pbp[pbp_key].astype(str)})

    # Dates & weeks
    date_col = _pick_first(pbp, _DATE_CANDIDATES)
    wk_col = _pick_first(pbp, _WEEK_CANDIDATES)
    out["_date"] = pd.to_datetime(pbp[date_col], errors="coerce") if date_col else pd.NaT
    if wk_col:
        wk = pbp[wk_col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        out["_week"] = pd.to_numeric(wk, errors="coerce")
    else:
        out["_week"] = pd.NA

    # Try explicit home/away first
    hname = _pick_first(pbp, _HOME_NAME_CANDS)
    aname = _pick_first(pbp, _AWAY_NAME_CANDS)
    hcode = _pick_first(pbp, _HOME_CODE_CANDS)
    acode = _pick_first(pbp, _AWAY_CODE_CANDS)

    out["_home_name"] = pbp[hname].map(_clean_text) if hname in pbp.columns else ""
    out["_away_name"] = pbp[aname].map(_clean_text) if aname in pbp.columns else ""
    out["_home_code"] = pbp[hcode].map(_normalize_code) if hcode in pbp.columns else ""
    out["_away_code"] = pbp[acode].map(_normalize_code) if acode in pbp.columns else ""

    # If home/away names are missing, derive from 'label'
    if (out["_home_name"].astype(str).str.strip() == "").any() or (out["_away_name"].astype(str).str.strip() == "").any():
        lbl_col = "label" if "label" in pbp.columns else None
        if lbl_col:
            homes, aways = [], []
            for s in pbp[lbl_col].fillna(""):
                a, b = _split_label_teams(s)
                homes.append(a)
                aways.append(b)
            m_home_blank = out["_home_name"].astype(str).str.strip() == ""
            m_away_blank = out["_away_name"].astype(str).str.strip() == ""
            out.loc[m_home_blank, "_home_name"] = pd.Series(homes, index=out.index)[m_home_blank].map(_clean_text)
            out.loc[m_away_blank, "_away_name"] = pd.Series(aways, index=out.index)[m_away_blank].map(_clean_text)

    # Aggregate to 1 row per game_id
    out = out.dropna(subset=["join_key"])
    out = out.groupby("join_key", as_index=False).agg(
        _date=("_date", "max"),
        _week=("_week", "max"),
        _home_name=("_home_name", "last"),
        _away_name=("_away_name", "last"),
        _home_code=("_home_code", "last"),
        _away_code=("_away_code", "last"),
    )
    if out["_week"].notna().any():
        out["_week"] = pd.to_numeric(out["_week"], errors="coerce").round().astype("Int64")

    return out


# =============================================================================
# ---- Week enrichment for players --------------------------------------------
# =============================================================================
def _enrich_players_with_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a usable Int64 _week column.
    Priority:
      1) Dataset's own week/round column
      2) Join from pbp_all using any common game id key
      3) ISO week from a date column
    """
    df = df.copy()

    # 1) Direct week/round on players_all
    wk_col = _pick_first(df, _WEEK_CANDIDATES)
    if wk_col:
        wk = df[wk_col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        df["_week"] = pd.to_numeric(wk, errors="coerce").round().astype("Int64")
        return df

    # 2) Join from pbp_all via any common id
    wkmap = _load_pbp_meta_map()
    if wkmap is not None and not wkmap.empty:
        pkey = _pick_first(df, _GAME_ID_KEYS)
        if pkey:
            j = df.merge(wkmap[["join_key", "_week"]], left_on=pkey, right_on="join_key", how="left")
            if j.get("_week") is not None and j["_week"].notna().any():
                j["_week"] = pd.to_numeric(j["_week"], errors="coerce").astype("Int64")
                return j.drop(columns=["join_key"])

    # 3) Derive ISO week from a date column
    date_col = _pick_first(df, _DATE_CANDIDATES)
    if date_col:
        dd = pd.to_datetime(df[date_col], errors="coerce")
        df["_week"] = dd.dt.isocalendar().week.astype("Int64")
        return df

    # Last resort: no week available
    df["_week"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df


# Load & enrich (module-level data)
_PLAYERS_RAW = _load_players()
_PLAYERS_RAW = _enrich_players_with_week(_PLAYERS_RAW)


# =============================================================================
# ---- Week helpers & opponent derivation -------------------------------------
# =============================================================================
def _week_bounds_from_data(df: pd.DataFrame) -> Tuple[int, int]:
    if "_week" in df.columns and df["_week"].notna().any():
        lo = int(df["_week"].min(skipna=True))
        hi = int(df["_week"].max(skipna=True))
        if lo == hi:
            return max(1, lo), max(1, hi)
        return lo, hi
    return 1, 34


def _derive_opponent_frame(gl: pd.DataFrame) -> pd.Series:
    opp_col = _pick_first(gl, _OPP_NAME_CANDIDATES) or _pick_first(gl, _OPP_CODE_CANDIDATES)
    if opp_col:
        opp = gl[opp_col].astype(str)
    else:
        opp = pd.Series([""] * len(gl), index=gl.index, dtype=object)

    need = opp.isna() | (opp.astype(str).str.strip() == "")
    if not need.any():
        return opp.fillna("")

    hname = _pick_first(gl, _HOME_NAME_CANDS)
    aname = _pick_first(gl, _AWAY_NAME_CANDS)
    hcode = _pick_first(gl, _HOME_CODE_CANDS)
    acode = _pick_first(gl, _AWAY_CODE_CANDS)

    def _row_team_name(r) -> str:
        for c in ["team", "team_display", "team_name"]:
            if c in r and _clean_text(r[c]):
                return _clean_text(r[c])
        return ""

    def _row_team_code(r) -> str:
        for c in ["team_code"]:
            if c in r and _clean_text(r[c]):
                return _normalize_code(r[c])
        return ""

    def _choose_opp(r):
        me_name = _row_team_name(r)
        me_code = _row_team_code(r)
        me_slug = _norm_slug(me_name)
        me_codeU = _normalize_code(me_code)

        hn = _clean_text(r.get(hname, "")) if hname else ""
        an = _clean_text(r.get(aname, "")) if aname else ""
        hc = _normalize_code(r.get(hcode, "")) if hcode else ""
        ac = _normalize_code(r.get(acode, "")) if acode else ""

        if me_slug and _norm_slug(hn) == me_slug:
            return an or ac
        if me_slug and _norm_slug(an) == me_slug:
            return hn or hc
        if me_codeU and hc and me_codeU == hc:
            return an or ac
        if me_codeU and ac and me_codeU == ac:
            return hn or hc
        return ""

    if hname or aname or hcode or acode:
        derived = gl.apply(_choose_opp, axis=1)
        opp = np.where(need, derived, opp)

    return pd.Series(opp, index=gl.index).fillna("")


# =============================================================================
# ---- Aggregation -------------------------------------------------------------
# =============================================================================
def _aggregate_players(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "uid", "team_key", "team", "team_code", "player_no", "player", "pos",
            "gp", "min_tot", "min_g",
            "pts_tot", "pts_g",
            "reb_tot", "reb_g", "oreb_tot", "oreb_g", "dreb_tot", "dreb_g",
            "ast_tot", "ast_g",
            "stl_tot", "stl_g", "blk_tot", "blk_g", "tov_tot", "tov_g",
            "fgm", "fga", "fg_pct",
            "tpm", "tpa", "tp_pct",
            "ftm", "fta", "ft_pct",
            "pm_tot", "player_link", "team_link",
        ])

    for k in ["team_key", "team", "team_code", "player_no", "player", "position_group"]:
        if k not in df.columns:
            df[k] = ""

    g = df.groupby(["team_key", "player_key"], dropna=False)
    agg = g.agg(
        gp=("_gp_flag", "sum"),
        sec_tot=("_seconds", "sum"),
        pts_tot=("sPoints", "sum"),
        ast_tot=("sAssists", "sum"),
        reb_tot=("sReboundsTotal", "sum"),
        oreb_tot=("sReboundsOffensive", "sum"),
        dreb_tot=("sReboundsDefensive", "sum"),
        stl_tot=("sSteals", "sum"),
        blk_tot=("sBlocks", "sum"),
        tov_tot=("sTurnovers", "sum"),
        fgm=("sFieldGoalsMade", "sum"),
        fga=("sFieldGoalsAttempted", "sum"),
        tpm=("sThreePointersMade", "sum"),
        tpa=("sThreePointersAttempted", "sum"),
        ftm=("sFreeThrowsMade", "sum"),
        fta=("sFreeThrowsAttempted", "sum"),
        pm_tot=("sPlusMinusPoints", "sum"),
        team=("team", "last"),
        team_code=("team_code", "last"),
        player=("player", "last"),
        player_no=("player_no", "last"),
        pos=("position_group", "last"),
    ).reset_index()

    for c in ["stl_tot", "blk_tot", "tov_tot", "oreb_tot", "dreb_tot", "reb_tot", "ast_tot", "pts_tot"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        agg["fg_pct"] = np.where(agg["fga"] > 0, 100.0 * agg["fgm"] / agg["fga"], np.nan)
        agg["tp_pct"] = np.where(agg["tpa"] > 0, 100.0 * agg["tpm"] / agg["tpa"], np.nan)
        agg["ft_pct"] = np.where(agg["fta"] > 0, 100.0 * agg["ftm"] / agg["fta"], np.nan)

    agg["min_tot"] = (agg["sec_tot"] / 60.0).round(1)
    agg["min_g"] = np.where(agg["gp"] > 0, (agg["sec_tot"] / 60.0) / agg["gp"], np.nan).round(1)

    for base, col in [
        ("pts", "pts_tot"),
        ("reb", "reb_tot"),
        ("oreb", "oreb_tot"),
        ("dreb", "dreb_tot"),
        ("ast", "ast_tot"),
        ("stl", "stl_tot"),
        ("blk", "blk_tot"),
        ("tov", "tov_tot"),
    ]:
        agg[f"{base}_g"] = np.where(agg["gp"] > 0, agg[col] / agg["gp"], np.nan).round(1)

    agg["uid"] = agg.apply(lambda r: f"{r['team_key']}|{r['player_key']}", axis=1)
    agg["player_link"] = agg.apply(
        lambda r: f"[{r['player']}]" f"(/player_profile/?player={r['player_key']}&team={r['team_key']})",
        axis=1
    )
    agg["team_link"] = agg.apply(lambda r: f"[{r['team']}](#team-{r['team_key']})", axis=1)

    agg = agg.sort_values(["pts_tot", "min_tot"], ascending=[False, False])

    for pct in ["fg_pct", "tp_pct", "ft_pct"]:
        agg[pct] = agg[pct].round(1)

    cols = [
        "uid", "team_key", "team", "team_code", "player_no", "player", "pos",
        "gp", "min_tot", "min_g",
        "pts_tot", "pts_g",
        "reb_tot", "reb_g", "oreb_tot", "oreb_g", "dreb_tot", "dreb_g",
        "ast_tot", "ast_g",
        "stl_tot", "stl_g", "blk_tot", "blk_g", "tov_tot", "tov_g",
        "fgm", "fga", "fg_pct",
        "tpm", "tpa", "tp_pct",
        "ftm", "fta", "ft_pct",
        "pm_tot", "player_link", "team_link",
    ]
    return agg.loc[:, [c for c in cols if c in agg.columns]]


def _aggregate_for_weeks(week_lo: Optional[int], week_hi: Optional[int]) -> pd.DataFrame:
    base = _PLAYERS_RAW
    if "_week" in base.columns and base["_week"].notna().any():
        if week_lo is not None and week_hi is not None:
            m = base["_week"].between(week_lo, week_hi, inclusive="both")
            base = base.loc[m].copy()
    return _aggregate_players(base)


# Precompute options / bounds
_AGG_ALL = _aggregate_players(_PLAYERS_RAW)
WEEK_MIN, WEEK_MAX = _week_bounds_from_data(_PLAYERS_RAW)


def _options(series: pd.Series) -> List[Dict[str, str]]:
    vals = [v for v in series.dropna().astype(str).unique() if v.strip()]
    vals.sort()
    return [{"label": v, "value": v} for v in vals]


TEAM_OPTIONS = _options(_AGG_ALL.get("team", pd.Series(dtype=str))) if not _AGG_ALL.empty else []
POS_OPTIONS = _options(_AGG_ALL.get("pos", pd.Series(dtype=str))) if not _AGG_ALL.empty else []


# =============================================================================
# ---- Shot classification helpers (for scoring style) -------------------------
# =============================================================================
def _made_flag(row: pd.Series, result_col: Optional[str], text: Optional[str]) -> bool:
    if result_col:
        v = row.get(result_col)
        if isinstance(v, (int, float)):
            return int(v) == 1
        t = str(v).strip().lower()
        if t in {"made", "make", "good", "scored", "scores", "success", "true", "yes", "y"}:
            return True
        if t in {"miss", "missed", "no good", "false", "no", "n"}:
            return False
    return _has_kw(text or "", "makes", "made", "is good", "scores", "converted")


def _is_three(row: pd.Series, ev_col: Optional[str], text: Optional[str]) -> bool:
    if ev_col:
        t = str(row.get(ev_col, "")).upper()
        if t in {"P3", "3PT", "THREE", "3P", "3-PT"}:
            return True
    return _has_kw(text or "", "3pt", "three", "3-pointer", "triple", "3-point")


def _assist_flag(row: pd.Series, text: Optional[str]) -> bool:
    for c in _PBP_ASSIST_CANDS:
        if c in row and str(row[c]).strip():
            return True
    return _has_kw(text or "", "assist")


def _fastbreak_flag(row: pd.Series, text: Optional[str]) -> bool:
    for c in _PBP_FASTBREAK_FLAGS:
        if c in row:
            v = str(row[c]).strip().lower()
            if v in {"1", "true", "yes", "y"}:
                return True
            if v in {"0", "false", "no", "n"}:
                return False
    return _has_kw(text or "", "fast break", "fastbreak", "fast-break")


def _second_chance_flag(text: Optional[str]) -> bool:
    return _has_kw(text or "", "second chance", "putback", "tip-in", "tip in", "follow up", "follow-up", "tap-in")


def _oreb_text_flag(text: Optional[str]) -> bool:
    return _has_kw(text or "", "offensive rebound", "oreb", "off. reb")


def _paint_flag(row: pd.Series, text: Optional[str], zone: Optional[str]) -> bool:
    if zone and str(row.get(zone, "")).strip():
        if _has_kw(str(row[zone]), "paint", "restricted", "key", "at rim", "close"):
            return True
    return _has_kw(text or "", "layup", "dunk", "hook", "tip-in", "putback", "floater", "runner", "at rim", "in the paint")


def _classify_style(text: Optional[str], is_three: bool) -> Tuple[str, str]:
    """
    Returns (category, subtype).
    Categories: Layup, Dunk, Hook, Tip-in, Jump Shot, Other
    """
    s = (text or "").lower()

    # Dunks
    if "dunk" in s or "alley" in s:
        return ("Dunk", "Standard" if "alley" not in s else "Alley-oop")

    # Tip-ins / putbacks
    if "tip-in" in s or "tip in" in s or "putback" in s or "tap-in" in s:
        return ("Tip-in", "Putback")

    # Hooks
    if "hook" in s:
        return ("Hook", "Jump Hook" if "jump" in s else "Hook")

    # Layups
    if "layup" in s or "finger roll" in s or "eurostep" in s or "euro step" in s:
        if "reverse" in s:
            sub = "Reverse"
        elif "finger roll" in s:
            sub = "Finger Roll"
        elif "driv" in s:
            sub = "Driving"
        elif "euro" in s:
            sub = "Eurostep"
        elif "putback" in s:
            sub = "Putback"
        else:
            sub = "Other"
        return ("Layup", sub)

    # Jumpers
    if "jumper" in s or "jump shot" in s or "jumpshot" in s or "fadeaway" in s or "turnaround" in s \
       or "pull-up" in s or "pullup" in s or "step-back" in s or "stepback" in s or "bank" in s or "mid" in s:
        if is_three:
            if "pull" in s:
                sub = "3PT (Pull-up)"
            elif "catch" in s:
                sub = "3PT (C&S)"
            else:
                sub = "3PT (Other)"
        else:
            if "pull" in s:
                sub = "Pull-up"
            elif "step" in s:
                sub = "Step-back"
            elif "fade" in s:
                sub = "Fadeaway"
            elif "turn" in s:
                sub = "Turnaround"
            elif "bank" in s:
                sub = "Bank"
            elif "mid" in s:
                sub = "Mid-Range"
            else:
                sub = "Jumper"
        return ("Jump Shot", sub)

    # 3PT (text only)
    if is_three:
        return ("Jump Shot", "3PT (Other)")

    return ("Other", "Other")


def _prev_oreb_window_flags(pbp: pd.DataFrame, game_col: Optional[str], team_col: Optional[str]) -> pd.Series:
    """
    Mark a shot as second-chance if within the previous 3 rows
    for the same game+team there was an 'offensive rebound' line.
    """
    if pbp.empty or not game_col or not team_col:
        return pd.Series([False] * len(pbp), index=pbp.index)
    flags = []
    pbp_reset = pbp.reset_index()
    for i, r in pbp_reset.iterrows():
        idx = r["index"]
        g = pbp.loc[idx, game_col] if game_col in pbp.columns else None
        t = pbp.loc[idx, team_col] if team_col in pbp.columns else None
        lo = max(0, i - 3)
        oreb = False
        for j in range(lo, i):
            jj = pbp_reset.loc[j, "index"]
            if (game_col in pbp.columns and pbp.loc[jj, game_col] == g) and (team_col in pbp.columns and pbp.loc[jj, team_col] == t):
                tx_col = _col(pbp, _PBP_TEXT_CANDS)
                tx = str(pbp.loc[jj, tx_col]) if tx_col else ""
                if _oreb_text_flag(tx):
                    oreb = True
                    break
        flags.append(oreb)
    return pd.Series(flags, index=pbp.index)


# =============================================================================
# ---- Game log ---------------------------------------------------------------
# =============================================================================
def _build_game_log(uid: str, limit: int = 12) -> pd.DataFrame:
    """
    Player game log with Opponent/H-A resolution.
    Adds PF (personal fouls), FD (fouls drawn), and 2P (made-attempted).
    Prevents any dependency on 'opponent' in CSVs by deriving from PBP/label.
    """
    try:
        team_key, player_key = uid.split("|", 1)
    except Exception:
        return pd.DataFrame()

    gl = _PLAYERS_RAW.copy()

    # Ensure keys exist
    if "team_key" not in gl.columns:
        gl["team_display"] = gl.apply(
            lambda r: _canonical_team_display(r.get("team_name", ""), r.get("team_code", "")), axis=1
        )
        gl["team_key"] = gl["team_display"].map(_team_key)
    if "player_key" not in gl.columns:
        gl["player_key"] = gl.apply(_stable_player_key, axis=1)

    # Filter player/team
    gl = gl.loc[(gl["team_key"] == team_key) & (gl["player_key"] == player_key)].copy()
    if gl.empty:
        return gl

    # --- PBP meta join
    pkey = _pick_first(gl, _GAME_ID_KEYS)
    meta = _load_pbp_meta_map()
    if pkey and meta is not None and not meta.empty and pkey in gl.columns:
        gl[pkey] = gl[pkey].astype(str)
        meta = meta.rename(columns={"_week": "_week_meta", "_date": "_date_meta"})
        gl = gl.merge(meta, left_on=pkey, right_on="join_key", how="left")

    # --- Date
    date_col = _pick_first(gl, _DATE_CANDIDATES)
    if date_col:
        gl["_date"] = pd.to_datetime(gl[date_col], errors="coerce")
    elif "_date_meta" in gl.columns:
        gl["_date"] = gl["_date_meta"]
    else:
        gl["_date"] = pd.NaT

    # --- Opponent/H-A from label first (then meta; then fallback maps)
    gl["_opp"] = ""
    gl["H/A"] = ""
    if "label" in gl.columns:
        def _from_label_row(r):
            me = _clean_text(r.get("team") or r.get("team_display") or r.get("team_name", ""))
            me_code = _normalize_code(r.get("team_code", ""))
            a, b = _split_label_teams(r.get("label", ""))
            if not a and not b:
                return "", ""
            me_slug = _norm_slug(me) or _norm_slug(me_code)
            a_slug, b_slug = _norm_slug(a), _norm_slug(b)
            if _norm_contains(me_slug, a_slug):
                return (_clean_text(b) if not _norm_contains(me_slug, b_slug) else _clean_text(a)), "H"
            if _norm_contains(me_slug, b_slug):
                return (_clean_text(a) if not _norm_contains(me_slug, a_slug) else _clean_text(b)), "A"
            return "", ""
        tmp = gl.apply(_from_label_row, axis=1, result_type="expand")
        if isinstance(tmp, pd.DataFrame):
            gl["_opp"] = tmp[0].fillna("")
            gl["H/A"] = tmp[1].fillna("")

    # Native opponent columns if present
    opp_direct = _derive_opponent_frame(gl)
    need = gl["_opp"].astype(str).str.strip().eq("")
    gl.loc[need, "_opp"] = opp_direct[need].fillna("")

    # Use meta for missing opp/venue
    def _opp_from_meta_and_venue(r):
        current_opp = _clean_text(r.get("_opp", ""))
        current_ven = _clean_text(r.get("H/A", ""))
        me = _clean_text(r.get("team") or r.get("team_display") or r.get("team_name", ""))
        me_code = _normalize_code(r.get("team_code", ""))
        me_slug = _norm_slug(me)
        hn = _clean_text(r.get("_home_name", "")); an = _clean_text(r.get("_away_name", ""))
        hc = _normalize_code(r.get("_home_code", "")); ac = _normalize_code(r.get("_away_code", ""))
        if current_opp and not current_ven:
            if (me_slug and _norm_slug(hn) == me_slug) or (me_code and me_code == hc):
                return current_opp, "H"
            if (me_slug and _norm_slug(an) == me_slug) or (me_code and me_code == ac):
                return current_opp, "A"
            return current_opp, current_ven
        if (me_slug and _norm_slug(hn) == me_slug) or (me_code and me_code == hc):
            return (an or ac), "H"
        if (me_slug and _norm_slug(an) == me_slug) or (me_code and me_code == ac):
            return (hn or hc), "A"
        return current_opp, current_ven

    if {"_home_name", "_away_name", "_home_code", "_away_code"}.issubset(gl.columns):
        tmp = gl.apply(_opp_from_meta_and_venue, axis=1, result_type="expand")
        need_opp = gl["_opp"].astype(str).str.strip().eq("")
        need_ven = gl["H/A"].astype(str).str.strip().eq("")
        gl.loc[need_opp, "_opp"] = tmp[0][need_opp].fillna("")
        gl.loc[need_ven, "H/A"] = tmp[1][need_ven].fillna("")

    # Canonicalize opponent label
    def _canon_opp(lbl: str) -> str:
        name = _clean_text(lbl); slug = _slugify(name)
        return TEAM_ALIASES.get(slug, name)
    gl["_opp"] = gl["_opp"].map(_canon_opp)

    # Week
    wk_col = _pick_first(gl, _WEEK_CANDIDATES)
    if wk_col:
        gl["_week"] = pd.to_numeric(gl[wk_col].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")
    elif "_week" in gl.columns and str(gl["_week"].dtype) != "Int64":
        gl["_week"] = pd.to_numeric(gl["_week"], errors="coerce").astype("Int64")
    elif "_week_meta" in gl.columns:
        gl["_week"] = gl["_week_meta"]
    else:
        gl["_week"] = pd.Series([pd.NA] * len(gl), dtype="Int64")

    # Numeric casts (+ fouls candidates incl. your 'sFousOn')
    numeric_cols = [
        "sPoints", "sAssists", "sReboundsTotal", "sReboundsOffensive", "sReboundsDefensive",
        "sSteals", "sBlocks", "sTurnovers",
        "sFieldGoalsMade", "sFieldGoalsAttempted",
        "sThreePointersMade", "sThreePointersAttempted",
        "sFreeThrowsMade", "sFreeThrowsAttempted",
        "sPlusMinusPoints",
        "sFoulsPersonal", "foulsPersonal", "personalFouls", "PF", "sFouls",
        "sFoulsDrawn", "foulsDrawn", "foulsReceived", "foulsOn", "sFoulsOn", "sFousOn", "FD",
    ]
    for c in numeric_cols:
        if c in gl.columns:
            gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0)

    # Choose canonical PF / FD
    pf_col = next((c for c in ["sFoulsPersonal", "foulsPersonal", "personalFouls", "PF", "sFouls"] if c in gl.columns), None)
    fd_col = next((c for c in ["sFoulsOn", "sFousOn", "sFoulsDrawn", "foulsDrawn", "foulsReceived", "foulsOn", "FD"] if c in gl.columns), None)

    # Minutes → MIN
    if "_seconds" not in gl.columns and "sMinutes" in gl.columns:
        gl["_seconds"] = gl["sMinutes"].apply(_parse_minutes_to_seconds)
    gl["MIN"] = (gl.get("_seconds", 0) / 60.0).round(1)

    # Shooting displays (and 2P)
    FGM = gl.get("sFieldGoalsMade", 0).astype(int)
    FGA = gl.get("sFieldGoalsAttempted", 0).astype(int)
    TPM = gl.get("sThreePointersMade", 0).astype(int)
    TPA = gl.get("sThreePointersAttempted", 0).astype(int)
    TWO_M = (FGM - TPM).clip(lower=0)
    TWO_A = (FGA - TPA).clip(lower=0)

    gl["FG"] = FGM.astype(str) + "-" + FGA.astype(str)
    gl["2P"] = TWO_M.astype(str) + "-" + TWO_A.astype(str)
    gl["3P"] = TPM.astype(str) + "-" + TPA.astype(str)
    gl["FT"] = gl.get("sFreeThrowsMade", 0).astype(int).astype(str) + "-" + gl.get("sFreeThrowsAttempted", 0).astype(int).astype(str)

    # Final shape
    cols = [
        "_date", "_week", "_opp", "H/A", "MIN",
        "sPoints", "sReboundsTotal", "sReboundsOffensive", "sReboundsDefensive",
        "sAssists", "sSteals", "sBlocks", "sTurnovers",
        "FG", "2P", "3P", "FT", "sPlusMinusPoints",
    ]
    if pf_col: cols.insert(cols.index("sAssists") + 1, pf_col)
    if fd_col: cols.insert(cols.index("sAssists") + 1, fd_col)

    cols = [c for c in cols if c in gl.columns]
    gl = gl.loc[:, cols].copy()

    rename = {
        "_date": "Date", "_week": "Week", "_opp": "Opponent",
        "sPoints": "PTS", "sReboundsTotal": "REB", "sReboundsOffensive": "OREB", "sReboundsDefensive": "DREB",
        "sAssists": "AST", "sSteals": "STL", "sBlocks": "BLK", "sTurnovers": "TOV", "sPlusMinusPoints": "+/-",
    }
    if pf_col: rename[pf_col] = "PF"
    if fd_col: rename[fd_col] = "FD"
    gl = gl.rename(columns=rename)

    if "Date" in gl.columns:
        gl["Date"] = pd.to_datetime(gl["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        gl = gl.sort_values("Date", ascending=False, na_position="last")
    else:
        gl = gl.iloc[::-1]

    if "Week" in gl.columns and gl["Week"].notna().any():
        gl["Week"] = gl["Week"].astype("Int64")

    return gl.head(limit)


# =============================================================================
# ---- Scoring style table -----------------------------------------------------
# =============================================================================
def _build_scoring_style(uid: str, include_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Returns a table with Category/Subtype, FGM, 3PM, PTS, Share%
    Scope is restricted to games whose Date (YYYY-MM-DD) is in include_dates (if provided).
    """
    try:
        team_key, player_key = uid.split("|", 1)
    except Exception:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])

    # 1) recover player's season rows and join to PBP meta to get join_key + date
    rows = _PLAYERS_RAW.loc[(_PLAYERS_RAW["team_key"] == team_key) & (_PLAYERS_RAW["player_key"] == player_key)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])

    pkey = _pick_first(rows, _PBP_GAME_ID_KEYS)
    meta = _load_pbp_meta_map()
    if meta is None or meta.empty or (not pkey) or (pkey not in rows.columns):
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])

    rows[pkey] = rows[pkey].astype(str)
    meta = meta.rename(columns={"_date": "_date_meta"})
    jj = rows.merge(meta[["join_key", "_date_meta"]], left_on=pkey, right_on="join_key", how="left")
    if include_dates:
        ds = set(pd.to_datetime(pd.Series(include_dates), errors="coerce").dt.strftime("%Y-%m-%d"))
        jj = jj[pd.to_datetime(jj["_date_meta"], errors="coerce").dt.strftime("%Y-%m-%d").isin(ds)]

    game_ids = set(jj["join_key"].dropna().astype(str).tolist())
    if not game_ids:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])

    # 2) read pbp and filter to those games
    p = _find_pbp_all()
    if not p:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])
    pbp = pd.read_csv(p, low_memory=False)

    game_col = _col(pbp, _PBP_GAME_ID_KEYS)
    if not game_col:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])
    pbp[game_col] = pbp[game_col].astype(str)
    pbp = pbp[pbp[game_col].isin(game_ids)].copy()
    if pbp.empty:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])

    text_col = _col(pbp, _PBP_TEXT_CANDS)
    ev_col = _col(pbp, _PBP_EVENT_CANDS)
    res_col = _col(pbp, _PBP_RESULT_CANDS)
    zone_col = _col(pbp, _PBP_ZONE_CANDS)
    shooter_col = _col(pbp, _PBP_PLAYER_CANDS)
    team_col = _col(pbp, _PBP_TEAM_CANDS)

    # 3) filter to shots MADE by this player
    my_name = _clean_text(rows.get("name", pd.Series([""])).iloc[-1])
    my_slug = _norm_slug(my_name)

    def _is_me(row) -> bool:
        if shooter_col and str(row.get(shooter_col, "")).strip():
            return _norm_slug(row[shooter_col]) == my_slug
        tx = str(row.get(text_col, ""))
        return _has_kw(tx, my_name.lower())

    sc_from_window = _prev_oreb_window_flags(pbp, game_col, team_col) if (text_col and team_col) else pd.Series([False] * len(pbp), index=pbp.index)

    shots = []
    for idx, r in pbp.iterrows():
        tx = str(r.get(text_col, ""))
        if not _is_me(r):
            continue
        if not _made_flag(r, res_col, tx):
            continue
        is3 = _is_three(r, ev_col, tx)
        cat, sub = _classify_style(tx, is3)
        assisted = _assist_flag(r, tx)
        fastbr = _fastbreak_flag(r, tx)
        secondc = _second_chance_flag(tx) or bool(sc_from_window.get(idx, False))
        paint = _paint_flag(r, tx, zone_col)

        pts = 3 if is3 else 2
        fgm = 1
        tpm = 1 if is3 else 0

        shots.append({
            "cat": cat, "sub": sub, "is3": is3, "assisted": assisted,
            "fastbreak": fastbr, "second": secondc, "paint": paint,
            "FGM": fgm, "3PM": tpm, "PTS": pts
        })

    if not shots:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])

    sh = pd.DataFrame(shots)

    def bucket_sum(mask) -> float:
        return float(sh.loc[mask, "PTS"].sum())

    total_pts = float(sh["PTS"].sum())
    if total_pts <= 0:
        total_pts = 1.0

    rows_out: List[Dict[str, Any]] = []
    rows_out.append({"Category": "Fast Break", "Subtype": "—", "FGM": int(sh["FGM"][sh["fastbreak"]].sum()), "3PM": int(sh["3PM"][sh["fastbreak"]].sum()), "PTS": bucket_sum(sh["fastbreak"]), "Share%": round(100.0 * bucket_sum(sh["fastbreak"]) / total_pts, 2)})
    rows_out.append({"Category": "Second Chance", "Subtype": "—", "FGM": int(sh["FGM"][sh["second"]].sum()), "3PM": int(sh["3PM"][sh["second"]].sum()), "PTS": bucket_sum(sh["second"]), "Share%": round(100.0 * bucket_sum(sh["second"]) / total_pts, 2)})
    rows_out.append({"Category": "Points in Paint", "Subtype": "—", "FGM": int(sh["FGM"][sh["paint"]].sum()), "3PM": int(sh["3PM"][sh["paint"]].sum()), "PTS": bucket_sum(sh["paint"]), "Share%": round(100.0 * bucket_sum(sh["paint"]) / total_pts, 2)})
    rows_out.append({"Category": "Assisted", "Subtype": "—", "FGM": int(sh["FGM"][sh["assisted"]].sum()), "3PM": int(sh["3PM"][sh["assisted"]].sum()), "PTS": bucket_sum(sh["assisted"]), "Share%": round(100.0 * bucket_sum(sh["assisted"]) / total_pts, 2)})
    rows_out.append({"Category": "Unassisted", "Subtype": "—", "FGM": int(sh["FGM"][~sh["assisted"]].sum()), "3PM": int(sh["3PM"][~sh["assisted"]].sum()), "PTS": bucket_sum(~sh["assisted"]), "Share%": round(100.0 * bucket_sum(~sh["assisted"]) / total_pts, 2)})

    # Layup subtypes
    lay = sh[sh["cat"] == "Layup"]
    if not lay.empty:
        for sub, grp in lay.groupby("sub", dropna=False):
            pts = float(grp["PTS"].sum())
            rows_out.append({"Category": "Layup", "Subtype": sub, "FGM": int(grp["FGM"].sum()), "3PM": int(grp["3PM"].sum()),
                             "PTS": pts, "Share%": round(100.0 * pts / total_pts, 2)})

    # Jump shot subtypes
    jmp = sh[sh["cat"] == "Jump Shot"]
    if not jmp.empty:
        for sub, grp in jmp.groupby("sub", dropna=False):
            pts = float(grp["PTS"].sum())
            rows_out.append({"Category": "Jump Shot", "Subtype": sub, "FGM": int(grp["FGM"].sum()), "3PM": int(grp["3PM"].sum()),
                             "PTS": pts, "Share%": round(100.0 * pts / total_pts, 2)})

    # Other residuals
    other = sh[~sh["cat"].isin(["Layup", "Jump Shot"])]
    if not other.empty:
        for sub, grp in other.groupby("cat", dropna=False):
            pts = float(grp["PTS"].sum())
            rows_out.append({"Category": sub, "Subtype": "—", "FGM": int(grp["FGM"].sum()), "3PM": int(grp["3PM"].sum()),
                             "PTS": pts, "Share%": round(100.0 * pts / total_pts, 2)})

    out = pd.DataFrame(rows_out)
    if out.empty:
        return pd.DataFrame(columns=["Category", "Subtype", "FGM", "3PM", "PTS", "Share%"])
    out = out.sort_values(["Category", "PTS"], ascending=[True, False])
    out["PTS"] = out["PTS"].round(1)
    return out


# =============================================================================
# ---- Dash app factory --------------------------------------------------------
# =============================================================================
def create_dash_players(server, base_pathname: str = "/players/") -> Dash:
    base = "/" + str(base_pathname or "").strip("/")
    if not base.endswith("/"):
        base = base + "/"

    app = dash.Dash(
        __name__,
        server=server,
        routes_pathname_prefix=base,
        requests_pathname_prefix=base,
        assets_url_path=base + "assets",
        suppress_callback_exceptions=True,
        serve_locally=True,
        title="Players"
    )

    # ---------- Controls ----------
    control_bar = html.Div([
        html.Div([
            html.Label("Team", className="control-label"),
            dcc.Dropdown(
                id="pl-team", options=TEAM_OPTIONS, multi=True, placeholder="All teams",
                persistence=True, persistence_type="memory", className="control-input"
            )
        ], className="control"),

        html.Div([
            html.Label("Position", className="control-label"),
            dcc.Dropdown(
                id="pl-pos", options=POS_OPTIONS, multi=True, placeholder="All positions",
                persistence=True, persistence_type="memory", className="control-input"
            )
        ], className="control"),

        html.Div([
            html.Label("Search player", className="control-label"),
            dcc.Input(
                id="pl-search", type="text", placeholder="Type a player name…", debounce=True,
                persistence=True, persistence_type="memory", className="control-input"
            )
        ], className="control"),

        html.Div([
            html.Label("View", className="control-label"),
            dcc.RadioItems(
                id="pl-view",
                options=[
                    {"label": "Totals", "value": "totals"},
                    {"label": "Per Game", "value": "per_game"},
                    {"label": "Both", "value": "both"}
                ],
                value="both", inline=True,
                persistence=True, persistence_type="memory", className="control-input"
            )
        ], className="control"),

        html.Div([
            html.Label("Min GP / Min (total)", className="control-label"),
            html.Div([
                dcc.Input(id="pl-min-gp", type="number", min=0, step=1, value=0, style={"width": "90px"}),
                dcc.Input(id="pl-min-min", type="number", min=0, step=1, value=0, style={"width": "110px", "marginLeft": "8px"}),
            ], style={"display": "flex", "alignItems": "center"})
        ], className="control"),

        html.Div([
            html.Label("Weeks", className="control-label"),
            dcc.RangeSlider(
                id="pl-week-range",
                min=WEEK_MIN, max=WEEK_MAX, step=1,
                value=[WEEK_MIN, WEEK_MAX],
                marks=_week_marks(WEEK_MIN, WEEK_MAX),
                allowCross=False,
                tooltip={"placement": "bottom", "always_visible": False},
                persistence=True, persistence_type="memory",
            )
        ], className="control control-wide"),

        html.Div([
            html.Label("Density", className="control-label"),
            dcc.Checklist(
                id="pl-compact",
                options=[{"label": "Compact mode", "value": "compact"}],
                value=[],
                persistence=True, persistence_type="memory",
                inputStyle={"marginRight": "6px"},
            )
        ], className="control"),

        html.Div([
            html.Button("Reset filters", id="pl-reset", className="button", n_clicks=0, title="Clear all filters"),
            html.Button("Download CSV", id="pl-download-btn", n_clicks=0, className="button", title="Export current table"),
            dcc.Download(id="pl-download")
        ], className="control control-actions"),
    ], className="controls", style={
        "display": "grid",
        "gridTemplateColumns": "repeat(6, minmax(180px, 1fr))",
        "gap": "10px",
        "alignItems": "end",
    })

    # ---------- Table ----------
    columns = [
        {"name": "", "id": "expand"},
        {"name": "Player", "id": "player_link", "presentation": "markdown"},
        {"name": "Team", "id": "team_link", "presentation": "markdown"},
        {"name": "Pos", "id": "pos"},
        {"name": "GP", "id": "gp", "type": "numeric"},
        {"name": "MIN", "id": "min_tot", "type": "numeric"},
        {"name": "MIN/G", "id": "min_g", "type": "numeric"},
        {"name": "PTS", "id": "pts_tot", "type": "numeric"},
        {"name": "PTS/G", "id": "pts_g", "type": "numeric"},

        {"name": "REB", "id": "reb_tot", "type": "numeric"},
        {"name": "REB/G", "id": "reb_g", "type": "numeric"},
        {"name": "OREB", "id": "oreb_tot", "type": "numeric"},
        {"name": "OREB/G", "id": "oreb_g", "type": "numeric"},
        {"name": "DREB", "id": "dreb_tot", "type": "numeric"},
        {"name": "DREB/G", "id": "dreb_g", "type": "numeric"},

        {"name": "AST", "id": "ast_tot", "type": "numeric"},
        {"name": "AST/G", "id": "ast_g", "type": "numeric"},
        {"name": "STL", "id": "stl_tot", "type": "numeric"},
        {"name": "STL/G", "id": "stl_g", "type": "numeric"},
        {"name": "BLK", "id": "blk_tot", "type": "numeric"},
        {"name": "BLK/G", "id": "blk_g", "type": "numeric"},
        {"name": "TOV", "id": "tov_tot", "type": "numeric"},
        {"name": "TOV/G", "id": "tov_g", "type": "numeric"},

        {"name": "FGM", "id": "fgm", "type": "numeric"},
        {"name": "FGA", "id": "fga", "type": "numeric"},
        {"name": "FG%", "id": "fg_pct", "type": "numeric"},
        {"name": "3PM", "id": "tpm", "type": "numeric"},
        {"name": "3PA", "id": "tpa", "type": "numeric"},
        {"name": "3P%", "id": "tp_pct", "type": "numeric"},
        {"name": "FTM", "id": "ftm", "type": "numeric"},
        {"name": "FTA", "id": "fta", "type": "numeric"},
        {"name": "FT%", "id": "ft_pct", "type": "numeric"},
        {"name": "+/-", "id": "pm_tot", "type": "numeric"},
    ]

    tooltip_header = {
        "fg_pct": "Field goal % (100 * FGM/FGA)",
        "tp_pct": "3P % (100 * 3PM/3PA)",
        "ft_pct": "Free throw % (100 * FTM/FTA)",
        "min_g": "Average minutes per game",
        "pts_g": "Average points per game",
    }

    table = dash_table.DataTable(
        id="pl-table",
        columns=columns,
        page_size=25,
        sort_action="native",
        filter_action="native",
        tooltip_header=tooltip_header,
        style_table={"overflowX": "auto", "minWidth": "100%"},
        style_header={"fontWeight": "700", "backgroundColor": "#f7f7fb"},
        style_cell={
            "padding": "6px",
            "minWidth": 60, "whiteSpace": "nowrap",
            "fontSize": "14px", "lineHeight": "18px",
        },
        fixed_rows={"headers": True},
        style_cell_conditional=[
            {"if": {"column_id": "expand"},
             "position": "sticky", "left": 0, "zIndex": 1, "backgroundColor": "white",
             "textAlign": "center", "width": 34, "minWidth": 34, "maxWidth": 34, "padding": "0 4px"},
            {"if": {"column_id": "player_link"}, "position": "sticky", "left": 34, "zIndex": 1, "backgroundColor": "white"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafbff"},
            {"if": {"column_id": "fg_pct"}, "backgroundColor": "#f9fffb"},
            {"if": {"column_id": "tp_pct"}, "backgroundColor": "#f9fcff"},
            {"if": {"column_id": "ft_pct"}, "backgroundColor": "#fffaf9"},
        ],
        markdown_options={"html": True},
        virtualization=True,
        persistence=True,
        persisted_props=["sort_by", "filter_query", "page_current", "page_size"],
        merge_duplicate_headers=True,
    )

    # KPI + Expansions + Root wrapper for compact mode CSS
    kpis = html.Div(id="pl-kpis")
    expansions = html.Div(id="pl-expansions")
    store = dcc.Store(id="pl-expanded", data={})
    store_data_cache = dcc.Store(id="pl-last-dataset", data=None)

    root = html.Div(
        id="pl-root",
        **{"data-compact": "0"},
        children=[
            html.Div([
                html.H2("Players"),
                html.Details([
                    html.Summary("What am I looking at?"),
                    html.P("Aggregated player stats with week filtering. Use the slider to slice the season by round/week.")
                ], style={"marginTop": "8px"})
            ]),
            html.Div([
                control_bar,
                html.Hr(),
                kpis,
                dcc.Loading(table, type="dot"),
                html.Div(style={"height": "8px"}),
                dcc.Loading(expansions, type="dot"),
                store,
                store_data_cache,
            ], className="page-content", style={"padding": "12px 0"})
        ]
    )

    if PageShell:
        app.layout = PageShell(root, nav_items=server.config.get("NAV", []), current_endpoint=base)
    else:
        app.layout = root

    # ---------- Main data callback ----------
    @app.callback(
        Output("pl-table", "data"),
        Output("pl-kpis", "children"),
        Output("pl-last-dataset", "data"),
        Input("pl-team", "value"),
        Input("pl-pos", "value"),
        Input("pl-search", "value"),
        Input("pl-view", "value"),
        Input("pl-min-gp", "value"),
        Input("pl-min-min", "value"),
        Input("pl-expanded", "data"),
        Input("pl-week-range", "value"),
        Input("pl-compact", "value"),
    )
    def _update_table(team_vals, pos_vals, q, view_mode, min_gp, min_min, expanded_map, week_range, compact_vals):
        # Aggregate on the fly respecting the selected week window
        if week_range and len(week_range) == 2:
            w0, w1 = int(week_range[0]), int(week_range[1])
        else:
            w0, w1 = WEEK_MIN, WEEK_MAX

        df = _aggregate_for_weeks(w0, w1)

        # filters
        if "team" not in df.columns and "team_key" in df.columns:
            df["team"] = df["team_key"].str.replace("-", " ", regex=False).str.title()

        if team_vals:
            df = df[df["team"].isin(team_vals)]
        if pos_vals and "pos" in df.columns:
            df = df[df["pos"].isin(pos_vals)]
        if q:
            ql = str(q).strip().lower()
            if ql and "player" in df.columns:
                df = df[df["player"].str.lower().str.contains(ql, na=False)]
        if min_gp and "gp" in df.columns:
            df = df[df["gp"] >= int(min_gp)]
        if min_min and "min_tot" in df.columns:
            df = df[df["min_tot"] >= float(min_min)]

        base_cols = ["uid", "team_key", "team", "team_code", "player_no", "player_link", "team_link", "pos", "gp"]

        totals = [
            "min_tot",
            "pts_tot",
            "reb_tot", "oreb_tot", "dreb_tot",
            "ast_tot", "stl_tot", "blk_tot", "tov_tot",
            "fgm", "fga", "fg_pct",
            "tpm", "tpa", "tp_pct",
            "ftm", "fta", "ft_pct",
            "pm_tot",
        ]
        per_g = ["min_g", "pts_g", "reb_g", "oreb_g", "dreb_g", "ast_g", "stl_g", "blk_g", "tov_g"]

        if view_mode == "totals":
            cols = base_cols + totals
        elif view_mode == "per_game":
            cols = base_cols + per_g + ["fg_pct", "tp_pct", "ft_pct"]
        else:
            cols = base_cols + ["min_tot", "min_g",
                                "pts_tot", "pts_g",
                                "reb_tot", "reb_g", "oreb_tot", "oreb_g", "dreb_tot", "dreb_g",
                                "ast_tot", "ast_g",
                                "stl_tot", "stl_g", "blk_tot", "blk_g", "tov_tot", "tov_g",
                                "fgm", "fga", "fg_pct", "tpm", "tpa", "tp_pct", "ftm", "fta", "ft_pct", "pm_tot"]

        out = df.loc[:, [c for c in cols if c in df.columns]].copy()

        expanded_map = expanded_map or {}
        out["expand"] = out["uid"].map(lambda u: "▼" if expanded_map.get(u) else "▶")

        # KPIs
        if out.empty:
            kpi = html.Div("No players match the current filters.", style={"color": "#a00"})
        else:
            tot_gp = int(df["gp"].sum()) if "gp" in df.columns else 0
            tot_min = float(df["min_tot"].sum()) if "min_tot" in df.columns else 0.0
            tot_pts = int(df["pts_tot"].sum()) if "pts_tot" in df.columns else 0
            kpi = html.Div([
                html.Div([
                    _kpi("Players", f"{len(df):,}"),
                    _kpi("Teams", f"{df['team_key'].nunique():,}" if "team_key" in df.columns else "—"),
                    _kpi("Weeks", f"{w0}–{w1}"),
                    _kpi("Total GP (rows)", f"{tot_gp:,}"),
                    _kpi("Total MIN", f"{tot_min:.1f}"),
                    _kpi("Total PTS", f"{tot_pts:,}"),
                ], className="kpi-grid"),
            ])

        # Cache for download (convert markdown to plain)
        download_df = out.copy()
        if "player_link" in download_df.columns:
            download_df["player"] = download_df["player_link"].str.extract(r"\[([^\]]+)\]")
        if "team_link" in download_df.columns:
            download_df["team"] = download_df["team_link"].str.extract(r"\[([^\]]+)\]")
        download_df = download_df.drop(columns=[c for c in ["player_link", "team_link"] if c in download_df.columns])

        return out.to_dict("records"), kpi, download_df.to_dict("list")

    # ---------- Toggle expansion ----------
    @app.callback(
        Output("pl-expanded", "data", allow_duplicate=True),
        Input("pl-table", "active_cell"),
        State("pl-table", "derived_virtual_data"),
        State("pl-expanded", "data"),
        prevent_initial_call=True,
    )
    def _toggle_expand(active_cell, rows, expanded_map):
        if not active_cell or not rows:
            return no_update
        if active_cell.get("column_id") != "expand":
            return no_update
        r = active_cell.get("row")
        if r is None or r < 0 or r >= len(rows):
            return no_update
        uid = rows[r].get("uid")
        if not uid:
            return no_update
        expanded_map = dict(expanded_map or {})
        expanded_map[uid] = not expanded_map.get(uid, False)
        return expanded_map

    # ---------- Render expansions ----------
    @app.callback(
        Output("pl-expansions", "children"),
        Input("pl-expanded", "data"),
        State("pl-table", "derived_virtual_data"),
    )
    def _render_expanded(expanded_map, rows):
        if not expanded_map or not rows:
            return []
        visible = {r["uid"]: r for r in rows if "uid" in r}
        panels = []
        for uid, is_open in expanded_map.items():
            if not is_open or uid not in visible:
                continue
            row = visible[uid]
            gl = _build_game_log(uid, limit=12)
            if gl.empty:
                body = html.Div("No game-by-game rows found for this player.", style={"color": "#888"})
            else:
                cols = [{"name": c, "id": c} for c in gl.columns]
                body = dash_table.DataTable(
                    columns=cols,
                    data=gl.to_dict("records"),
                    page_size=min(12, len(gl)),
                    style_table={"overflowX": "auto"},
                    style_header={"fontWeight": "600", "backgroundColor": "#f7f7fb"},
                    style_cell={"padding": "6px", "minWidth": 60, "whiteSpace": "nowrap"},
                )
            panels.append(html.Div([
                html.Hr(),
                html.Div([
                    html.Strong(f"{row.get('player_link','').split('](')[0].strip('[')}"),
                    html.Span(" — Game Log", style={"color": "#666", "marginLeft": "6px"}),
                ], style={"marginBottom": "6px"}),
                body
            ]))
        return panels

    # ---------- Download current view ----------
    @app.callback(
        Output("pl-download", "data"),
        Input("pl-download-btn", "n_clicks"),
        State("pl-last-dataset", "data"),
        prevent_initial_call=True,
    )
    def _download(n, cached):
        if not n or not cached:
            return no_update
        try:
            df = pd.DataFrame(cached)
        except Exception:
            return no_update
        return dcc.send_data_frame(df.to_csv, "players_current_view.csv", index=False)

    # ---------- Reset filters ----------
    @app.callback(
        Output("pl-team", "value"),
        Output("pl-pos", "value"),
        Output("pl-search", "value"),
        Output("pl-view", "value"),
        Output("pl-min-gp", "value"),
        Output("pl-min-min", "value"),
        Output("pl-week-range", "value"),
        Output("pl-compact", "value"),
        Input("pl-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_filters(n_clicks):
        if not n_clicks:
            return no_update
        return None, None, "", "both", 0, 0, [WEEK_MIN, WEEK_MAX], []

    # ---------- Compact mode → set root data-attribute for CSS ----------
    @app.callback(
        Output("pl-root", "data-compact"),
        Input("pl-compact", "value"),
        prevent_initial_call=False,
    )
    def _apply_compact_mode(vals):
        return "1" if ("compact" in (vals or [])) else "0"

    return app


# =============================================================================
# ---- Small view helpers ------------------------------------------------------
# =============================================================================
def _kpi(label: str, value: Any) -> html.Div:
    return html.Div([
        html.Div(str(label), className="kpi-label"),
        html.Div(str(value), className="kpi-value"),
    ], className="kpi")


def _week_marks(lo: int, hi: int) -> Dict[int, str]:
    span = max(hi - lo, 1)
    step = 1 if span <= 12 else 2 if span <= 30 else 4
    marks = {w: str(w) for w in range(lo, hi + 1, step)}
    marks[lo] = str(lo)
    marks[hi] = str(hi)
    return marks
