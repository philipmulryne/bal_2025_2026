# possessions_exact.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import json
import pandas as pd

# =============================================================================
# Event-driven, exact possession computation for Genius Sports (FIBA) data.json
# =============================================================================

@dataclass
class GameMetrics:
    week: Optional[str]
    game_id: str
    team1_name: str
    team1_code: Optional[str]
    team2_name: str
    team2_code: Optional[str]
    team1_points: int
    team2_points: int
    team1_possessions: int
    team2_possessions: int
    team1_ortg: float
    team2_ortg: float
    team1_drtg: float
    team2_drtg: float
    team1_net: float
    team2_net: float


def _is_last_ft(subtype: str) -> bool:
    """
    Genius 'freethrow' subType strings look like '1of2','2of2','1of1','1of3','3of3'.
    """
    if not subtype:
        return False
    if subtype in ("1of1", "free"):
        return True
    if "of" in subtype:
        a, b = subtype.split("of", 1)
        try:
            return int(a) == int(b)
        except Exception:
            return False
    return False


def _opp(t: Optional[int]) -> Optional[int]:
    return 3 - t if t in (1, 2) else None


def compute_exact_from_genius_json_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a dict:
      {
        'teams': {
           1: {'name': str, 'code': str|None, 'points': int, 'possessions': int,
               'ortg': float, 'drtg': float, 'net': float},
           2: { ... }
        }
      }
    """
    events: List[Dict[str, Any]] = sorted(data.get("pbp") or [], key=lambda e: e.get("actionNumber", 0))
    if not events:
        raise RuntimeError("PBP list is empty – cannot compute possessions.")

    tm = data.get("tm") or {}
    t1 = tm.get("1") or {}
    t2 = tm.get("2") or {}

    def _tname(tn: int) -> str:
        src = t1 if tn == 1 else t2
        return src.get("shortName") or src.get("name") or f"Team{tn}"

    def _tcode(tn: int) -> Optional[str]:
        src = t1 if tn == 1 else t2
        return src.get("code")

    # Final points from tm block is stable/reliable
    p1 = int(t1.get("score") or t1.get("full_score") or 0)
    p2 = int(t2.get("score") or t2.get("full_score") or 0)

    # State
    poss = {1: 0, 2: 0}
    pos_team: Optional[int] = None
    last_terminal: Optional[Tuple[str, Optional[int]]] = None
    recent_foul_type: Optional[str] = None

    def _start_pos(team: Optional[int], reason: str) -> None:
        nonlocal pos_team
        if team in (1, 2) and pos_team != team:
            poss[team] += 1
            pos_team = team

    def _end_to(team: Optional[int], cause: str) -> None:
        nonlocal last_terminal
        # Start new possession for the receiving team immediately (deterministic by rule)
        _start_pos(team, f"switch-{cause}")
        # 'last_terminal' stores (cause, previous_offense_team)
        last_terminal = (cause, _opp(team))

    for e in events:
        at = e.get("actionType")
        sub = (e.get("subType") or "").strip()
        tno = e.get("tno")
        succ = e.get("success", 0) or 0

        # Start-of-period control (jump ball won)
        if at == "jumpball" and sub == "won":
            _start_pos(tno, "jumpball-won")
            recent_foul_type = None
            continue

        # Live-ball team control actions
        if at in ("2pt", "3pt"):
            _start_pos(tno, "shot")
            if succ == 1:  # made FG ends this possession, defense inbound next
                _end_to(_opp(tno), "made_fg")
            continue

        if at == "rebound":
            if sub == "defensive":
                _end_to(tno, "def_reb")
            elif sub == "offensive":
                _start_pos(tno, "off_reb_claim")
            continue

        if at == "turnover":
            _start_pos(tno, "turnover-as-offense")
            _end_to(_opp(tno), "turnover")
            recent_foul_type = None
            continue

        if at == "foul":
            recent_foul_type = sub or "personal"
            if sub == "offensive":
                _start_pos(tno, "offensive-foul")
                _end_to(_opp(tno), "turnover_offensive_foul")
                recent_foul_type = None
            continue

        if at == "freethrow":
            _start_pos(tno, "free-throw")
            if _is_last_ft(sub):
                if succ == 1:
                    # Retain only for technical/unsportsmanlike/flagrant (defense penalized without turnover of possession)
                    if recent_foul_type in ("technical", "unsportsmanlike", "flagrant"):
                        recent_foul_type = None
                    else:
                        # Guard against and-1: if a made FG just ended it, do not end again
                        if not (last_terminal and last_terminal[0] == "made_fg" and last_terminal[1] == tno):
                            _end_to(_opp(tno), "last_ft_made")
                else:
                    # last FT missed: rebound will decide possession (handled above)
                    pass
            continue

        # Non-control events we ignore for possession boundaries:
        # ('assist','steal','block','timeout','substitution','period','foulon','headcoachchallenge', etc.)

    # Compose result
    def _metrics(tn: int, pf: int, pa: int, poss_t: int) -> Dict[str, Any]:
        return {
            "name": _tname(tn),
            "code": _tcode(tn),
            "points": int(pf),
            "possessions": int(poss_t),
            "ortg": round(100.0 * pf / poss_t, 2) if poss_t else None,
            "drtg": round(100.0 * pa / poss_t, 2) if poss_t else None,
            "net": round(100.0 * (pf - pa) / poss_t, 2) if poss_t else None,
        }

    return {
        "teams": {
            1: _metrics(1, p1, p2, poss[1]),
            2: _metrics(2, p2, p1, poss[2]),
        }
    }


# ----------------------------- IO utilities ----------------------------------

def compute_exact_from_genius_json_file(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return compute_exact_from_genius_json_dict(data)


def _game_id_from_path(p: Path) -> str:
    # Use stem if it's like /.../weekXX/<game_id>.json
    try:
        return p.stem
    except Exception:
        return "unknown"


def write_game_metrics_sidecars(
    out_root: Path,
    week: str,
    json_path: Path,
    metrics: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    Persist under out_root/week/<game_id>/:
      - game_metrics.json
      - game_metrics.csv
    And upsert into out_root/_cumulative/game_ratings.csv (and Parquet).
    """
    game_id = _game_id_from_path(json_path)
    game_dir = out_root / week / game_id
    game_dir.mkdir(parents=True, exist_ok=True)

    # Per-game, two rows (team1/team2)
    t1 = metrics["teams"][1]
    t2 = metrics["teams"][2]
    rows = [
        {
            "week": week,
            "game_id": game_id,
            "team_no": 1,
            "team_name": t1["name"],
            "team_code": t1.get("code"),
            "points_for": t1["points"],
            "points_against": t2["points"],
            "possessions": t1["possessions"],
            "ortg": t1["ortg"],
            "drtg": t1["drtg"],
            "net": t1["net"],
        },
        {
            "week": week,
            "game_id": game_id,
            "team_no": 2,
            "team_name": t2["name"],
            "team_code": t2.get("code"),
            "points_for": t2["points"],
            "points_against": t1["points"],
            "possessions": t2["possessions"],
            "ortg": t2["ortg"],
            "drtg": t2["drtg"],
            "net": t2["net"],
        },
    ]
    df = pd.DataFrame(rows)

    # Write per-game sidecars
    per_json = game_dir / "game_metrics.json"
    per_csv = game_dir / "game_metrics.csv"
    per_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(per_csv, index=False)

    # Upsert into cumulative game_ratings
    dest_dir = out_root / "_cumulative"
    dest_dir.mkdir(parents=True, exist_ok=True)
    cum_csv = dest_dir / "game_ratings.csv"
    if cum_csv.exists():
        existing = pd.read_csv(cum_csv)
        existing = existing[~((existing["week"].astype(str) == str(week)) &
                              (existing["game_id"].astype(str) == str(game_id)))]
        all_df = pd.concat([existing, df], ignore_index=True)
    else:
        all_df = df.copy()
    all_df.to_csv(cum_csv, index=False)

    # Parquet best-effort
    try:
        all_df.to_parquet(dest_dir / "game_ratings.parquet", index=False)
    except Exception:
        pass

    return per_json, per_csv


def rebuild_overall_team_ratings(out_root: Path) -> Path:
    """
    Aggregate _cumulative/game_ratings.csv -> _cumulative/team_ratings_overall.csv
    (totals & efficiencies across all games).
    """
    dest_dir = out_root / "_cumulative"
    src = dest_dir / "game_ratings.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found – run some games first.")
    df = pd.read_csv(src)

    # Aggregate by (team_name, team_code)
    g = df.groupby(["team_name", "team_code"], dropna=False).agg(
        games=("game_id", "nunique"),
        poss=("possessions", "sum"),
        pts_for=("points_for", "sum"),
        pts_against=("points_against", "sum"),
    ).reset_index()
    g["ortg"] = (100.0 * g["pts_for"] / g["poss"]).round(2)
    g["drtg"] = (100.0 * g["pts_against"] / g["poss"]).round(2)
    g["net"] = (g["ortg"] - g["drtg"]).round(2)

    out = dest_dir / "team_ratings_overall.csv"
    g.to_csv(out, index=False)
    try:
        g.to_parquet(dest_dir / "team_ratings_overall.parquet", index=False)
    except Exception:
        pass
    return out


# ----------------------------- Batch rebuild ---------------------------------

def rebuild_game_ratings_from_raw_json(raw_json_root: Path, out_root: Path, weeks: Optional[List[str]] = None) -> str:
    """
    Walk RAW JSON root (e.g., 'raw_json/week01/*.json'), compute & upsert per-game ratings,
    then write team-level overall aggregation.

    If 'weeks' is None -> all week folders under raw_json_root.
    """
    raw_json_root = raw_json_root.resolve()
    out_root = out_root.resolve()
    if not raw_json_root.exists():
        return f"[RATINGS ERROR] RAW JSON root not found → {raw_json_root}"

    # Clear cumulative game_ratings (hard rebuild)
    dest_dir = out_root / "_cumulative"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "game_ratings.csv").unlink(missing_ok=True)
    (dest_dir / "game_ratings.parquet").unlink(missing_ok=True)

    # Discover weeks
    if not weeks:
        weeks = [p.name for p in raw_json_root.iterdir() if p.is_dir()]
        weeks.sort()

    processed = 0
    for wk in weeks:
        wk_dir = raw_json_root / wk
        if not wk_dir.exists() or not wk_dir.is_dir():
            continue
        for jpath in wk_dir.glob("*.json"):
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metrics = compute_exact_from_genius_json_dict(data)
                write_game_metrics_sidecars(out_root, wk, jpath, metrics)
                processed += 1
            except Exception as e:
                # keep going
                print(f"[RATINGS WARN] {jpath} -> {type(e).__name__}: {e}")

    # Team-level overall
    try:
        overall = rebuild_overall_team_ratings(out_root)
        return f"[RATINGS OK] Processed={processed}\nWrote overall → {overall}"
    except Exception as e:
        return f"[RATINGS PARTIAL] Processed={processed}\nOverall aggregation failed: {type(e).__name__}: {e}"
