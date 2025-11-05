from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

# ============================================================
# CLI / IO MODEL
# ============================================================

@dataclass
class GameMeta:
    week: str
    game_id: str
    label: str
    teams: List[str]
    created_at: str
    source_json: str
    files: Dict[str, str]  # relative file paths we write

CSV_NAMES = {
    "pbp": "game_pbp_normalized.csv",
    "pos": "game_possessions.csv",
    "team_pm": "game_team_pm.csv",
    "stints": "game_stints.csv",
    "stints_pm": "game_stints_pm.csv",
    "matchups": "game_lineup_matchups.csv",
    "players": "game_players_with_positions.csv",
    "pos_lookup": "game_player_positions_lookup.csv",
}

# ============================================================
# CONFIG (default fallback for single-file mode)
# ============================================================

DEFAULT_SRC = Path("data.json")
DEFAULT_OVERRIDE = Path("positions_override.csv")

# ============================================================
# Position normalization
# ============================================================

_POS_SYNONYMS = {
    "pg": "Guard", "point": "Guard", "pointguard": "Guard", "guard": "Guard", "g": "Guard", "sg": "Guard",
    "sf": "Forward", "pf": "Forward", "forward": "Forward", "f": "Forward",
    "c": "Center", "center": "Center",
}

def _clean_str(x: Optional[str]) -> str:
    return str(x).strip() if x is not None else ""

def normalize_position_label(pos: Optional[str]) -> Tuple[str, str]:
    raw = _clean_str(pos)
    if not raw:
        return "", ""
    s = raw.replace(" ", "").replace("\\", "/").replace("-", "/").lower()

    if s in ("gf","g/f","fg","f/g"):
        return raw, "Guard-Forward"
    if s in ("fc","f/c","cf","c/f"):
        return raw, "Forward-Center"
    if s in _POS_SYNONYMS:
        return raw, _POS_SYNONYMS[s]
    if "guard" in s:
        return raw, "Guard"
    if "forward" in s:
        return raw, "Forward"
    if "center" in s or s == "c5" or s == "big":
        return raw, "Center"
    if len(s) <= 2 and s in _POS_SYNONYMS:
        return raw, _POS_SYNONYMS[s]
    return raw, ""

def coalesce_position_dict(p: Dict) -> Optional[str]:
    for k in ("playingPosition","position","pos","primaryPosition","positionShort","role"):
        v = p.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return None

# ============================================================
# Builders (team map, player map, normalization)
# ============================================================

def build_team_map(raw) -> Dict[int, Dict]:
    tmap = {}
    for tno, t in (raw.get("tm") or {}).items():
        try:
            tmap[int(tno)] = {
                "team_no": int(tno),
                "team_name": t.get("name"),
                "team_code": t.get("code"),
                "shortName": t.get("shortName"),
                "home": t.get("home") or t.get("isHome"),
            }
        except Exception:
            continue
    return tmap

def build_player_map(raw) -> Dict[int, Dict]:
    """Legacy: index by player_no only (jersey number). Kept for backward-compat fields."""
    pmap = {}
    for tno, t in (raw.get("tm") or {}).items():
        for pid, p in (t.get("pl") or {}).items():
            pos_raw_guess = coalesce_position_dict(p)
            pos_raw, pos_group = normalize_position_label(pos_raw_guess)
            try:
                pmap[int(pid)] = {
                    "player_no": int(pid),
                    "team_no": int(tno),
                    "firstName": p.get("firstName"),
                    "familyName": p.get("familyName"),
                    "name": p.get("name") or f"{p.get('firstName','')} {p.get('familyName','')}".strip(),
                    "shirtNumber": p.get("shirtNumber"),
                    "playingPosition_raw": pos_raw,
                    "position_group": pos_group,
                    "starter": p.get("starter"),
                    "captain": p.get("captain"),
                    "active": p.get("active"),
                }
            except Exception:
                continue
    return pmap

def build_player_pair_map(raw) -> Dict[Tuple[int, int], Dict]:
    """Canonical index: (team_no, player_no) to disambiguate shared jersey numbers across teams."""
    by_pair = {}
    for tno, t in (raw.get("tm") or {}).items():
        for pid, p in (t.get("pl") or {}).items():
            try:
                tno_i, pid_i = int(tno), int(pid)
            except Exception:
                continue
            pos_guess = coalesce_position_dict(p)
            pos_raw, pos_group = normalize_position_label(pos_guess)
            by_pair[(tno_i, pid_i)] = {
                "player_no": pid_i,
                "team_no": tno_i,
                "firstName": p.get("firstName"),
                "familyName": p.get("familyName"),
                "name": p.get("name") or f"{p.get('firstName','')} {p.get('familyName','')}".strip(),
                "shirtNumber": p.get("shirtNumber"),
                "playingPosition_raw": pos_raw,
                "position_group": pos_group,
                "starter": p.get("starter"),
                "captain": p.get("captain"),
                "active": p.get("active"),
            }
    return by_pair

# ---------- Team-aware resolvers ----------

def _pinfo(team_no, pid, player_map, player_pair_map):
    """Prefer (team_no, pid); fall back to pid-only if needed."""
    if pd.isna(pid):
        return {}
    try:
        pid_i = int(pid)
    except Exception:
        return {}
    if pd.notna(team_no):
        try:
            tno_i = int(team_no)
            rec = player_pair_map.get((tno_i, pid_i))
            if rec:
                return rec
        except Exception:
            pass
    return player_map.get(pid_i, {})

def _pid_team(pid, player_map, player_pair_map, tno_hint=None):
    """Return team_no for a pid; if duplicated pid, use tno_hint to disambiguate."""
    if pd.isna(pid):
        return None
    try:
        pid_i = int(pid)
    except Exception:
        return None
    if tno_hint is not None:
        try:
            tno_i = int(tno_hint)
            if (tno_i, pid_i) in player_pair_map:
                return tno_i
        except Exception:
            pass
    rec = player_map.get(pid_i)
    return int(rec["team_no"]) if rec and rec.get("team_no") is not None else None

def _pid_belongs_to_team(pid, team_no, player_map, player_pair_map):
    return _pid_team(pid, player_map, player_pair_map, tno_hint=team_no) == int(team_no)

# ============================================================
# Normalization of events
# ============================================================

def normalize_row(ev: pd.Series) -> Tuple[str, Optional[str], str]:
    at = _clean_str(ev.get("actionType")).lower()
    st = _clean_str(ev.get("subType")).lower()
    success = ev.get("success")
    player = ev.get("player") or ev.get("player_name_from_roster") or ""
    team_name = ev.get("team_name") or ""

    if at in ("2pt","2pointer","2points"):
        return ("P2", "made" if success == 1 else "missed", f"{team_name} {player} 2Pt {'made' if success==1 else 'missed'}")
    if at in ("3pt","3pointer","3points"):
        return ("P3", "made" if success == 1 else "missed", f"{team_name} {player} 3Pt {'made' if success==1 else 'missed'}")
    if at in ("freethrow","free-throw","ft"):
        return ("FT", "made" if success == 1 else "missed", f"{team_name} {player} FT {st or ''} {'made' if success==1 else 'missed'}".strip())
    if at == "rebound":
        is_team = (not ev.get("player")) and (not ev.get("shirtNumber"))
        code = "TREB" if is_team else "REB"
        side = st or ""
        return (code, None, f"{team_name} {'team ' if is_team else ''}rebound {side}".strip())
    if at == "assist":
        return ("ASS", None, f"{team_name} {player} assist")
    if at == "turnover":
        return ("TO", None, f"{team_name} {player} turnover {st}".strip())
    if at == "steal":
        return ("ST", None, f"{team_name} {player} steal")
    if at == "block":
        return ("BS", None, f"{team_name} {player} block")
    if at == "foul":
        code = "RFOUL" if "offensive" in st else "FOUL"
        return (code, None, f"{team_name} {player} foul {st}".strip())
    if at == "timeout":
        return ("TIMEOUT", None, f"{team_name} timeout")
    if at in ("sub","substitution"):
        return ("SUB", None, f"{team_name} substitution")
    if at in ("jumpball","jump"):
        return ("JUMP", None, "jump ball")
    if at in ("violation","travel","double-dribble","lane-violation","lane"):
        return ("VIOL", None, f"{team_name} violation {st}".strip())
    return ("OTHER", None, f"{team_name} {at} {st}".strip())

def event_points(ev: pd.Series) -> int:
    if ev["ev_code"] == "P2" and ev["ev_result"] == "made":
        return 2
    if ev["ev_code"] == "P3" and ev["ev_result"] == "made":
        return 3
    if ev["ev_code"] == "FT" and ev["ev_result"] == "made":
        return 1
    return 0

def possession_change(ev_curr: pd.Series, ev_next: Optional[pd.Series]) -> Tuple[bool, Optional[str]]:
    code = ev_curr["ev_code"]
    st = _clean_str(ev_curr.get("subType")).lower()

    if code in ("P2","P3","FT") and ev_curr["ev_result"] == "made":
        if code in ("P2","P3"):
            return True, "score"
        subtype = st.replace(" ", "")
        if any(tag in subtype for tag in ["1of1","2of2","3of3"]):
            return True, "made_ft_last"
        return True, "made_ft"
    if code == "TO":
        return True, "turnover"
    if code == "RFOUL":
        return True, "offensive_foul"

    if code in ("P2","P3","FT") and ev_curr["ev_result"] == "missed":
        if ev_next is not None and ev_next["ev_code"] in ("REB","TREB"):
            next_st = _clean_str(ev_next.get("subType")).lower()
            if "defensive" in next_st or ev_next["ev_code"] == "TREB":
                return True, "def_reb"
    if code in ("REB","TREB"):
        if "defensive" in st or code == "TREB":
            return True, "def_reb"
    return False, None

# ============================================================
# Possessions, stints, ratings
# ============================================================

def summarize_possessions(df: pd.DataFrame, team_map: Dict[int, Dict]) -> pd.DataFrame:
    g = df.groupby("possession_id", as_index=False).agg(
        period=("period","first"),
        start_action=("actionNumber","first"),
        end_action=("actionNumber","last"),
        owner_tno=("possession_owner_tno","first"),
        end_reason=("possession_end_reason","last"),
        pts_owner=("points", "sum"),
    )
    def opp_pts(pid):
        seg = df[df["possession_id"] == pid]
        owner = g.loc[g["possession_id"] == pid, "owner_tno"].iloc[0]
        return int(seg.loc[seg["tno"] != owner, "points"].sum())
    g["pts_against"] = g["possession_id"].map(opp_pts)
    g["owner_team_name"] = g["owner_tno"].map(lambda t: team_map.get(int(t), {}).get("team_name") if pd.notna(t) else None)
    g["net_pts"] = g["pts_owner"] - g["pts_against"]
    return g

def initial_lineup_for_team(team_no: int,
                            pbp_df: pd.DataFrame,
                            player_map: Dict[int, Dict],
                            player_pair_map: Dict[Tuple[int,int], Dict]) -> List[int]:
    # 1) explicit starters (from roster)
    starters = [info["player_no"] for (t,p), info in player_pair_map.items()
                if t == team_no and info.get("starter") in (1, True)]
    if len(starters) >= 5:
        return starters[:5]

    # 2) fallback: first-period appearances with true team
    seen: List[int] = []
    p1 = pbp_df[pbp_df["period"] == 1]
    mask = (p1["tno"] == team_no) | (p1["pno"].map(lambda p: _pid_team(p, player_map, player_pair_map, tno_hint=team_no)) == team_no)
    for _, ev in p1[mask].iterrows():
        pid = ev.get("pno")
        if pd.notna(pid):
            pid = int(pid)
            if _pid_belongs_to_team(pid, team_no, player_map, player_pair_map) and pid not in seen:
                seen.append(pid)
                if len(seen) == 5:
                    break

    if len(seen) < 5:
        rest = [info["player_no"] for (t,p), info in player_pair_map.items()
                if t == team_no and info["player_no"] not in seen]
        seen.extend(rest[:max(0, 5 - len(seen))])
    return seen[:5]

def infer_stints(pbp_df: pd.DataFrame,
                 team_map: Dict[int, Dict],
                 player_map: Dict[int, Dict],
                 player_pair_map: Dict[Tuple[int,int], Dict]) -> pd.DataFrame:
    subs = pbp_df[pbp_df["ev_code"] == "SUB"].copy()
    if subs.empty:
        rows = []
        for team_no in sorted(team_map.keys()):
            lineup = [pid for pid in initial_lineup_for_team(team_no, pbp_df, player_map, player_pair_map)
                      if _pid_belongs_to_team(pid, team_no, player_map, player_pair_map)]
            rows.append({
                "team_no": team_no, "team_name": team_map[team_no]["team_name"],
                "start_action": float(pbp_df["actionNumber"].min()),
                "end_action":   float(pbp_df["actionNumber"].max()),
                "players_on_court": lineup,
            })
        return pd.DataFrame(rows)

    rows = []
    for team_no in sorted(team_map.keys()):
        current = set(initial_lineup_for_team(team_no, pbp_df, player_map, player_pair_map))
        current = {pid for pid in current if _pid_belongs_to_team(pid, team_no, player_map, player_pair_map)}
        start_action = float(pbp_df["actionNumber"].min())

        # take SUB rows where either tno==team_no OR the 'in' player's true team == team_no
        team_subs = subs[(subs["tno"] == team_no) |
                         (subs.apply(lambda r: _pid_belongs_to_team(r["pno"], team_no, player_map, player_pair_map), axis=1))] \
                         .sort_values("actionNumber")

        for _, ev in team_subs.iterrows():
            end_action = float(ev["actionNumber"])
            rows.append({
                "team_no": team_no, "team_name": team_map[team_no]["team_name"],
                "start_action": start_action, "end_action": end_action,
                "players_on_court": sorted(current),
            })

            in_pid = int(ev["pno"]) if pd.notna(ev.get("pno")) else None
            if in_pid is not None and not _pid_belongs_to_team(in_pid, team_no, player_map, player_pair_map):
                # misattributed sub; ignore for this team
                start_action = end_action + 0.1
                continue

            # try to read an explicit 'out' field if your feed has it
            out_pid = None
            for k in ("playerOutId","playerOut","outPlayerId","outPlayer","pno_out"):
                if k in ev and pd.notna(ev[k]):
                    try: out_pid = int(ev[k])
                    except Exception: pass
                    break
            if out_pid is not None and not _pid_belongs_to_team(out_pid, team_no, player_map, player_pair_map):
                out_pid = None

            # apply the change
            if out_pid is not None and out_pid in current:
                current.remove(out_pid)
            if in_pid is not None:
                current.add(in_pid)

            # hard guard: never allow cross-team ids
            current = {pid for pid in current if _pid_belongs_to_team(pid, team_no, player_map, player_pair_map)}

            # cap to 5 if a feed glitch duplicated someone
            if len(current) > 5:
                keep = set()
                if in_pid is not None: keep.add(in_pid)
                for pid in sorted(current, key=lambda x: (x in keep, x), reverse=True):
                    keep.add(pid)
                    if len(keep) == 5: break
                current = keep

            start_action = end_action + 0.1

        # tail
        rows.append({
            "team_no": team_no, "team_name": team_map[team_no]["team_name"],
            "start_action": start_action, "end_action": float(pbp_df["actionNumber"].max()),
            "players_on_court": sorted(current),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # final sanitize (+ lineup_size for convenience)
        df["players_on_court"] = df.apply(
            lambda r: sorted([pid for pid in r["players_on_court"]
                              if _pid_belongs_to_team(pid, int(r["team_no"]), player_map, player_pair_map)]),
            axis=1
        )
        df["lineup_size"] = df["players_on_court"].map(len)
    return df

def lineup_pos_mix(players_on_court: List[int],
                   player_map: Dict[int, Dict],
                   player_pair_map: Dict[Tuple[int,int], Dict],
                   team_no: Optional[int] = None) -> str:
    groups = []
    for pid in players_on_court:
        rec = _pinfo(team_no, pid, player_map, player_pair_map)
        groups.append(rec.get("position_group","") if rec else "")
    g = sum(1 for x in groups if x == "Guard")
    f = sum(1 for x in groups if x == "Forward")
    c = sum(1 for x in groups if x == "Center")
    u = sum(1 for x in groups if not x or x not in ("Guard","Forward","Center"))
    parts = []
    if g: parts.append(f"Gx{g}")
    if f: parts.append(f"Fx{f}")
    if c: parts.append(f"Cx{c}")
    if u: parts.append(f"Unknownx{u}")
    return ",".join(parts)

def stint_plus_minus_with_ratings(stints: pd.DataFrame,
                                  poss: pd.DataFrame,
                                  pbp_df: pd.DataFrame) -> pd.DataFrame:
    if stints.empty:
        return pd.DataFrame()
    rows = []
    for _, st in stints.iterrows():
        team_no = int(st["team_no"])
        s = float(st["start_action"])
        e = float(st["end_action"])

        evseg = pbp_df[(pbp_df["actionNumber"] > s) & (pbp_df["actionNumber"] <= e)]
        pts_for = int(evseg.loc[evseg["tno"] == team_no, "points"].sum())
        pts_against = int(evseg.loc[(evseg["tno"].notna()) & (evseg["tno"] != team_no), "points"].sum())

        pseg = poss[(poss["end_action"] > s) & (poss["end_action"] <= e)]
        poss_for = int(pseg.loc[pseg["owner_tno"] == team_no, "possession_id"].nunique())
        poss_against = int(pseg.loc[pseg["owner_tno"] != team_no, "possession_id"].nunique())

        ORtg = (100.0 * pts_for / poss_for) if poss_for > 0 else None
        DRtg = (100.0 * pts_against / poss_against) if poss_against > 0 else None

        rows.append({
            "team_no": st["team_no"],
            "team_name": st["team_name"],
            "start_action": st["start_action"],
            "end_action": st["end_action"],
            "players_on_court": st["players_on_court"],
            "possessions": poss_for,      # kept for compatibility
            "poss_for": poss_for,
            "poss_against": poss_against,
            "pts_for": pts_for,
            "pts_against": pts_against,
            "net": pts_for - pts_against,
            "ORtg": ORtg,
            "DRtg": DRtg,
        })
    return pd.DataFrame(rows)

# -------- Timing & Score helpers --------

def add_cumulative_scores(pbp_df: pd.DataFrame, team_nos: List[int]) -> pd.DataFrame:
    for tno in team_nos:
        pbp_df[f"pts_{tno}"] = pbp_df.apply(lambda r: int(r["points"]) if pd.notna(r.get("tno")) and int(r["tno"])==tno else 0, axis=1)
        pbp_df[f"cum_{tno}"] = pbp_df[f"pts_{tno}"].cumsum()
    return pbp_df

def score_for_team_at_action(pbp_df: pd.DataFrame, team_nos: List[int], team_no: int, action: float, inclusive: bool) -> Tuple[int,int]:
    assert len(team_nos) == 2, "Assumes exactly two teams."
    opp_no = team_nos[1] if team_nos[0]==team_no else team_nos[0]
    seg = pbp_df[pbp_df["actionNumber"] <= action] if inclusive else pbp_df[pbp_df["actionNumber"] < action]
    if seg.empty:
        return 0, 0
    last = seg.iloc[-1]
    return int(last[f"cum_{team_no}"]), int(last[f"cum_{opp_no}"])

def period_clock_at_action(pbp_df: pd.DataFrame, action: float, mode: str) -> Tuple[Optional[int], Optional[str]]:
    if mode == "start":
        seg = pbp_df[pbp_df["actionNumber"] >= action]
        if seg.empty:
            return None, None
        r = seg.iloc[0]
    else:
        seg = pbp_df[pbp_df["actionNumber"] <= action]
        if seg.empty:
            return None, None
        r = seg.iloc[-1]
    return (int(r["period"]) if pd.notna(r.get("period")) else None, r.get("clock"))

def build_lineup_matchups(stints_df: pd.DataFrame,
                          possum_df: pd.DataFrame,
                          team_nos: List[int],
                          team_map: Dict[int, Dict],
                          player_map: Dict[int, Dict],
                          player_pair_map: Dict[Tuple[int,int], Dict],
                          pbp_df: pd.DataFrame) -> pd.DataFrame:
    t1, t2 = team_nos
    A = stints_df[stints_df["team_no"]==t1].sort_values("start_action").reset_index(drop=True)
    B = stints_df[stints_df["team_no"]==t2].sort_values("start_action").reset_index(drop=True)
    i = j = 0
    rows = []
    while i < len(A) and j < len(B):
        a = A.iloc[i]
        b = B.iloc[j]
        s = max(a["start_action"], b["start_action"])
        e = min(a["end_action"],   b["end_action"])
        if s < e:
            mask = (possum_df["start_action"]>=s) & (possum_df["end_action"]<=e)
            seg = possum_df[mask]

            poss_A = int(seg[seg["owner_tno"]==t1]["possession_id"].nunique())
            poss_B = int(seg[seg["owner_tno"]==t2]["possession_id"].nunique())
            pts_A_for = int(seg[seg["owner_tno"]==t1]["pts_owner"].sum())
            pts_B_for = int(seg[seg["owner_tno"]==t2]["pts_owner"].sum())

            ORtg_A = (100.0 * pts_A_for / poss_A) if poss_A > 0 else None
            DRtg_A = (100.0 * (pts_B_for) / poss_B) if poss_B > 0 else None
            ORtg_B = (100.0 * pts_B_for / poss_B) if poss_B > 0 else None
            DRtg_B = (100.0 * (pts_A_for) / poss_A) if poss_A > 0 else None

            sp, sc = period_clock_at_action(pbp_df, s, "start")
            ep, ec = period_clock_at_action(pbp_df, e, "end")

            team_nos_order = [t1, t2]
            sA_for, _ = score_for_team_at_action(pbp_df, team_nos_order, t1, s, inclusive=False)
            sB_for, _ = score_for_team_at_action(pbp_df, team_nos_order, t2, s, inclusive=False)
            eA_for, _ = score_for_team_at_action(pbp_df, team_nos_order, t1, e, inclusive=True)
            eB_for, _ = score_for_team_at_action(pbp_df, team_nos_order, t2, e, inclusive=True)

            rows.append({
                "start_action": s, "end_action": e,
                "start_period": sp, "start_clock": sc, "end_period": ep, "end_clock": ec,
                "teamA_no": t1, "teamA_name": team_map[t1]["team_name"],
                "lineupA_players": a["players_on_court"],
                "lineupA_pos_mix": lineup_pos_mix(a["players_on_court"], player_map, player_pair_map, team_no=t1),
                "teamB_no": t2, "teamB_name": team_map[t2]["team_name"],
                "lineupB_players": b["players_on_court"],
                "lineupB_pos_mix": lineup_pos_mix(b["players_on_court"], player_map, player_pair_map, team_no=t2),
                "poss_A": poss_A, "pts_A_for": pts_A_for, "pts_A_against": pts_B_for,
                "net_A": pts_A_for - pts_B_for, "ORtg_A": ORtg_A, "DRtg_A": DRtg_A,
                "poss_B": poss_B, "pts_B_for": pts_B_for, "pts_B_against": pts_A_for,
                "net_B": pts_B_for - pts_A_for, "ORtg_B": ORtg_B, "DRtg_B": DRtg_B,
                "score_start_A": sA_for, "score_start_B": sB_for,
                "score_end_A": eA_for, "score_end_B": eB_for,
            })
        if a["end_action"] <= b["end_action"]:
            i += 1
        else:
            j += 1
    return pd.DataFrame(rows)

# ============================================================
# Core “process one” + metadata
# ============================================================

def _derive_label(team_map: Dict[int, Dict]) -> str:
    teams = [t.get("team_name") for t in team_map.values() if t.get("team_name")]
    teams = list(dict.fromkeys(teams))  # preserve order, unique
    if len(teams) == 2:
        return f"{teams[0]} vs {teams[1]}"
    return " / ".join(teams) if teams else "Unknown teams"

def process_one_game(src_json: Path, out_dir: Path, override_csv: Optional[Path], week_name: str) -> GameMeta:
    raw = json.loads(src_json.read_text(encoding="utf-8"))

    team_map = build_team_map(raw)
    player_map = build_player_map(raw)
    player_pair_map = build_player_pair_map(raw)

    pbp_df = pd.DataFrame(raw.get("pbp") or [])
    if pbp_df.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = GameMeta(
            week=week_name,
            game_id=src_json.stem,
            label=_derive_label(team_map),
            teams=[v["team_name"] for v in team_map.values() if v.get("team_name")],
            created_at=datetime.utcnow().isoformat() + "Z",
            source_json=str(src_json.resolve()),
            files={k: v for k, v in CSV_NAMES.items()},
        )
        (out_dir/"meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    for c in ["actionNumber","tno","pno","period","lead"]:
        if c in pbp_df.columns:
            pbp_df[c] = pd.to_numeric(pbp_df[c], errors="coerce")

    pbp_df = pbp_df.sort_values(["period","actionNumber"]).reset_index(drop=True)

    pbp_df["team_name"] = pbp_df["tno"].map(lambda t: team_map.get(int(t), {}).get("team_name") if pd.notna(t) else None)
    pbp_df["team_code"] = pbp_df["tno"].map(lambda t: team_map.get(int(t), {}).get("team_code") if pd.notna(t) else None)

    # team-aware player fields
    pbp_df["player_name_from_roster"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_map, player_pair_map).get("name"), axis=1)
    pbp_df["player_shirt"]            = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_map, player_pair_map).get("shirtNumber"), axis=1)
    pbp_df["player_pos_raw"]          = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_map, player_pair_map).get("playingPosition_raw"), axis=1)
    pbp_df["player_pos_group"]        = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_map, player_pair_map).get("position_group"), axis=1)

    # shots (x,y,r) merge if present
    shot_rows = []
    for tno, t in (raw.get("tm") or {}).items():
        for s in (t.get("shot") or []):
            rec = s.copy()
            rec["tno"] = int(tno)
            shot_rows.append(rec)
    if shot_rows:
        shots_df = pd.DataFrame(shot_rows)
        for c in ["x","y","r","pno","tno","actionNumber","per"]:
            if c in shots_df.columns:
                shots_df[c] = pd.to_numeric(shots_df[c], errors="coerce")
        shots_df.rename(columns={"per":"period"}, inplace=True)
        pbp_df = pbp_df.merge(shots_df[["actionNumber","x","y","r"]], on="actionNumber", how="left")

    norm = pbp_df.apply(normalize_row, axis=1, result_type="expand")
    pbp_df[["ev_code","ev_result","ev_text"]] = norm

    # stitch assists/blocks by previousAction (include tno to disambiguate same-pno players)
    assist_map, block_map = {}, {}
    for _, ev in pbp_df.iterrows():
        if ev.get("actionType") == "assist" and pd.notna(ev.get("previousAction")):
            assist_map[ev["previousAction"]] = {
                "assist_pno": ev.get("pno"),
                "assist_tno": ev.get("tno"),
                "assist_player": ev.get("player") or ev.get("player_name_from_roster"),
            }
        if ev.get("actionType") == "block" and pd.notna(ev.get("previousAction")):
            block_map[ev["previousAction"]] = {
                "block_pno": ev.get("pno"),
                "block_tno": ev.get("tno"),
                "block_player": ev.get("player") or ev.get("player_name_from_roster"),
            }

    is_shot = pbp_df["actionType"].isin(["2pt","3pt"])
    pbp_df.loc[is_shot, "assist_pno"]    = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: assist_map.get(an, {}).get("assist_pno"))
    pbp_df.loc[is_shot, "assist_tno"]    = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: assist_map.get(an, {}).get("assist_tno"))
    pbp_df.loc[is_shot, "assist_player"] = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: assist_map.get(an, {}).get("assist_player"))
    pbp_df.loc[is_shot, "block_pno"]     = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: block_map.get(an, {}).get("block_pno"))
    pbp_df.loc[is_shot, "block_tno"]     = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: block_map.get(an, {}).get("block_tno"))
    pbp_df.loc[is_shot, "block_player"]  = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: block_map.get(an, {}).get("block_player"))

    pbp_df["assist_pos_group"] = pbp_df.apply(lambda r: _pinfo(r.get("assist_tno"), r.get("assist_pno"), player_map, player_pair_map).get("position_group"), axis=1)
    pbp_df["block_pos_group"]  = pbp_df.apply(lambda r: _pinfo(r.get("block_tno"),  r.get("block_pno"),  player_map, player_pair_map).get("position_group"), axis=1)

    pbp_df["points"] = pbp_df.apply(event_points, axis=1)

    # possession ids
    pos_id = 0
    curr_pos_team: Optional[int] = None
    pos_ids: List[int] = []
    pos_owners: List[Optional[int]] = []
    pos_end_reason: List[Optional[str]] = []

    rows = pbp_df.to_dict(orient="records")
    for i, ev in enumerate(rows):
        if curr_pos_team is None:
            if ev["ev_code"] in ("P2","P3","FT","TO","RFOUL"):
                curr_pos_team = ev.get("tno")
                pos_id += 1
        pos_ids.append(pos_id if curr_pos_team is not None else 0)
        pos_owners.append(curr_pos_team)

        nxt = rows[i+1] if i+1 < len(rows) else None
        is_change, reason = possession_change(ev, nxt)
        if is_change:
            pos_end_reason.append(reason)
            if curr_pos_team in (1,2):
                curr_pos_team = 3 - curr_pos_team
            else:
                curr_pos_team = nxt.get("tno") if nxt is not None else None
            pos_id += 1
        else:
            pos_end_reason.append(None)

    pbp_df["possession_id"] = pos_ids
    pbp_df["possession_owner_tno"] = pos_owners
    pbp_df["possession_end_reason"] = pos_end_reason

    possum_df = summarize_possessions(pbp_df, team_map)
    team_pm = possum_df.groupby("owner_team_name", as_index=False).agg(
        poss=("possession_id","nunique"),
        pts_for=("pts_owner","sum"),
        pts_against=("pts_against","sum"),
        net=("net_pts","sum")
    )

    # stints & ratings
    stints_df = infer_stints(pbp_df, team_map, player_map, player_pair_map)
    stints_df["lineup_pos_mix"] = stints_df.apply(
        lambda r: lineup_pos_mix(r["players_on_court"], player_map, player_pair_map, int(r["team_no"])),
        axis=1
    )
    stints_df["lineup_size"] = stints_df["players_on_court"].map(lambda lst: len(lst))
    stints_pm_df = stint_plus_minus_with_ratings(stints_df, possum_df, pbp_df)

    # timing & scores on stints
    team_nos = sorted(list(team_map.keys()))
    pbp_df = add_cumulative_scores(pbp_df, team_nos)

    time_cols = {
        "start_period": [], "start_clock": [],
        "end_period": [],   "end_clock": [],
        "score_start_for": [], "score_start_against": [],
        "score_end_for": [],   "score_end_against": [],
    }
    for _, st in stints_df.iterrows():
        sp, sc = period_clock_at_action(pbp_df, st["start_action"], "start")
        ep, ec = period_clock_at_action(pbp_df, st["end_action"], "end")
        s_for, s_against = score_for_team_at_action(pbp_df, team_nos, int(st["team_no"]), st["start_action"], inclusive=False)
        e_for, e_against = score_for_team_at_action(pbp_df, team_nos, int(st["team_no"]), st["end_action"], inclusive=True)
        time_cols["start_period"].append(sp); time_cols["start_clock"].append(sc)
        time_cols["end_period"].append(ep);   time_cols["end_clock"].append(ec)
        time_cols["score_start_for"].append(s_for); time_cols["score_start_against"].append(s_against)
        time_cols["score_end_for"].append(e_for);   time_cols["score_end_against"].append(e_against)
    for k, v in time_cols.items():
        stints_df[k] = v

    st_attrs = stints_df[[
        "team_no","start_action","end_action",
        "start_period","start_clock","end_period","end_clock",
        "score_start_for","score_start_against","score_end_for","score_end_against",
        "lineup_pos_mix","lineup_size","players_on_court","team_name"
    ]]
    stints_pm_df = stints_pm_df.merge(
        st_attrs, on=["team_no","start_action","end_action"],
        how="left", suffixes=("", "_st"), validate="many_to_one"
    )

    lineup_matchups_df = build_lineup_matchups(
        stints_df=stints_df, possum_df=possum_df,
        team_nos=team_nos, team_map=team_map,
        player_map=player_map, player_pair_map=player_pair_map, pbp_df=pbp_df
    )

    # players table + optional overrides
    players_rows = []
    for tno, t in (raw.get("tm") or {}).items():
        for pid, p in (t.get("pl") or {}).items():
            row = {"team_no": int(tno), "team_name": t.get("name"), "team_code": t.get("code"), "player_no": int(pid)}
            for k in ["name","firstName","familyName","shirtNumber","starter","captain","active"]:
                row[k] = p.get(k)
            pos_guess = coalesce_position_dict(p)
            pos_raw, pos_group = normalize_position_label(pos_guess)
            row["position_raw"] = pos_raw
            row["position_group"] = pos_group
            for k, v in p.items():
                if k.startswith("s"):
                    row[k] = v
            players_rows.append(row)
    players_df = pd.DataFrame(players_rows)

    if override_csv and override_csv.exists() and not players_df.empty:
        ov = pd.read_csv(override_csv)
        ov.columns = [c.strip() for c in ov.columns]
        if "position" in ov.columns:
            ov["position_raw"], ov["position_group"] = zip(*ov["position"].astype(str).map(normalize_position_label))
        # by player_no
        if "player_no" in ov.columns:
            players_df = players_df.merge(
                ov[["player_no","position_raw","position_group"]],
                on="player_no", how="left", suffixes=("","_ov1")
            )
            players_df["position_raw"]   = players_df["position_raw_ov1"].combine_first(players_df["position_raw"])
            players_df["position_group"] = players_df["position_group_ov1"].combine_first(players_df["position_group"])
            players_df.drop(columns=[c for c in players_df.columns if c.endswith("_ov1")], inplace=True)
        # by (team_code, shirtNumber)
        if {"team_code","shirtNumber"}.issubset(ov.columns):
            players_df = players_df.merge(
                ov[["team_code","shirtNumber","position_raw","position_group"]],
                on=["team_code","shirtNumber"], how="left", suffixes=("","_ov2")
            )
            players_df["position_raw"]   = players_df["position_raw_ov2"].combine_first(players_df["position_raw"])
            players_df["position_group"] = players_df["position_group_ov2"].combine_first(players_df["position_group"])
            players_df.drop(columns=[c for c in players_df.columns if c.endswith("_ov2")], inplace=True)
        # by name
        if "name" in ov.columns:
            players_df = players_df.merge(
                ov[["name","position_raw","position_group"]],
                on="name", how="left", suffixes=("","_ov3")
            )
            players_df["position_raw"]   = players_df["position_raw_ov3"].combine_first(players_df["position_raw"])
            players_df["position_group"] = players_df["position_group_ov3"].combine_first(players_df["position_group"])
            players_df.drop(columns=[c for c in players_df.columns if c.endswith("_ov3")], inplace=True)

        # reflect overrides into both maps
        for _, r in players_df[["team_no","player_no","position_raw","position_group"]].dropna(subset=["player_no"]).iterrows():
            pid = int(r["player_no"]); tno = int(r["team_no"])
            if pid in player_map:
                player_map[pid]["playingPosition_raw"] = r["position_raw"]
                player_map[pid]["position_group"] = r["position_group"]
            key = (tno, pid)
            if key in player_pair_map:
                player_pair_map[key]["playingPosition_raw"] = r["position_raw"]
                player_pair_map[key]["position_group"] = r["position_group"]

        # refresh PBP pos cols (assist/block use apply already)
        pbp_df["player_pos_raw"]   = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_map, player_pair_map).get("playingPosition_raw"), axis=1)
        pbp_df["player_pos_group"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_map, player_pair_map).get("position_group"), axis=1)

        # recompute stints pos mix (lineup is unchanged)
        stints_df["lineup_pos_mix"] = stints_df.apply(
            lambda r: lineup_pos_mix(r["players_on_court"], player_map, player_pair_map, int(r["team_no"])),
            axis=1
        )

    pos_lookup = players_df[[
        "team_no","team_name","team_code","player_no","name","shirtNumber","position_raw","position_group"
    ]].sort_values(["team_no","name"])

    # WRITE
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/CSV_NAMES["pbp"]).write_text(pbp_df.to_csv(index=False), encoding="utf-8")
    (out_dir/CSV_NAMES["pos"]).write_text(summarize_possessions(pbp_df, team_map).to_csv(index=False), encoding="utf-8")
    (out_dir/CSV_NAMES["team_pm"]).write_text(team_pm.to_csv(index=False), encoding="utf-8")
    (out_dir/CSV_NAMES["stints"]).write_text(stints_df.to_csv(index=False), encoding="utf-8")
    (out_dir/CSV_NAMES["stints_pm"]).write_text(stints_pm_df.to_csv(index=False), encoding="utf-8")
    (out_dir/CSV_NAMES["matchups"]).write_text(lineup_matchups_df.to_csv(index=False), encoding="utf-8")
    (out_dir/CSV_NAMES["players"]).write_text(players_df.to_csv(index=False), encoding="utf-8")
    (out_dir/CSV_NAMES["pos_lookup"]).write_text(pos_lookup.to_csv(index=False), encoding="utf-8")

    label = _derive_label(team_map)
    meta = GameMeta(
        week=week_name,
        game_id=src_json.stem,
        label=label,
        teams=[v["team_name"] for v in team_map.values() if v.get("team_name")],
        created_at=datetime.utcnow().isoformat() + "Z",
        source_json=str(src_json.resolve()),
        files={k: v for k, v in CSV_NAMES.items()},
    )
    (out_dir/"meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

# ============================================================
# Batch per week + index
# ============================================================

def process_week(week_json_dir: Path, out_root: Path, override_csv: Optional[Path]) -> Path:
    week_name = week_json_dir.name
    out_week_dir = out_root / week_name
    out_week_dir.mkdir(parents=True, exist_ok=True)

    metas: List[dict] = []
    for src in sorted(week_json_dir.glob("*.json")):
        game_id = src.stem
        out_game_dir = out_week_dir / game_id
        meta = process_one_game(src, out_game_dir, override_csv, week_name)
        metas.append(asdict(meta))

    (out_week_dir/"week_index.json").write_text(json.dumps({
        "week": week_name,
        "games": metas,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_week_dir

# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Extract FIBA/Genius game(s) → CSVs with stints/possession/lineup matchups.")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--src", type=Path, help="Single game JSON (e.g., week01/2714807.json)")
    g.add_argument("--week-dir", type=Path, help="Folder of JSON games (e.g., week01/)")

    p.add_argument("--out-root", type=Path, default=Path("out_data"),
                   help="Root output folder (default: ./out_data)")
    p.add_argument("--week", type=str, default=None,
                   help="Week name for single --src mode (default: parent folder name or 'adhoc')")
    p.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE,
                   help="Optional positions_override.csv")

    args = p.parse_args()

    if args.src is None and args.week_dir is None:
        # fallback to legacy single-file behavior
        src = DEFAULT_SRC
        if not src.exists():
            print("No --src or --week-dir given and data.json not found.", file=sys.stderr)
            sys.exit(2)
        week_name = args.week or "adhoc"
        out_game_dir = args.out_root / week_name / src.stem
        process_one_game(src, out_game_dir, args.override, week_name)
        print(f"Wrote game outputs: {out_game_dir}")
        return

    if args.src:
        week_name = args.week or (args.src.parent.name if args.src.parent.name else "adhoc")
        out_game_dir = args.out_root / week_name / args.src.stem
        meta = process_one_game(args.src, out_game_dir, args.override, week_name)
        print(f"[{meta.week}] {meta.label} → {out_game_dir}")
        return

    if args.week_dir:
        out_week_dir = process_week(args.week_dir, args.out_root, args.override)
        print(f"Wrote week index + games: {out_week_dir}")
        return

if __name__ == "__main__":
    main()
