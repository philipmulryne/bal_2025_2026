# extractor_stints_fixed.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

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
    "validation": "validation_issues.csv",
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
    "pg": "Guard",
    "point": "Guard",
    "pointguard": "Guard",
    "guard": "Guard",
    "g": "Guard",
    "sg": "Guard",
    "sf": "Forward",
    "pf": "Forward",
    "forward": "Forward",
    "f": "Forward",
    "c": "Center",
    "center": "Center",
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
    """
    Legacy pid-only index (not used for team decisions).
    """
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

def _teams_for_pid(pid: int, player_pair_map: Dict[Tuple[int, int], Dict]) -> set[int]:
    return {t for (t, p) in player_pair_map.keys() if p == pid}

def _pinfo(team_no, pid, player_pair_map) -> Dict:
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
            return rec or {}
        except Exception:
            return {}
    teams = _teams_for_pid(pid_i, player_pair_map)
    if len(teams) == 1:
        t = next(iter(teams))
        return player_pair_map.get((t, pid_i), {})
    return {}

def _pid_team(pid, player_pair_map, tno_hint=None) -> Optional[int]:
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
    teams = _teams_for_pid(pid_i, player_pair_map)
    if len(teams) == 1:
        return next(iter(teams))
    return None

def _pid_belongs_to_team(pid, team_no, player_pair_map) -> bool:
    try:
        return (int(team_no), int(pid)) in player_pair_map
    except Exception:
        return False

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
        return ("SUB", st or "", f"{team_name} substitution {st}".strip())
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
                            player_pair_map: Dict[Tuple[int,int], Dict]) -> List[int]:
    # Prefer declared starters
    starters = [p for (t, p) in player_pair_map.keys()
                if t == team_no and player_pair_map[(t, p)].get("starter") in (1, True)]
    starters = sorted(starters)[:5]
    if len(starters) == 5:
        return starters

    # Fallback: first-period true appearances
    seen: List[int] = []
    p1 = pbp_df[pbp_df["period"] == 1]
    mask = (p1["tno"] == team_no) | (p1["pno"].map(lambda p: _pid_team(p, player_pair_map, tno_hint=team_no)) == team_no)
    for _, ev in p1[mask].iterrows():
        pid = ev.get("pno")
        if pd.notna(pid):
            pid = int(pid)
            if _pid_belongs_to_team(pid, team_no, player_pair_map) and pid not in seen:
                seen.append(pid)
        if len(seen) == 5:
            break
    if len(seen) < 5:
        rest = [p for (t,p) in player_pair_map.keys() if t == team_no and p not in seen]
        seen.extend(sorted(rest)[:max(0, 5 - len(seen))])
    return sorted(seen)[:5]

# ---------- SUB helpers (fixed) ----------

_SUB_IN_KEYS = ("playerInId","playerIn","inPlayerId","inPlayer","pno_in")
_SUB_OUT_KEYS = ("playerOutId","playerOut","outPlayerId","outPlayer","pno_out")

def _to_int_or_none(v):
    try:
        return int(v) if pd.notna(v) and v is not None and str(v) != "" else None
    except Exception:
        return None

def _sub_in_out(ev: pd.Series,
                team_no: int,
                player_pair_map: Dict[Tuple[int,int], Dict]) -> Tuple[Optional[int], Optional[int]]:
    """
    Interpret substitutions in vendor-neutral way:
      * Prefer explicit subType == 'in'/'out' + pno (the Genius/FIBA pattern).
      * Fall back to vendor-specific fields if present.
      * Guard by (team_no, pid) membership.
    """
    st = _clean_str(ev.get("subType")).lower()
    pid_from_pno = _to_int_or_none(ev.get("pno"))

    in_pid = None
    out_pid = None
    if st in ("in","out") and pid_from_pno is not None:
        if st == "in":
            in_pid = pid_from_pno
        else:
            out_pid = pid_from_pno
    else:
        # legacy vendors: try explicit fields
        for k in _SUB_IN_KEYS:
            if k in ev and pd.notna(ev[k]):
                in_pid = _to_int_or_none(ev[k]); break
        for k in _SUB_OUT_KEYS:
            if k in ev and pd.notna(ev[k]):
                out_pid = _to_int_or_none(ev[k]); break
        # last-resort guess: if only pno present, assume IN for our team
        if in_pid is None and out_pid is None and pid_from_pno is not None:
            if _pid_belongs_to_team(pid_from_pno, team_no, player_pair_map):
                in_pid = pid_from_pno

    # hard fence to our team
    if in_pid is not None and not _pid_belongs_to_team(in_pid, team_no, player_pair_map):
        in_pid = None
    if out_pid is not None and not _pid_belongs_to_team(out_pid, team_no, player_pair_map):
        out_pid = None

    return in_pid, out_pid

# --- NEW: robust team resolution for SUB rows ---

def _name_to_teams_map(player_pair_map):
    m = {}
    for (tno, pid), rec in player_pair_map.items():
        nm = _clean_str(rec.get("name")).lower()
        nm = " ".join(nm.split())
        if nm:
            m.setdefault(nm, set()).add(int(tno))
    return m

def _which_team_for_sub_row(ev: pd.Series,
                            player_pair_map: Dict[Tuple[int,int], Dict],
                            valid_team_nos: set[int]) -> Optional[int]:
    """
    Decide which team a SUB row belongs to, even if vendor 'tno' is wrong/missing.
    Order of checks:
      1) If tno is one of the two teams, use it.
      2) Use player display name -> team map (if unique).
      3) Use unique pno -> team (if pid appears on exactly one team).
    """
    # 1) trust good tno if valid
    tno = _to_int_or_none(ev.get("tno"))
    if tno in valid_team_nos:
        return tno

    # 2) unique name
    if not hasattr(_which_team_for_sub_row, "_name_map"):
        _which_team_for_sub_row._name_map = _name_to_teams_map(player_pair_map)  # cache across calls
    nm = _clean_str(ev.get("player") or ev.get("player_name_from_roster")).lower()
    nm = " ".join(nm.split())
    teams = _which_team_for_sub_row._name_map.get(nm)
    if teams and len(teams) == 1:
        return next(iter(teams))

    # 3) unique pid
    pid = _to_int_or_none(ev.get("pno"))
    if pid is not None:
        teams = {t for (t, p) in player_pair_map.keys() if p == pid}
        if len(teams) == 1:
            return next(iter(teams))

    return None

def _sub_clusters_for_team(pbp_df: pd.DataFrame,
                           team_no: int,
                           player_pair_map: Dict[Tuple[int,int], Dict]) -> List[Dict]:
    """
    Collapse all SUB rows that affect this team into clusters at a stoppage
    (grouped by period+clock). Each cluster holds sets of outs and ins.
    Robustly resolves which team each SUB row belongs to; avoids cross-team bleed.
    """
    subs = pbp_df[pbp_df["ev_code"] == "SUB"].copy()
    if subs.empty:
        return []

    valid_team_nos = {t for (t, _) in player_pair_map.keys()}

    # keep only SUB rows resolved for *this* team
    subs = subs[subs.apply(lambda r: _which_team_for_sub_row(r, player_pair_map, valid_team_nos) == int(team_no), axis=1)]
    if subs.empty:
        return []

    # Normalize fields needed for grouping
    for c in ("period", "actionNumber"):
        if c in subs.columns:
            subs[c] = pd.to_numeric(subs[c], errors="coerce")
    subs["clock"] = subs["clock"].astype(str)

    clusters = []
    # Group by stoppage signature (period + clock)
    for (per, clk), grp in subs.groupby(["period", "clock"], dropna=False):
        if pd.isna(per):
            continue

        # parse ins/outs within the cluster
        ins: Set[int] = set()
        outs: Set[int] = set()
        for _, r in grp.sort_values("actionNumber").iterrows():
            i_pid, o_pid = _sub_in_out(r, team_no, player_pair_map)
            if o_pid is not None:
                outs.add(int(o_pid))
            if i_pid is not None:
                ins.add(int(i_pid))

        if not ins and not outs:
            continue

        clusters.append({
            "period": int(per),
            "clock": clk,
            "tno": int(team_no),
            "action_min": float(grp["actionNumber"].min()),
            "outs": sorted(outs),
            "ins": sorted(ins),
        })

    # Sort by earliest action number (chronology)
    clusters.sort(key=lambda x: x["action_min"])
    return clusters

def _tokens_for_lineup(team_no: int, pids: List[int]) -> List[str]:
    return [f"{int(team_no)}:{int(pid)}" for pid in pids]

def _names_for_lineup(team_no: int, pids: List[int],
                      player_pair_map: Dict[Tuple[int,int], Dict]) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for pid in pids:
        rec = player_pair_map.get((int(team_no), int(pid)))
        out.append(rec.get("name") if rec else None)
    return out

def infer_stints(pbp_df: pd.DataFrame,
                 team_map: Dict[int, Dict],
                 player_pair_map: Dict[Tuple[int,int], Dict]) -> pd.DataFrame:
    rows = []

    # Fast path: no subs at all -> single stint per team
    if (pbp_df["ev_code"] == "SUB").sum() == 0:
        for team_no in sorted(team_map.keys()):
            lineup = [pid for pid in initial_lineup_for_team(team_no, pbp_df, player_pair_map)
                      if _pid_belongs_to_team(pid, team_no, player_pair_map)]
            rows.append({
                "team_no": team_no,
                "team_name": team_map[team_no]["team_name"],
                "start_action": float(pbp_df["actionNumber"].min()),
                "end_action": float(pbp_df["actionNumber"].max()),
                "players_on_court": sorted(set(lineup)),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["lineup_size"] = df["players_on_court"].map(len)
            df["lineup_uid"] = df["players_on_court"].map(lambda lst: "|".join(map(str, sorted(set(lst)))))
            df["players_on_court_tokens"] = df.apply(
                lambda r: _tokens_for_lineup(int(r["team_no"]), r["players_on_court"]), axis=1
            )
            df["players_on_court_names"] = df.apply(
                lambda r: _names_for_lineup(int(r["team_no"]), r["players_on_court"], player_pair_map), axis=1
            )
            df["lineup_uid_pair"] = df["players_on_court_tokens"].map(lambda lst: "|".join(lst))
        return df

    game_start = float(pbp_df["actionNumber"].min())
    game_end   = float(pbp_df["actionNumber"].max())

    for team_no in sorted(team_map.keys()):
        current = set(initial_lineup_for_team(team_no, pbp_df, player_pair_map))
        current = {pid for pid in current if _pid_belongs_to_team(pid, team_no, player_pair_map)}
        start_action = game_start

        clusters = _sub_clusters_for_team(pbp_df, team_no, player_pair_map)

        # Recent history bag for repair if we ever dip below 5
        seen_for_team: List[int] = [p for p in current]

        for cl in clusters:
            end_action = float(cl["action_min"])

            # 1) Emit the stint up to this stoppage (if time advanced)
            if end_action > start_action:
                rows.append({
                    "team_no": team_no,
                    "team_name": team_map[team_no]["team_name"],
                    "start_action": start_action,
                    "end_action": end_action,
                    "players_on_court": sorted(current),
                })

            # 2) Apply OUTs then INs atomically
            for o in cl["outs"]:
                if o in current:
                    current.remove(o)
            for i in cl["ins"]:
                current.add(i); seen_for_team.append(i)

            # 3) Hard fence + repair
            current = {pid for pid in current if _pid_belongs_to_team(pid, team_no, player_pair_map)}

            # Overfill: keep newest INs first, then lowest pid
            if len(current) > 5:
                keep = []
                for i in reversed([p for p in cl["ins"] if p in current]):
                    if i not in keep:
                        keep.append(i)
                for p in sorted(current):
                    if p not in keep:
                        keep.append(p)
                    if len(keep) == 5:
                        break
                current = set(keep)

            # Underfill: top up from recent history/roster
            if len(current) < 5:
                candidates = []
                for p in reversed(seen_for_team):
                    if p not in current and _pid_belongs_to_team(p, team_no, player_pair_map):
                        candidates.append(p)
                        if len(current) + len(candidates) == 5:
                            break
                if candidates:
                    current.update(candidates[:max(0, 5 - len(current))])
                if len(current) < 5:
                    bench = [p for (t,p) in player_pair_map.keys() if t == team_no and p not in current]
                    current.update(sorted(bench)[:max(0, 5 - len(current))])

            start_action = max(start_action, end_action) + 0.1

        # Tail stint
        if start_action < game_end + 1e-6:
            rows.append({
                "team_no": team_no,
                "team_name": team_map[team_no]["team_name"],
                "start_action": start_action,
                "end_action": game_end,
                "players_on_court": sorted(current),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        # final sanitize (+ team-aware exports)
        df["players_on_court"] = df.apply(
            lambda r: sorted([pid for pid in r["players_on_court"]
                              if _pid_belongs_to_team(pid, int(r["team_no"]), player_pair_map)]),
            axis=1
        )
        df["lineup_size"] = df["players_on_court"].map(len)
        df["lineup_uid"] = df["players_on_court"].map(lambda lst: "|".join(map(str, sorted(set(lst)))))

        # pair-safe, unambiguous fields
        df["players_on_court_tokens"] = df.apply(
            lambda r: _tokens_for_lineup(int(r["team_no"]), r["players_on_court"]),
            axis=1
        )
        df["players_on_court_names"] = df.apply(
            lambda r: _names_for_lineup(int(r["team_no"]), r["players_on_court"], player_pair_map),
            axis=1
        )
        df["lineup_uid_pair"] = df["players_on_court_tokens"].map(lambda lst: "|".join(lst))
    return df

def lineup_pos_mix(players_on_court: List[int],
                   player_pair_map: Dict[Tuple[int,int], Dict],
                   team_no: Optional[int] = None) -> str:
    groups = []
    for pid in players_on_court:
        rec = _pinfo(team_no, pid, player_pair_map)
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
            "lineup_uid": st.get("lineup_uid"),
            "possessions": poss_for,
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
        if seg.empty: return None, None
        r = seg.iloc[0]
    else:
        seg = pbp_df[pbp_df["actionNumber"] <= action]
        if seg.empty: return None, None
        r = seg.iloc[-1]
    return (int(r["period"]) if pd.notna(r.get("period")) else None, r.get("clock"))

def build_lineup_matchups(stints_df: pd.DataFrame,
                          possum_df: pd.DataFrame,
                          team_nos: List[int],
                          team_map: Dict[int, Dict],
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
        e = min(a["end_action"], b["end_action"])
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
                "start_action": s,
                "end_action": e,
                "start_period": sp,
                "start_clock": sc,
                "end_period": ep,
                "end_clock": ec,
                "teamA_no": t1,
                "teamA_name": team_map[t1]["team_name"],
                "lineupA_players": a["players_on_court"],
                "lineupA_uid": a.get("lineup_uid"),
                "lineupA_pos_mix": lineup_pos_mix(a["players_on_court"], player_pair_map, team_no=t1),
                # explicit identity
                "lineupA_tokens": [f"{t1}:{int(pid)}" for pid in a["players_on_court"]],
                "lineupA_names":  [_pinfo(t1, pid, player_pair_map).get("name") for pid in a["players_on_court"]],
                "teamB_no": t2,
                "teamB_name": team_map[t2]["team_name"],
                "lineupB_players": b["players_on_court"],
                "lineupB_uid": b.get("lineup_uid"),
                "lineupB_pos_mix": lineup_pos_mix(b["players_on_court"], player_pair_map, team_no=t2),
                # explicit identity
                "lineupB_tokens": [f"{t2}:{int(pid)}" for pid in b["players_on_court"]],
                "lineupB_names":  [_pinfo(t2, pid, player_pair_map).get("name") for pid in b["players_on_court"]],
                "poss_A": poss_A,
                "pts_A_for": pts_A_for,
                "pts_A_against": pts_B_for,
                "net_A": pts_A_for - pts_B_for,
                "ORtg_A": ORtg_A,
                "DRtg_A": DRtg_A,
                "poss_B": poss_B,
                "pts_B_for": pts_B_for,
                "pts_B_against": pts_A_for,
                "net_B": pts_B_for - pts_A_for,
                "ORtg_B": ORtg_B,
                "DRtg_B": DRtg_B,
                "score_start_A": sA_for,
                "score_start_B": sB_for,
                "score_end_A": eA_for,
                "score_end_B": eB_for,
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

def _validate_stints(stints_df: pd.DataFrame,
                     player_pair_map: Dict[Tuple[int,int], Dict]) -> pd.DataFrame:
    """
    Return a dataframe of issues (lineup_size != 5 or cross-team pid present).
    """
    if stints_df.empty:
        return pd.DataFrame()
    issues = []
    for _, r in stints_df.iterrows():
        team_no = int(r["team_no"])
        pids = list(r["players_on_court"])
        size_ok = (len(pids) == 5)
        cross = [pid for pid in pids if not _pid_belongs_to_team(pid, team_no, player_pair_map)]
        if (not size_ok) or cross:
            issues.append({
                "team_no": team_no,
                "start_action": r["start_action"],
                "end_action": r["end_action"],
                "players_on_court": pids,
                "players_on_court_tokens": "|".join(_tokens_for_lineup(team_no, pids)),
                "lineup_size": len(pids),
                "cross_team_pids": cross,
            })
    return pd.DataFrame(issues)

def process_one_game(src_json: Path,
                     out_dir: Path,
                     override_csv: Optional[Path],
                     week_name: str,
                     validate: bool = False) -> GameMeta:
    raw = json.loads(src_json.read_text(encoding="utf-8"))
    team_map = build_team_map(raw)
    player_map_legacy = build_player_map(raw)  # columns only
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
    pbp_df["player_name_from_roster"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_pair_map).get("name"), axis=1)
    pbp_df["player_shirt"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_pair_map).get("shirtNumber"), axis=1)
    pbp_df["player_pos_raw"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_pair_map).get("playingPosition_raw"), axis=1)
    pbp_df["player_pos_group"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_pair_map).get("position_group"), axis=1)

    # Merge shot coordinates if provided
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

    # stitch assists/blocks by previousAction
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
    pbp_df.loc[is_shot, "assist_pno"] = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: assist_map.get(an, {}).get("assist_pno"))
    pbp_df.loc[is_shot, "assist_tno"] = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: assist_map.get(an, {}).get("assist_tno"))
    pbp_df.loc[is_shot, "assist_player"] = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: assist_map.get(an, {}).get("assist_player"))

    pbp_df.loc[is_shot, "block_pno"] = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: block_map.get(an, {}).get("block_pno"))
    pbp_df.loc[is_shot, "block_tno"] = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: block_map.get(an, {}).get("block_tno"))
    pbp_df.loc[is_shot, "block_player"] = pbp_df.loc[is_shot, "actionNumber"].map(lambda an: block_map.get(an, {}).get("block_player"))

    pbp_df["assist_pos_group"] = pbp_df.apply(lambda r: _pinfo(r.get("assist_tno"), r.get("assist_pno"), player_pair_map).get("position_group"), axis=1)
    pbp_df["block_pos_group"] = pbp_df.apply(lambda r: _pinfo(r.get("block_tno"), r.get("block_pno"), player_pair_map).get("position_group"), axis=1)

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
        is_change, reason = possession_change(pd.Series(ev), pd.Series(nxt) if nxt is not None else None)
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
    stints_df = infer_stints(pbp_df, team_map, player_pair_map)
    stints_df["lineup_pos_mix"] = stints_df.apply(
        lambda r: lineup_pos_mix(r["players_on_court"], player_pair_map, int(r["team_no"])),
        axis=1
    )
    stints_df["lineup_size"] = stints_df["players_on_court"].map(lambda lst: len(lst))

    # Optional validation
    issues_df = _validate_stints(stints_df, player_pair_map)
    if validate and not issues_df.empty:
        print(f"[VALIDATE] Found {len(issues_df)} lineup issues (see CSV).", file=sys.stderr)

    stints_pm_df = stint_plus_minus_with_ratings(stints_df, possum_df, pbp_df)

    # timing & scores on stints
    team_nos = sorted(list(team_map.keys()))
    pbp_df = add_cumulative_scores(pbp_df, team_nos)

    time_cols = {
        "start_period": [], "start_clock": [],
        "end_period": [], "end_clock": [],
        "score_start_for": [], "score_start_against": [],
        "score_end_for": [], "score_end_against": [],
    }
    for _, st in stints_df.iterrows():
        sp, sc = period_clock_at_action(pbp_df, st["start_action"], "start")
        ep, ec = period_clock_at_action(pbp_df, st["end_action"], "end")
        s_for, s_against = score_for_team_at_action(pbp_df, team_nos, int(st["team_no"]), st["start_action"], inclusive=False)
        e_for, e_against = score_for_team_at_action(pbp_df, team_nos, int(st["team_no"]), st["end_action"], inclusive=True)
        time_cols["start_period"].append(sp);   time_cols["start_clock"].append(sc)
        time_cols["end_period"].append(ep);     time_cols["end_clock"].append(ec)
        time_cols["score_start_for"].append(s_for);     time_cols["score_start_against"].append(s_against)
        time_cols["score_end_for"].append(e_for);       time_cols["score_end_against"].append(e_against)
    for k, v in time_cols.items():
        stints_df[k] = v

    st_attrs = stints_df[[
        "team_no","start_action","end_action",
        "start_period","start_clock","end_period","end_clock",
        "score_start_for","score_start_against","score_end_for","score_end_against",
        "lineup_pos_mix","lineup_size","lineup_uid","lineup_uid_pair",
        "players_on_court","players_on_court_tokens","players_on_court_names","team_name"
    ]]

    stints_pm_df = stints_pm_df.merge(
        st_attrs,
        on=["team_no","start_action","end_action"],
        how="left",
        suffixes=("", "_st"),
        validate="many_to_one"
    )

    lineup_matchups_df = build_lineup_matchups(
        stints_df=stints_df,
        possum_df=possum_df,
        team_nos=team_nos,
        team_map=team_map,
        player_pair_map=player_pair_map,
        pbp_df=pbp_df
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

    # Pair token for unambiguous joins
    if not players_df.empty:
        players_df["pair_token"] = players_df.apply(
            lambda r: f"{int(r['team_no'])}:{int(r['player_no'])}", axis=1
        )

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
            players_df["position_raw"] = players_df["position_raw_ov1"].combine_first(players_df["position_raw"])
            players_df["position_group"] = players_df["position_group_ov1"].combine_first(players_df["position_group"])
            players_df.drop(columns=[c for c in players_df.columns if c.endswith("_ov1")], inplace=True)
        # by (team_code, shirtNumber)
        if {"team_code","shirtNumber"}.issubset(ov.columns):
            players_df = players_df.merge(
                ov[["team_code","shirtNumber","position_raw","position_group"]],
                on=["team_code","shirtNumber"], how="left", suffixes=("","_ov2")
            )
            players_df["position_raw"] = players_df["position_raw_ov2"].combine_first(players_df["position_raw"])
            players_df["position_group"] = players_df["position_group_ov2"].combine_first(players_df["position_group"])
            players_df.drop(columns=[c for c in players_df.columns if c.endswith("_ov2")], inplace=True)
        # by name
        if "name" in ov.columns:
            players_df = players_df.merge(
                ov[["name","position_raw","position_group"]],
                on="name", how="left", suffixes=("","_ov3")
            )
            players_df["position_raw"] = players_df["position_raw_ov3"].combine_first(players_df["position_raw"])
            players_df["position_group"] = players_df["position_group_ov3"].combine_first(players_df["position_group"])
            players_df.drop(columns=[c for c in players_df.columns if c.endswith("_ov3")], inplace=True)

        # Reflect overrides into pair map
        for _, r in players_df[["team_no","player_no","position_raw","position_group"]].dropna(subset=["player_no"]).iterrows():
            pid = int(r["player_no"]); tno = int(r["team_no"])
            key = (tno, pid)
            if key in player_pair_map:
                player_pair_map[key]["playingPosition_raw"] = r["position_raw"]
                player_pair_map[key]["position_group"] = r["position_group"]

        # refresh PBP pos cols (assist/block already use apply)
        pbp_df["player_pos_raw"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_pair_map).get("playingPosition_raw"), axis=1)
        pbp_df["player_pos_group"] = pbp_df.apply(lambda r: _pinfo(r["tno"], r["pno"], player_pair_map).get("position_group"), axis=1)
        # recompute stints pos mix (lineup composition unchanged)
        stints_df["lineup_pos_mix"] = stints_df.apply(
            lambda r: lineup_pos_mix(r["players_on_court"], player_pair_map, int(r["team_no"])),
            axis=1
        )

    pos_lookup = players_df[[
        "team_no","team_name","team_code","player_no","name","shirtNumber","position_raw","position_group","pair_token"
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
    if validate and not issues_df.empty:
        issues_df.to_csv(out_dir/CSV_NAMES["validation"], index=False, encoding="utf-8")

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

def process_week(week_json_dir: Path, out_root: Path, override_csv: Optional[Path], validate: bool=False) -> Path:
    week_name = week_json_dir.name
    out_week_dir = out_root / week_name
    out_week_dir.mkdir(parents=True, exist_ok=True)
    metas: List[dict] = []
    for src in sorted(week_json_dir.glob("*.json")):
        game_id = src.stem
        out_game_dir = out_week_dir / game_id
        meta = process_one_game(src, out_game_dir, override_csv, week_name, validate=validate)
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
    p.add_argument("--out-root", type=Path, default=Path("out_data"), help="Root output folder (default: ./out_data)")
    p.add_argument("--week", type=str, default=None, help="Week name for single --src mode (default: parent folder name or 'adhoc')")
    p.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE, help="Optional positions_override.csv")
    p.add_argument("--validate", action="store_true", help="Emit validation_issues.csv and print summary if any lineup issues are found.")
    args = p.parse_args()

    if args.src is None and args.week_dir is None:
        # fallback to legacy single-file behavior
        src = DEFAULT_SRC
        if not src.exists():
            print("No --src or --week-dir given and data.json not found.", file=sys.stderr)
            sys.exit(2)
        week_name = args.week or "adhoc"
        out_game_dir = args.out_root / week_name / src.stem
        process_one_game(src, out_game_dir, args.override, week_name, validate=args.validate)
        print(f"Wrote game outputs: {out_game_dir}")
        return

    if args.src:
        week_name = args.week or (args.src.parent.name if args.src.parent.name else "adhoc")
        out_game_dir = args.out_root / week_name / args.src.stem
        meta = process_one_game(args.src, out_game_dir, args.override, week_name, validate=args.validate)
        print(f"[{meta.week}] {meta.label} → {out_game_dir}")
        return

    if args.week_dir:
        out_week_dir = process_week(args.week_dir, args.out_root, args.override, validate=args.validate)
        print(f"Wrote week index + games: {out_week_dir}")
        return

if __name__ == "__main__":
    main()
