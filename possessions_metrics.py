# possessions_metrics.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import pandas as pd

# ----------------------------- Config ---------------------------------------
@dataclass
class PossessionConfig:
    ft_weight: float = 0.44
    # exact-mode toggles (schema-agnostic heuristics)
    treat_treb_as_defensive: bool = True
    retain_after_tech_unsports: bool = True
    lookahead_inbound_rows: int = 6

# -------------------------- Column helpers ----------------------------------
def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in low:
            return low[k.lower()]
    return None

def _booly(x) -> Optional[bool]:
    if pd.isna(x): return None
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in {"1","true","t","yes","y"}: return True
    if s in {"0","false","f","no","n"}: return False
    return None

# ---------------------- Net possessions (box-based) --------------------------
def _aggregate_box_from_players_csv(players_csv: Path) -> Optional[pd.DataFrame]:
    if not players_csv.exists():
        return None
    df = pd.read_csv(players_csv)
    team = _first_col(df, ["team_id","team","team_code","team_name","team_id_int"])
    if not team:
        return None
    FGA = _first_col(df, ["sFieldGoalsAttempted","FGA","fga","field_goals_attempted"])
    FTA = _first_col(df, ["sFreeThrowsAttempted","FTA","fta","free_throws_attempted"])
    OREB= _first_col(df, ["sReboundsOffensive","OREB","orb","o_rebounds","off_reb"])
    TOV = _first_col(df, ["sTurnovers","TOV","to","turnovers"])
    PTS = _first_col(df, ["sPoints","PTS","pts","points"])
    name= _first_col(df, ["team_name","team","team_code","team_display"])

    need = [FGA, FTA, OREB, TOV, PTS]
    if any(x is None for x in need):
        return None

    g = df.groupby(team, dropna=False)[[FGA,FTA,OREB,TOV,PTS]].sum().reset_index()
    if name and name != team and name in df.columns:
        name_map = df[[team, name]].drop_duplicates().set_index(team)[name].to_dict()
        g["team_label"] = g[team].map(name_map)
    else:
        g["team_label"] = g[team].astype(str)
    g = g.rename(columns={FGA:"FGA", FTA:"FTA", OREB:"OREB", TOV:"TOV", PTS:"PTS", team:"team"})
    return g[["team","team_label","FGA","FTA","OREB","TOV","PTS"]]

def _aggregate_box_from_raw_json(raw_json: Path) -> Optional[pd.DataFrame]:
    if not raw_json.exists(): return None
    try:
        data = json.loads(raw_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "tm" not in data: return None
    rows = []
    for k, t in data["tm"].items():
        try:
            rows.append({
                "team": str(k),
                "team_label": t.get("name") or t.get("code") or str(k),
                "FGA": int(t.get("tot_sFieldGoalsAttempted", 0)),
                "FTA": int(t.get("tot_sFreeThrowsAttempted", 0)),
                "OREB": int(t.get("tot_sReboundsOffensive", 0)),
                "TOV": int(t.get("tot_sTurnovers", 0)),
                "PTS": int(t.get("tot_sPoints", 0)),
            })
        except Exception:
            continue
    if not rows: return None
    return pd.DataFrame(rows)

def _poss_from_box_row(row: pd.Series, ft_weight: float) -> float:
    return float(row["FGA"] - row["OREB"] + row["TOV"] + ft_weight * row["FTA"])

def _ratings_from_box(box_df: pd.DataFrame, cfg: PossessionConfig) -> pd.DataFrame:
    if box_df is None or len(box_df) < 2:
        return pd.DataFrame()
    teams = list(box_df["team"].astype(str).unique())
    a, b = teams[0], teams[1]
    A = box_df.loc[box_df["team"].astype(str) == a].iloc[0]
    B = box_df.loc[box_df["team"].astype(str) == b].iloc[0]

    poss_a = _poss_from_box_row(A, cfg.ft_weight)
    poss_b = _poss_from_box_row(B, cfg.ft_weight)
    poss_avg = 0.5 * (poss_a + poss_b)
    poss_harm = 2.0 * poss_a * poss_b / (poss_a + poss_b) if (poss_a + poss_b) else float("nan")

    def pack(me, opp, poss):
        if poss <= 0 or math.isnan(poss):
            return float("nan"), float("nan"), float("nan")
        ortg = 100.0 * me / poss
        drtg = 100.0 * opp / poss
        return ortg, drtg, ortg - drtg

    a_avg = pack(A["PTS"], B["PTS"], poss_a)
    b_avg = pack(B["PTS"], A["PTS"], poss_b)

    out = pd.DataFrame([
        {"team": str(a), "team_label": A["team_label"], "PTS": A["PTS"],
         "poss_team_est": poss_a, "poss_team_est_opp": poss_b,
         "poss_game_avg": poss_avg, "poss_game_harm": poss_harm,
         "ORtg": a_avg[0], "DRtg": a_avg[1], "Net": a_avg[2]},
        {"team": str(b), "team_label": B["team_label"], "PTS": B["PTS"],
         "poss_team_est": poss_b, "poss_team_est_opp": poss_a,
         "poss_game_avg": poss_avg, "poss_game_harm": poss_harm,
         "ORtg": b_avg[0], "DRtg": b_avg[1], "Net": b_avg[2]},
    ])
    return out

# ---------------------- Exact possessions (pbp-based) ------------------------
def _load_pbp(game_dir: Path) -> Optional[pd.DataFrame]:
    p = game_dir / "game_pbp_normalized.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None

def _infer_flags(row: pd.Series) -> Tuple[Optional[str], Optional[str], Optional[bool], Optional[str]]:
    etype = None
    for c in ["etype","event_type","type","actionType","action","EventType","event"]:
        if c in row and pd.notna(row[c]):
            etype = str(row[c]).strip().lower()
            break
    team = None
    for c in ["team_id","team","team_code","Team","tno","teamId"]:
        if c in row and pd.notna(row[c]):
            team = str(row[c]).strip()
            break
    made = None
    for c in ["made","is_made","shot_made","made_flag","result","r","isMade"]:
        if c in row:
            m = _booly(row[c])
            if m is not None:
                made = m
                break
    rtype = None
    for c in ["rebound_type","rb_type","reboundTeam","rebound_team_type"]:
        if c in row and pd.notna(row[c]):
            rtype = str(row[c]).strip().lower()
            break
    if rtype is None:
        for c in ["detail","description","text","play","msg"]:
            if c in row and pd.notna(row[c]):
                s = str(row[c]).lower()
                if "offensive rebound" in s:
                    rtype = "offensive"
                elif "defensive rebound" in s:
                    rtype = "defensive"
                break
    return etype, team, made, rtype

def _sorted_pbp(df: pd.DataFrame) -> pd.DataFrame:
    per = _first_col(df, ["period","per","qtr","quarter"])
    clk = _first_col(df, ["gt","clock","time","game_clock"])
    idx = _first_col(df, ["sequence_no","event_number","event_no","row_id","idx"])
    if per and clk:
        def _sec(x):
            try:
                m, s = str(x).split(":")
                return int(m)*60 + int(s)
            except Exception:
                return 0
        df = df.copy()
        df["_sec"] = df[clk].map(_sec)
        df = df.sort_values([per, "_sec"], ascending=[True, False], kind="mergesort")
        df = df.drop(columns=["_sec"])
        return df
    if idx:
        return df.sort_values(idx, kind="mergesort")
    return df

def exact_possessions_from_pbp(game_dir: Path, cfg: PossessionConfig) -> Optional[Dict[str, int]]:
    df = _load_pbp(game_dir)
    if df is None or df.empty:
        return None
    df = _sorted_pbp(df)
    poss: Dict[str, int] = {}
    cur_team: Optional[str] = None
    per_col = _first_col(df, ["period","per","qtr","quarter"])
    for _, row in df.iterrows():
        etype, team, made, rtype = _infer_flags(row)
        if cur_team is None and team:
            cur_team = team
            poss.setdefault(cur_team, 0)
        if etype and "turnover" in etype:
            if cur_team is None and team:
                cur_team = team
                poss.setdefault(cur_team, 0)
            if cur_team is not None:
                poss[cur_team] = poss.get(cur_team, 0) + 1
            cur_team = team if team and team != cur_team else None
            continue
        if etype and ("2pt" in etype or "3pt" in etype or "shot" in etype):
            if made is True:
                if cur_team is None and team:
                    cur_team = team
                    poss.setdefault(cur_team, 0)
                if cur_team is not None:
                    poss[cur_team] = poss.get(cur_team, 0) + 1
                cur_team = None
            continue
        if etype and "rebound" in etype:
            if rtype is None and cfg.treat_treb_as_defensive:
                rtype = "defensive"
            if rtype == "defensive":
                if cur_team is not None:
                    poss[cur_team] = poss.get(cur_team, 0) + 1
                cur_team = None
            elif rtype == "offensive":
                if cur_team is None and team:
                    cur_team = team
                    poss.setdefault(cur_team, 0)
            continue
        if etype and ("free" in etype or "ft" in etype):
            txt = ""
            for c in ["detail","description","text","play","msg","subType"]:
                if c in df.columns and pd.notna(row.get(c)):
                    txt = str(row[c]).lower()
                    break
            last_ft = any(k in txt for k in ["2 of 2","3 of 3","last","final"])
            if last_ft and ("miss" not in txt):
                if cur_team is not None:
                    poss[cur_team] = poss.get(cur_team, 0) + 1
                cur_team = None
            continue
        if per_col and etype and ("end of period" in etype or "period end" in etype):
            if cur_team is not None:
                poss[cur_team] = poss.get(cur_team, 0) + 1
            cur_team = None
    return poss if poss else None

# -------------------- Public entry point for one game ------------------------
def compute_possessions_and_ratings(
    out_root: Path, week: str, game_id: str,
    src_json: Optional[Path] = None,
    cfg: PossessionConfig = PossessionConfig()
) -> Tuple[str, pd.DataFrame]:
    game_dir = out_root / week / game_id
    lines: List[str] = []

    poss_exact = exact_possessions_from_pbp(game_dir, cfg)

    box_df = _aggregate_box_from_players_csv(game_dir / "game_players_with_positions.csv")
    if box_df is None and src_json is not None:
        box_df = _aggregate_box_from_raw_json(src_json)
    if box_df is None:
        return ("[POSSESSIONS ERROR] Could not derive team box totals from players CSV or JSON.", pd.DataFrame())

    ratings_df = _ratings_from_box(box_df, cfg)
    if ratings_df.empty:
        return ("[POSSESSIONS ERROR] Box totals aggregated, but ratings could not be computed.", pd.DataFrame())

    def fmt(x):
        return "—" if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else (f"{x:.2f}" if isinstance(x, float) else str(x))

    if poss_exact:
        total_exact = sum(poss_exact.values())
        lines.append(f"[EXACT] possessions by team: { {k: poss_exact[k] for k in poss_exact} } (total ~ {total_exact})")
    else:
        lines.append("[EXACT] not available (schema insufficient); reported net estimates below.")

    for _, r in ratings_df.iterrows():
        tlabel = str(r["team_label"])
        poss_team_est = r["poss_team_est"]
        poss_game_avg = r["poss_game_avg"]
        poss_game_harm = r["poss_game_harm"]
        ortg, drtg, net = r["ORtg"], r["DRtg"], r["Net"]

        if poss_exact:
            exact = None
            k1 = tlabel
            k2 = str(r["team"])
            if k1 in poss_exact: exact = poss_exact[k1]
            elif k2 in poss_exact: exact = poss_exact[k2]
            lines.append(
                f"{tlabel}: ORtg {fmt(ortg)}  DRtg {fmt(drtg)}  Net {fmt(net)}  | poss_exact {exact}  | poss_est(team) {fmt(poss_team_est)}  game_avg {fmt(poss_game_avg)}  harm {fmt(poss_game_harm)}"
            )
        else:
            lines.append(
                f"{tlabel}: ORtg {fmt(ortg)}  DRtg {fmt(drtg)}  Net {fmt(net)}  | poss_est(team) {fmt(poss_team_est)}  game_avg {fmt(poss_game_avg)}  harm {fmt(poss_game_harm)}"
            )

    return ("\n".join(lines), ratings_df)
