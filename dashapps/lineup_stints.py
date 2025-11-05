# lineup_stints.py
from __future__ import annotations

import os
import re
import ast
import json
import hashlib
from pathlib import Path
from typing import List, Any, Tuple, Iterable, Dict, Optional

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State
from dash.dash_table import DataTable
from dash.dash_table.Format import Format
import plotly.graph_objects as go

# ============================================================
# CONFIG (robust data-root discovery)
# ============================================================

STINTS_PM_FILE = "game_stints_pm.csv"
POS_LOOKUP_FILE = "game_player_positions_lookup.csv"

APP_TITLE = "Lineup Stints"
APP_BASE_PATH = "/lineup_stints/"

PERIOD_SECONDS = int(os.environ.get("PERIOD_SECONDS", 600))
OVERTIME_SECONDS = int(os.environ.get("OVERTIME_SECONDS", 300))

# You can override these with DATA_ROOT or DEFAULT_WEEK env vars at runtime
PREFERRED_DATA_ROOTS = [
    Path("/Users/uygarkaraca/JSON/BSL_2025_26/out_data"),  # explicit absolute
    Path.home() / "JSON/BSL_2025_26/out_data",             # home-relative
]

def _looks_like_out_data(p: Path) -> bool:
    """A directory that has week* subfolders (ignore _cumulative)."""
    try:
        if not p.exists():
            return False
        for d in p.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name.lower().startswith("week"):
                return True
        return False
    except Exception:
        return False

def _find_out_data_root() -> Path:
    # 1) Explicit env overrides – always win if valid
    for key in ("DATA_ROOT", "DATA_DIR"):
        v = os.environ.get(key)
        if v:
            p = Path(v).expanduser().resolve()
            if _looks_like_out_data(p):
                return p

    # 2) Strong preferences – your real tree first
    for p in PREFERRED_DATA_ROOTS:
        p = p.expanduser().resolve()
        if _looks_like_out_data(p):
            return p

    # 3) Nearby fallbacks (project-local, cwd, /mnt/data, etc.)
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "out_data",
        here / "out_data",
        here.parent / "out_data",
        Path("/mnt/data/out_data"),
        Path("/mnt/data"),
    ]
    # Also try any parent’s sibling "out_data"
    for base in [here] + list(here.parents):
        candidates.append(base / "out_data")

    seen, ordered = set(), []
    for c in candidates:
        c = c.resolve()
        if str(c) not in seen:
            seen.add(str(c))
            ordered.append(c)

    for c in ordered:
        if _looks_like_out_data(c):
            return c

    # 4) Last resort
    return (Path.cwd() / "out_data").resolve()

DATA_ROOT = _find_out_data_root()


# --- sanity check that chosen game dir actually has the files we will read ---
def _assert_game_dir_has_required_files(game_dir: Path) -> None:
    st_p = game_dir / STINTS_PM_FILE
    pos_p = game_dir / POS_LOOKUP_FILE
    if not st_p.exists():
        raise FileNotFoundError(f"{STINTS_PM_FILE} missing in {game_dir}")
    if not pos_p.exists():
        # soft warning: we can still render with names = tokens
        print(f"[lineup_stints][WARN] {POS_LOOKUP_FILE} missing in {game_dir}. "
              f"Proceeding without names/positions.")

# ============================================================
# DISCOVERY: weeks → games (deterministic, natural sort)
# ============================================================

def _is_game_dir(p: Path) -> bool:
    return p.is_dir() and (p / STINTS_PM_FILE).exists()

def _label_from_meta(game_dir: Path) -> str:
    meta = game_dir / "meta.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            return m.get("label") or ""
        except Exception:
            pass
    # fallback: read teams from stints file
    try:
        st = pd.read_csv(game_dir / STINTS_PM_FILE, usecols=["team_name"])
        teams = st["team_name"].dropna().unique().tolist()
        if len(teams) >= 2:
            return f"{teams[0]} vs {teams[1]}"
        return ", ".join(teams) if teams else game_dir.name
    except Exception:
        return game_dir.name

def _is_week_dir(p: Path) -> bool:
    # Accept only weekXX style folders (ignore _cumulative, accidental dirs)
    return p.is_dir() and re.match(r"^week\d+$", p.name, flags=re.I) is not None

def _week_num(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1

def _sorted_weeks(week_dirs: List[Path]) -> List[Path]:
    # Natural sort: week2 < week10
    return sorted(week_dirs, key=lambda p: (_week_num(p.name), p.name.lower()))

def _sorted_games(game_dirs: List[Path]) -> List[Path]:
    # Sort game folders numerically when possible (2714805 < 2736922), fallback lexicographic
    def _num(p: Path):
        try:
            return int(p.name)
        except Exception:
            return float("inf")
    return sorted(game_dirs, key=lambda p: (_num(p), p.name.lower()))

def discover_weeks(root: Path) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    if not root.exists():
        print(f"[lineup_stints] DATA_ROOT does not exist: {root}")
        return out
    
    week_dirs = _sorted_weeks([p for p in root.iterdir() if _is_week_dir(p)])
    if not week_dirs:
        print(f"[lineup_stints] No week directories found under {root}")
        return out

    for week_dir in week_dirs:
        games: List[Dict[str, str]] = []
        game_dirs = _sorted_games([p for p in week_dir.iterdir() if p.is_dir()])
        for game_dir in game_dirs:
            if _is_game_dir(game_dir):
                games.append({
                    "game_id": game_dir.name,
                    "label": _label_from_meta(game_dir),
                    "path": str(game_dir.resolve()),
                })
        if games:
            out[week_dir.name] = games

    # Debug: print a compact catalog summary
    summary = ", ".join([f"{wk}:{len(gs)}" for wk, gs in out.items()])
    print(f"[lineup_stints] Catalog weeks (count per week) → {summary}")
    return out

# --- Now safe to call discover_weeks() ---
try:
    _cat_probe = {}
    if DATA_ROOT.exists():
        _cat_probe = {wk: len(v) for wk, v in discover_weeks(DATA_ROOT).items()}
    print("[lineup_stints] Weeks discovered at startup →",
          ", ".join(f"{k}:{v}" for k, v in sorted(_cat_probe.items())))
except Exception as _e:
    print("[lineup_stints] Discovery probe failed:", _e)

# ============================================================
# Utility: parsing, timing, labels
# ============================================================

def _parse_players_list(cell: Any) -> List[int]:
    # legacy int list (kept for reference/sanity)
    if isinstance(cell, list):
        return [int(x) for x in cell]
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    try:
        if s.startswith("["):
            return [int(x) for x in ast.literal_eval(s)]
        parts = [t for t in s.replace(" ", "").split(",") if t]
        return [int(x) for x in parts]
    except Exception:
        return []

# --- NEW: parse "team_no:player_no" token lists (pair-safe) ---
def _parse_tokens_list(cell: Any) -> List[str]:
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    try:
        if s.startswith("["):
            vals = ast.literal_eval(s)
            return [str(x) for x in vals]
        parts = [t for t in s.replace(" ", "").split(",") if t]
        return [str(x) for x in parts]
    except Exception:
        return []

def _period_length_seconds(period: int) -> int:
    return PERIOD_SECONDS if int(period) <= 4 else OVERTIME_SECONDS

def _clock_to_secs(clock_str: str) -> int:
    if not isinstance(clock_str, str) or ":" not in clock_str:
        return 0
    m, s = clock_str.split(":")[:2]
    return int(m) * 60 + int(s)

def _abs_seconds(period: int, clock: str) -> int:
    prev = 0
    for p in range(1, int(period)):
        prev += _period_length_seconds(p)
    rem = _clock_to_secs(clock)
    length = _period_length_seconds(int(period))
    return prev + (length - rem)

def _fmt_mmss(total_seconds: float) -> str:
    sec = max(0, int(round(total_seconds)))
    return f"{sec // 60:02d}:{sec % 60:02d}"

def _isna(x) -> bool:
    try:
        res = pd.isna(x)
        return bool(res.all()) if hasattr(res, "all") else bool(res)
    except Exception:
        return x is None

def _clean_strish(x: Any) -> str:
    if isinstance(x, (pd.Series, list, tuple)):
        for v in (x if not isinstance(x, pd.Series) else x.tolist()):
            if not _isna(v):
                s = str(v).strip()
                return "" if s.lower() in {"","nan","none","na","<na>"} else s
        return ""
    if _isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"","nan","none","na","<na>"} else s

def _abbr_pos_group(val: Any) -> str:
    if _isna(val): return ""
    s = str(val).strip()
    if not s: return ""
    return {"Guard":"G","Forward":"F","Center":"C","Guard-Forward":"G/F","Forward-Center":"F/C"}.get(s, s[:1].upper())

# --- OLD int-keyed helpers (kept for completeness, not used for labels now) ---
def _lineup_key(players: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in players))

# --- NEW: pair-aware lineup key & labels ---
def _lineup_key_tokens(tokens: Iterable[str]) -> Tuple[str, ...]:
    # order-independent, pair-safe
    return tuple(sorted(str(t) for t in tokens))

def _players_to_labels_tokens(tokens: Iterable[str], pos_lookup: pd.DataFrame) -> List[str]:
    if pos_lookup.empty:
        return [t for t in tokens]
    m = pos_lookup.copy()
    if "pair_token" not in m.columns:
        # best-effort fallback
        if {"team_no","player_no"}.issubset(m.columns):
            m["pair_token"] = m.apply(lambda r: f"{int(r['team_no'])}:{int(r['player_no'])}", axis=1)
        else:
            return [t for t in tokens]
    m = m.dropna(subset=["pair_token"])
    m = m[~m["pair_token"].duplicated(keep="first")].set_index("pair_token", drop=False)

    out: List[str] = []
    for tok in tokens:
        if tok in m.index:
            row = m.loc[tok]
            nm = _clean_strish(row.get("name")) or f"#{_clean_strish(row.get('shirtNumber'))}" or tok
            pg_abbr = _abbr_pos_group(row.get("position_group"))
            out.append(nm if not pg_abbr else f"{nm} ({pg_abbr})")
        else:
            out.append(tok)
    return out

def _pos_mix_from_tokens(tokens: Iterable[str], pos_lookup: pd.DataFrame) -> str:
    if pos_lookup.empty:
        return ""
    m = pos_lookup.copy()
    if "pair_token" not in m.columns:
        return ""
    m = m.dropna(subset=["pair_token"])
    m = m[~m["pair_token"].duplicated(keep="first")].set_index("pair_token", drop=False)
    groups = []
    for tok in tokens:
        val = m.at[tok, "position_group"] if tok in m.index else ""
        groups.append(str(val) if pd.notna(val) else "")
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

def _color_from_key(key: Tuple[Any, ...]) -> str:
    h = int(hashlib.md5(",".join(map(str, key)).encode("utf-8")).hexdigest(), 16) % 360
    return f"hsl({h}, 65%, 45%)"

def _lookup_label_for_action(st_df: pd.DataFrame, action: float) -> str:
    hit = st_df.loc[(st_df["start_action"] == action)]
    if not hit.empty:
        r = hit.iloc[0]; return f"P{int(r['start_period'])} {r['start_clock']}"
    hit = st_df.loc[(st_df["end_action"] == action)]
    if not hit.empty:
        r = hit.iloc[0]; return f"P{int(r['end_period'])} {r['end_clock']}"
    return f"act {int(action) if pd.notna(action) else ''}"

def _active_lineup_at(team_df: pd.DataFrame, action: float):
    row = team_df[(team_df["start_action"] <= action) & (team_df["end_action"] > action)]
    if row.empty:
        row = team_df[(team_df["start_action"] <= action) & (team_df["end_action"] >= action)]
    return row.iloc[0] if not row.empty else None

def _abs_score_at_boundary(st_df: pd.DataFrame, action: float, t1: int, t2: int) -> Optional[tuple[int, int]]:
    """
    Return absolute game score at boundary `action` as a pair (score_t1, score_t2),
    where t1 and t2 are team_nos in a fixed order. We read whichever stint row
    starts/ends at `action` and remap (for,against) into (t1,t2).
    """
    rows = st_df[(st_df["start_action"] == action) | (st_df["end_action"] == action)]
    if rows.empty:
        return None

    r = rows.loc[rows["team_no"].astype("Int64") == t1]
    r = (r.iloc[0] if not r.empty else rows.iloc[0])

    if float(r["start_action"]) == float(action):
        sf, sa = r["score_start_for"], r["score_start_against"]
    else:
        sf, sa = r["score_end_for"], r["score_end_against"]

    r_team = int(r["team_no"])
    if r_team == t1:
        return int(sf), int(sa)
    elif r_team == t2:
        return int(sa), int(sf)
    else:
        return None

# ============================================================
# Load stints/lookup (PAIR-AWARE)
# ============================================================

def _load_data_from_game_dir(game_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[lineup_stints] LOAD game_dir={game_dir}")
    _assert_game_dir_has_required_files(game_dir)
    st = pd.read_csv(game_dir / STINTS_PM_FILE)
    for col in [
        "start_action","end_action","team_no","lineup_size","net","poss_for","poss_against",
        "pts_for","pts_against","ORtg","DRtg","score_start_for","score_start_against",
        "score_end_for","score_end_against","start_period","end_period"
    ]:
        if col in st.columns:
            st[col] = pd.to_numeric(st[col], errors="coerce")

    # legacy ints (still useful for sanity)
    st["players_on_court"] = st["players_on_court"].apply(_parse_players_list)

    # NEW: pair-aware tokens from extractor, or synthesize from team_no
    if "players_on_court_tokens" in st.columns:
        st["lineup_tokens"] = st["players_on_court_tokens"].apply(_parse_tokens_list)
    else:
        st["lineup_tokens"] = st.apply(
            lambda r: [f"{int(r['team_no'])}:{int(p)}" for p in r["players_on_court"]],
            axis=1
        )

    # Optional: names list (if extractor wrote it)
    if "players_on_court_names" in st.columns:
        st["players_on_court_names"] = st["players_on_court_names"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith("[") else (x if isinstance(x, list) else [])
        )

    st["score_start"] = st["score_start_for"].astype(int).astype(str) + "–" + st["score_start_against"].astype(int).astype(str)
    st["score_end"]   = st["score_end"] = st["score_end_for"].astype(int).astype(str) + "–" + st["score_end_against"].astype(int).astype(str)
    st["margin_delta"] = (st["score_end_for"] - st["score_end_against"]) - (st["score_start_for"] - st["score_start_against"])
    st["start_abs_sec"] = [_abs_seconds(p, c) for p, c in zip(st["start_period"], st["start_clock"])]
    st["end_abs_sec"]   = [_abs_seconds(p, c) for p, c in zip(st["end_period"],   st["end_clock"])]
    st["stint_seconds"] = (st["end_abs_sec"] - st["start_abs_sec"]).clip(lower=0)
    st["stint_id"] = np.arange(1, len(st) + 1)

    pos = pd.DataFrame()
    pl = game_dir / POS_LOOKUP_FILE
    if pl.exists():
        pos = pd.read_csv(pl)
        keep = ["player_no","name","shirtNumber","position_group","team_no","team_name","pair_token"]  # include pair_token
        for k in keep:
            if k not in pos.columns:
                pos[k] = None
        pos = pos[keep].copy()
        # build pair_token if missing
        if "pair_token" not in pos.columns or pos["pair_token"].isna().all():
            pos["pair_token"] = pos.apply(
                lambda r: f"{int(r['team_no'])}:{int(r['player_no'])}" if pd.notna(r["team_no"]) and pd.notna(r["player_no"]) else None,
                axis=1
            )
        pos["player_no"] = pd.to_numeric(pos["player_no"], errors="coerce").astype("Int64")
        pos["shirtNumber"] = pos["shirtNumber"].astype(str)

        # --- diagnostics ---
    print(f"[lineup_stints]  stints rows={len(st)} cols={list(st.columns)}")
    if (game_dir / POS_LOOKUP_FILE).exists():
        print(f"[lineup_stints]  pos-lookup rows={len(pos)} cols={list(pos.columns)}")
    else:
        print(f"[lineup_stints]  pos-lookup not found (soft-fallback)")
    return st, pos

# ============================================================
# Sequence builder (absolute deltas; token-aware)
# ============================================================

def _build_sequence_rows_general(st_df: pd.DataFrame, pos_df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    if st_df.empty:
        return [], []

    # Stable team order by team_no
    team_meta = st_df.dropna(subset=["team_no", "team_name"]).drop_duplicates(subset=["team_no", "team_name"])
    tnos = sorted(team_meta["team_no"].astype(int).unique().tolist())
    if len(tnos) != 2:
        return [], []
    t1, t2 = int(tnos[0]), int(tnos[1])

    name_to_no = {row.team_name: int(row.team_no) for row in team_meta.itertuples(index=False)}
    no_to_name = {v: k for k, v in name_to_no.items()}
    teams = [no_to_name[t] for t in tnos]

    # Union cuts across teams
    cuts_set = set()
    for t in teams:
        tdf = st_df[st_df["team_name"] == t]
        cuts_set |= set(tdf["start_action"].dropna().astype(float)) | set(tdf["end_action"].dropna().astype(float))
    cuts = sorted(float(c) for c in cuts_set if pd.notna(c))
    if len(cuts) < 2:
        return [], []

    seq_rows: list[dict] = []
    traces: list[dict] = []

    team_dfs: Dict[str, pd.DataFrame] = {t: st_df[st_df["team_name"] == t].sort_values(["start_action", "end_action"]) for t in teams}

    # Table rows
    for i in range(len(cuts) - 1):
        a0, a1 = cuts[i], cuts[i + 1]
        if not (a1 > a0):
            continue
        row = {
            "segment": f"{int(a0)}→{int(a1)}",
            "start": _lookup_label_for_action(st_df, a0),
            "end": _lookup_label_for_action(st_df, a1),
            "_a0": a0, "_a1": a1
        }
        any_lineup = False
        for t in teams:
            r = _active_lineup_at(team_dfs[t], a0)
            if r is not None:
                any_lineup = True
                row[f"{t} lineup"] = ", ".join(_players_to_labels_tokens(r["lineup_tokens"], pos_df))
        if any_lineup:
            seq_rows.append(row)

    # Bars with absolute nets
    for t in teams:
        tdf = team_dfs[t]
        x, base, text, colors, hover = [], [], [], [], []
        t_no = name_to_no[t]

        for i in range(len(cuts) - 1):
            a0, a1 = cuts[i], cuts[i + 1]
            if not (a1 > a0):
                continue

            r = _active_lineup_at(tdf, a0)
            if r is None:
                continue

            width = a1 - a0
            key = _lineup_key_tokens(r["lineup_tokens"])
            col = _color_from_key(key)

            s0 = _abs_score_at_boundary(st_df, a0, t1, t2)
            s1 = _abs_score_at_boundary(st_df, a1, t1, t2)

            if s0 is None or s1 is None:
                seg_text = ""
                seg_hover_delta = "ΔPts: — | Net: —"
                s0_label = s1_label = "—"
            else:
                (t1_s0, t2_s0) = s0
                (t1_s1, t2_s1) = s1
                d1, d2 = (t1_s1 - t1_s0), (t2_s1 - t2_s0)
                net_t1 = d1 - d2
                net_t2 = -net_t1

                if t_no == t1:
                    net_me, d_for, d_against = net_t1, d1, d2
                else:
                    net_me, d_for, d_against = net_t2, d2, d1

                seg_text = f"{net_me:+d}"
                s0_label = f"{t1_s0}–{t2_s0}"
                s1_label = f"{t1_s1}–{t2_s1}"
                seg_hover_delta = f"ΔPts: {d_for}-{d_against} | Net: {net_me:+d}"

            base.append(a0)
            x.append(width)
            colors.append(col)
            text.append(seg_text)

            ltxt = ", ".join(_players_to_labels_tokens(r["lineup_tokens"], pos_df))
            hover.append(
                f"<b>{t}</b><br>"
                f"{_lookup_label_for_action(st_df, a0)} → {_lookup_label_for_action(st_df, a1)} "
                f"(acts {int(a0)}→{int(a1)})<br>"
                f"Lineup: {ltxt}<br>"
                f"S0: {s0_label} → S1: {s1_label}<br>"
                f"{seg_hover_delta}"
            )

        traces.append(dict(
            type="bar",
            x=x, y=[t] * len(x), base=base, orientation="h",
            name=t,
            text=text, textposition="inside", insidetextanchor="middle",
            textfont=dict(size=11),
            marker=dict(color=colors, line=dict(width=0)),
            hoverinfo="text", hovertext=hover,
            opacity=0.92, cliponaxis=False,
        ))

    return seq_rows, traces

def _encode_lineup_value(team: str, key: Tuple[str, ...]) -> str:
    return f"{team}|{','.join(str(x) for x in key)}"

def _decode_lineup_value(val: Optional[str]) -> Optional[Tuple[str, Tuple[str, ...]]]:
    if not val or "|" not in val: return None
    team, rest = val.split("|", 1)
    key = tuple(sorted(x for x in rest.split(",") if x))
    return team, key

def _labels_for_key(key: Tuple[str, ...], pos_lookup: pd.DataFrame) -> List[str]:
    return _players_to_labels_tokens(key, pos_lookup)

def _timeline_figure_line_rows(df: pd.DataFrame, pos_lookup: pd.DataFrame,
                               cmpA_val: Optional[str] = None, cmpB_val: Optional[str] = None) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=320, margin=dict(l=20, r=10, t=30, b=35))
        return fig
    cmpA = _decode_lineup_value(cmpA_val)
    cmpB = _decode_lineup_value(cmpB_val)
    fig = go.Figure()
    df = df.copy()
    if "lineup_key" not in df.columns:
        df["lineup_key"] = df["lineup_tokens"].apply(_lineup_key_tokens)
    df["lineup_color"] = df["lineup_key"].apply(_color_from_key)

    delta_plus_names, delta_minus_names = [], []
    if cmpA and cmpB:
        A_team, A_key = cmpA
        B_team, B_key = cmpB
        delta_plus_names = _labels_for_key(tuple(sorted(set(A_key) - set(B_key))), pos_lookup)
        delta_minus_names = _labels_for_key(tuple(sorted(set(B_key) - set(A_key))), pos_lookup)

    teams = df["team_name"].dropna().unique().tolist()
    for t in teams:
        seg = df[df["team_name"] == t]
        y = [f"{t} | {', '.join(_players_to_labels_tokens(k, pos_lookup))}" for k in seg["lineup_key"]]
        base = seg["start_action"].astype(float)
        width = (seg["end_action"] - seg["start_action"]).astype(float)
        colors = seg["lineup_color"].tolist()
        texts = [f"{int(n):+d}" if pd.notna(n) else "" for n in seg["net"]]

        key_series = seg["lineup_key"].tolist()
        line_widths, line_colors, hovers = [], [], []
        for k, sp, sc, ep, ec, ss, se, pf, pa, ptf, pta, nt, or_, dr_, color in zip(
            key_series, seg["start_period"], seg["start_clock"], seg["end_period"], seg["end_clock"],
            seg["score_start"], seg["score_end"], seg["poss_for"], seg["poss_against"], seg["pts_for"], seg["pts_against"],
            seg["net"], seg["ORtg"].fillna(np.nan), seg["DRtg"].fillna(np.nan), colors,
        ):
            isA = cmpA and (t == cmpA[0]) and (k == cmpA[1])
            isB = cmpB and (t == cmpB[0]) and (k == cmpB[1])
            lw = 2 if (isA or isB) else 0
            lc = "#111" if (isA or isB) else "rgba(0,0,0,0)"
            line_widths.append(lw); line_colors.append(lc)
            delta_line = ""
            if (isA or isB) and cmpA and cmpB:
                plus_txt = ", ".join(delta_plus_names) if delta_plus_names else "—"
                minus_txt = ", ".join(delta_minus_names) if delta_minus_names else "—"
                delta_line = f"<br><b>Δ vs other</b>: <span style='color:#0a0'>+{plus_txt}</span> | <span style='color:#a00'>−{minus_txt}</span>"
            hovers.append(
                f"<b>{t}</b><br>"
                f"Start: P{int(sp)} {sc}<br>"
                f"End:   P{int(ep)} {ec}<br>"
                f"Score: {ss} → {se}<br>"
                f"Poss: {int(pf)}/{int(pa)} | Pts: {int(ptf)}-{int(pta)}<br>"
                f"Net: {int(nt) if pd.notna(nt) else 'NA'}" + (f" | ORtg {or_:.1f} | DRtg {dr_:.1f}" if pd.notna(or_) and pd.notna(dr_) else "") + delta_line
            )

        fig.add_bar(
            x=width, y=y, base=base, orientation="h", text=texts, textposition="inside", insidetextanchor="middle",
            textfont=dict(size=11), marker=dict(color=colors, line=dict(width=line_widths, color=line_colors)),
            hoverinfo="text", hovertext=hovers, name=t,
        )

    fig.update_layout(
        barmode="stack", xaxis_title="Action Number (start→end)", yaxis_title="Team | Lineup (order-independent)",
        height=max(460, 28 * len(df)), margin=dict(l=10, r=10, t=40, b=30), legend_title_text="Team", hovermode="closest", bargap=0.15,
    )
    fig.update_yaxes(categoryorder="category ascending")
    return fig

def _timeline_figure_sequence(st_df: pd.DataFrame, pos_df: pd.DataFrame) -> go.Figure:
    rows, traces = _build_sequence_rows_general(st_df, pos_df)
    fig = go.Figure(data=traces)
    lanes = max(1, st_df["team_name"].nunique())
    fig.update_layout(
        barmode="overlay", xaxis_title="Action Number (game sequence)", yaxis_title="Team",
        height=240 + 60 * lanes, margin=dict(l=10, r=10, t=40, b=30), legend_title_text="Team", hovermode="closest", bargap=0.05,
    )
    return fig

# ============================================================
# APP FACTORY
# ============================================================

def create_dash_lineup_stints(server=None, base_pathname: str = APP_BASE_PATH):
    DEFAULT_WEEK = os.environ.get("DEFAULT_WEEK", "latest")  # 'latest', 'earliest', or 'week05'

    catalog = discover_weeks(DATA_ROOT)
    week_names = sorted(catalog.keys(), key=lambda n: _week_num(n))

    if DEFAULT_WEEK.lower() == "latest" and week_names:
        initial_week = week_names[-1]
    elif DEFAULT_WEEK.lower() == "earliest" and week_names:
        initial_week = week_names[0]
    elif DEFAULT_WEEK in catalog:
        initial_week = DEFAULT_WEEK
    else:
        initial_week = week_names[0] if week_names else None

    week_options = [{"label": wk, "value": wk} for wk in week_names]
    game_options = [{"label": f"{g['label']} ({g['game_id']})", "value": g["path"]} for g in catalog.get(initial_week, [])]
    initial_game = game_options[0]["value"] if game_options else None

    print(f"[lineup_stints] Initial week → {initial_week} | initial game → {initial_game}")

    app = dash.Dash(
        __name__, server=server, url_base_pathname=base_pathname,
        suppress_callback_exceptions=True, title=APP_TITLE,
    )

    app.layout = html.Div(
        className="p-4",
        children=[
            html.H2("Lineup Stints", className="mb-2"),

            html.Div(className="grid grid-cols-1 md:grid-cols-6 gap-3", children=[
                html.Div(children=[
                    html.Label("Week"),
                    dcc.Dropdown(id="ls-week", options=week_options, value=initial_week, placeholder="Select week"),
                ]),
                html.Div(className="md:col-span-2", children=[
                    html.Label("Game"),
                    dcc.Dropdown(id="ls-game", options=game_options, value=initial_game, placeholder="Select game"),
                ]),
                html.Div(children=[
                    html.Label("Filter periods"),
                    dcc.Dropdown(id="ls-periods", options=[], value=None, multi=True, placeholder="All"),
                ]),
                html.Div(children=[html.Label("Min ORtg"), dcc.Slider(id="ls-min-ortg", min=0, max=180, step=1, value=0, tooltip={"placement": "bottom"})]),
                html.Div(children=[html.Label("Max DRtg"), dcc.Slider(id="ls-max-drtg", min=50, max=200, step=1, value=200, tooltip={"placement": "bottom"})]),
                html.Div(children=[html.Label("Min Net"),  dcc.Slider(id="ls-min-net",  min=-50, max=50, step=1, value=-50, tooltip={"placement": "bottom"})]),
            ]),

            html.Div(className="grid grid-cols-1 md:grid-cols-6 gap-3 mt-2", children=[
                html.Div(children=[
                    html.Label("View"),
                    dcc.RadioItems(
                        id="ls-view",
                        options=[{"label":"Sequence (2 rows)","value":"sequence"},{"label":"Line-by-line","value":"line_rows"}],
                        value="sequence", labelStyle={"display":"inline-block","marginRight":"16px"},
                    ),
                ]),
                html.Div(children=[
                    html.Label("Empty stints"),
                    dcc.Checklist(id="ls-drop-empty", options=[{"label":"Drop 0-possession & 0 Δscore","value":"drop"}], value=[], inputStyle={"marginRight":"6px"}),
                ]),
                html.Div(children=[html.Label("Compare lineup A (line-by-line)"), dcc.Dropdown(id="ls-cmpA", options=[], value=None, placeholder="Pick lineup A")]),
                html.Div(children=[html.Label("Compare lineup B (line-by-line)"), dcc.Dropdown(id="ls-cmpB", options=[], value=None, placeholder="Pick lineup B")]),
                html.Div(className="md:col-span-2", children=[
                    html.Label("Players contain (comma-separated)"),
                    dcc.Input(id="ls-players-like", type="text", placeholder="e.g., Larkin, Pleiss", className="w-full", debounce=True),
                ]),
            ]),

            html.Div(className="grid grid-cols-1 md:grid-cols-6 gap-3 mt-3", children=[
                html.Div(children=[
                    html.Label("Lineup size"),
                    dcc.Dropdown(id="ls-lineup-size", options=[], value=None, clearable=True, placeholder="Any"),
                ]),
                html.Div(children=[html.Button("Refresh", id="ls-refresh", className="px-3 py-2 rounded border", n_clicks=0)]),
            ]),

            html.Hr(className="my-4"),
            dcc.Loading(dcc.Graph(id="ls-timeline"), type="dot"),

            html.H3("Stints Table (filtered, chronological)", className="mt-6 mb-2"),
            DataTable(
                id="ls-table",
                columns=[
                    {"name":"Team","id":"team_name"},
                    {"name":"Start","id":"start_label"},
                    {"name":"End","id":"end_label"},
                    {"name":"Score (start)","id":"score_start"},
                    {"name":"Score (end)","id":"score_end"},
                    {"name":"Time","id":"time_str"},
                    {"name":"Poss (For/Ag.)","id":"poss_str"},
                    {"name":"Pts (For-Ag.)","id":"pts_str"},
                    {"name":"Net","id":"net","type":"numeric","format":Format(precision=0)},
                    {"name":"ORtg","id":"ORtg","type":"numeric","format":Format(precision=1)},
                    {"name":"DRtg","id":"DRtg","type":"numeric","format":Format(precision=1)},
                    {"name":"Pos Mix","id":"lineup_pos_mix"},
                    {"name":"Lineup","id":"lineup_str"},
                ],
                data=[], sort_action="native", filter_action="native", page_size=25,
                style_table={"overflowX":"auto"},
                style_cell={"fontFamily":"system-ui, -apple-system, Segoe UI, Roboto, Arial","fontSize":13,"padding":"6px"},
                style_header={"fontWeight":"700"}, export_format="csv",
            ),

            html.H3("Game Sequence (filtered teams)", className="mt-8 mb-2"),
            DataTable(id="ls-seq-table", columns=[], data=[], sort_action="native", page_size=30,
                      style_table={"overflowX":"auto"},
                      style_cell={"fontFamily":"system-ui, -apple-system, Segoe UI, Roboto, Arial","fontSize":13,"padding":"6px"},
                      style_header={"fontWeight":"700"}, export_format="csv"),

            html.H3("Game-Cumulative Lineups (order-independent)", className="mt-8 mb-2"),
            DataTable(
                id="ls-cum-table",
                columns=[
                    {"name":"Team","id":"team_name"},
                    {"name":"Lineup","id":"lineup_str"},
                    {"name":"Pos Mix","id":"pos_mix"},
                    {"name":"Stints","id":"stints","type":"numeric","format":Format(precision=0)},
                    {"name":"Time","id":"time_str"},
                    {"name":"Poss For","id":"poss_for","type":"numeric","format":Format(precision=0)},
                    {"name":"Poss Ag.","id":"poss_against","type":"numeric","format":Format(precision=0)},
                    {"name":"Pts For","id":"pts_for","type":"numeric","format":Format(precision=0)},
                    {"name":"Pts Ag.","id":"pts_against","type":"numeric","format":Format(precision=0)},
                    {"name":"Net","id":"net","type":"numeric","format":Format(precision=0)},
                    {"name":"ORtg","id":"ORtg","type":"numeric","format":Format(precision=1)},
                    {"name":"DRtg","id":"DRtg","type":"numeric","format":Format(precision=1)},
                    {"name":"NetRtg","id":"NetRtg","type":"numeric","format":Format(precision=1)},
                ],
                data=[], sort_action="native", page_size=25,
                style_table={"overflowX":"auto"},
                style_cell={"fontFamily":"system-ui, -apple-system, Segoe UI, Roboto, Arial","fontSize":13,"padding":"6px"},
                style_header={"fontWeight":"700"}, export_format="csv",
            ),

            dcc.Store(id="ls-cache"),
        ]
    )

    # -----------------------------
    # Callbacks
    # -----------------------------

    @app.callback(
        Output("ls-game", "options"),
        Output("ls-game", "value"),
        Input("ls-week", "value"),
        State("ls-game", "value"),
        prevent_initial_call=False,
    )
    def _on_week_change(week, current_game):
        cat = discover_weeks(DATA_ROOT)
        games = cat.get(week, [])
        print(f"[lineup_stints] Week changed → {week} | games found: {len(games)}")
        opts = [{"label": f"{g['label']} ({g['game_id']})", "value": g["path"]} for g in games]
        val = current_game if current_game in {g["path"] for g in games} else (opts[0]["value"] if opts else None)
        print(f"[lineup_stints] Selected game → {val}")
        return opts, val

    @app.callback(
        Output("ls-cache", "data"),
        Output("ls-periods", "options"),
        Output("ls-periods", "value"),
        Output("ls-lineup-size", "options"),
        Input("ls-week", "value"),
        Input("ls-game", "value"),
        Input("ls-refresh", "n_clicks"),
        prevent_initial_call=False,
    )
    def _refresh(_week, game_path, _n):
        print(f"[lineup_stints] Refresh: week={_week} | game_path={game_path}")
        if not game_path:
            return {"error": "No game selected"}, [], None, []
        try:
            st, pos = _load_data_from_game_dir(Path(game_path))

            # minimal schema check (these are required to render anything)
            required = {"team_name","start_action","end_action","start_period","end_period",
                        "start_clock","end_clock","poss_for","poss_against","pts_for","pts_against","net"}
            missing = sorted(list(required - set(st.columns)))
            if missing:
                msg = f"Missing columns in stints: {missing}"
                print("[lineup_stints][ERROR]", msg)
                return {"error": msg}, [], None, []

            period_opts = sorted(st["start_period"].dropna().unique().astype(int).tolist())
            size_opts = [{"label": str(k), "value": int(k)} for k in sorted(st["lineup_size"].dropna().unique().astype(int).tolist())] if "lineup_size" in st.columns else []

            tmp = st.copy()
            tmp["lineup_key"] = tmp["lineup_tokens"].apply(_lineup_key_tokens) if "lineup_tokens" in tmp.columns else tmp["players_on_court"].apply(_lineup_key)
            uniq = tmp.drop_duplicates(subset=["team_name","lineup_key"])[["team_name","lineup_key"]]

            cmp_opts = [{
                "label": f"{row.team_name} | {', '.join(_players_to_labels_tokens(row.lineup_key, pos))}",
                "value": f"{row.team_name}|{','.join(row.lineup_key)}",
            } for row in uniq.itertuples(index=False)]

            print(f"[lineup_stints]  loaded stints={len(st)} periods={period_opts} compare_opts={len(cmp_opts)}")

            return {
                "st": st.to_dict("records"),
                "pos": pos.to_dict("records"),
                "cmp_opts": cmp_opts,
            }, [{"label": f"P{p}", "value": int(p)} for p in period_opts], period_opts, size_opts

        except Exception as e:
            print("[lineup_stints][EXCEPTION in _refresh]", repr(e))
            return {"error": str(e)}, [], None, []


    @app.callback(
        Output("ls-cmpA", "options"),
        Output("ls-cmpB", "options"),
        Input("ls-cache", "data"),
    )
    def _cmp_opts(cache):
        if not cache or "cmp_opts" not in cache:
            return [], []
        return cache["cmp_opts"], cache["cmp_opts"]

    def _apply_filters(st_df: pd.DataFrame, pos_df: pd.DataFrame,
                       periods: List[int] | None, min_ortg: float, max_drtg: float,
                       min_net: int, players_like: str | None, lineup_size: int | None,
                       drop_empty_segments: bool) -> pd.DataFrame:
        df = st_df.copy()
        # keys based on TOKENS
        df["lineup_key"] = df["lineup_tokens"].apply(_lineup_key_tokens)

        if periods:
            df = df[df["start_period"].isin(periods)]
        if min_ortg is not None:
            df = df[(df["ORtg"].isna()) | (df["ORtg"] >= min_ortg)]
        if max_drtg is not None:
            df = df[(df["DRtg"].isna()) | (df["DRtg"] <= max_drtg)]
        if min_net is not None:
            df = df[df["net"] >= min_net]
        if lineup_size:
            df = df[df["lineup_size"] == lineup_size]

        if drop_empty_segments:
            d_for = (df["score_end_for"] - df["score_start_for"]).fillna(0).astype(int)
            d_against = (df["score_end_against"] - df["score_start_against"]).fillna(0).astype(int)
            zero_poss = (df["poss_for"].fillna(0).astype(int) == 0) & (df["poss_against"].fillna(0).astype(int) == 0)
            zero_score = (d_for == 0) & (d_against == 0)
            df = df[~(zero_poss & zero_score)]

        if players_like:
            tokens = [t.strip().lower() for t in players_like.split(",") if t.strip()]
            if tokens:
                df["_lineup_names"] = df["lineup_key"].apply(lambda key: ", ".join(_players_to_labels_tokens(key, pos_df)).lower())
                for tkn in tokens:
                    df = df[df["_lineup_names"].str.contains(tkn, na=False)]
                df.drop(columns=["_lineup_names"], inplace=True, errors="ignore")

        df["start_label"] = "P" + df["start_period"].astype(int).astype(str) + " " + df["start_clock"].astype(str)
        df["end_label"]   = "P" + df["end_period"].astype(int).astype(str) + " " + df["end_clock"].astype(str)
        df["lineup_str"]  = df["lineup_key"].apply(lambda key: ", ".join(_players_to_labels_tokens(key, pos_df)))
        df["lineup_pos_mix"] = df["lineup_key"].apply(lambda key: _pos_mix_from_tokens(key, pos_df))
        df["poss_str"]    = df["poss_for"].astype(int).astype(str) + "/" + df["poss_against"].astype(int).astype(str)
        df["pts_str"]     = df["pts_for"].astype(int).astype(str) + "-" + df["pts_against"].astype(int).astype(str)
        df["time_str"]    = df["stint_seconds"].apply(_fmt_mmss)
        return df

    def _aggregate_lineups(st_df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
        if st_df.empty:
            return pd.DataFrame(columns=[
                "team_name","lineup_str","pos_mix","stints","time_str","poss_for","poss_against",
                "pts_for","pts_against","net","ORtg","DRtg","NetRtg"
            ])
        tmp = st_df.copy()
        if "lineup_key" not in tmp.columns:
            tmp["lineup_key"] = tmp["lineup_tokens"].apply(_lineup_key_tokens)
        grp = tmp.groupby(["team_name","lineup_key"], as_index=False).agg(
            stints=("stint_id","count"),
            time_sec=("stint_seconds","sum"),
            poss_for=("poss_for","sum"),
            poss_against=("poss_against","sum"),
            pts_for=("pts_for","sum"),
            pts_against=("pts_against","sum"),
            net=("net","sum"),
        )
        grp["ORtg"] = np.where(grp["poss_for"]>0, 100.0*grp["pts_for"]/grp["poss_for"], np.nan)
        grp["DRtg"] = np.where(grp["poss_against"]>0, 100.0*grp["pts_against"]/grp["poss_against"], np.nan)
        grp["NetRtg"] = grp["ORtg"] - grp["DRtg"]
        grp["lineup_str"] = grp["lineup_key"].apply(lambda key: ", ".join(_players_to_labels_tokens(key, pos_df)))
        grp["pos_mix"] = grp["lineup_key"].apply(lambda key: _pos_mix_from_tokens(key, pos_df))
        grp["time_str"] = grp["time_sec"].apply(_fmt_mmss)
        grp = grp.sort_values(["team_name","NetRtg","time_sec"], ascending=[True, False, False])
        return grp[["team_name","lineup_str","pos_mix","stints","time_str","poss_for","poss_against","pts_for","pts_against","net","ORtg","DRtg","NetRtg"]]

    @app.callback(
        Output("ls-timeline","figure"),
        Output("ls-table","data"),
        Output("ls-cum-table","data"),
        Output("ls-seq-table","columns"),
        Output("ls-seq-table","data"),
        Input("ls-cache","data"),
        Input("ls-periods","value"),
        Input("ls-min-ortg","value"),
        Input("ls-max-drtg","value"),
        Input("ls-min-net","value"),
        Input("ls-players-like","value"),
        Input("ls-lineup-size","value"),
        Input("ls-view","value"),
        Input("ls-drop-empty","value"),
        Input("ls-cmpA","value"),
        Input("ls-cmpB","value"),
    )
    def _update(cache, periods, min_ortg, max_drtg, min_net, players_like, lineup_size, view_mode,
                drop_empty_values, cmpA_val, cmpB_val):
        if not cache or "st" not in cache:
            msg = (cache.get("error") if isinstance(cache, dict) else "No data") if cache else "No data"
            print(f"[lineup_stints] _update received no data. error={msg}")
            fig = go.Figure()
            fig.add_annotation(text=f"⚠️ {msg}", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            return fig, [], [], [], []
        st = pd.DataFrame(cache["st"])
        pos = pd.DataFrame(cache["pos"]) if cache.get("pos") else pd.DataFrame()
        drop_empty = isinstance(drop_empty_values, list) and ("drop" in drop_empty_values)

        dff = _apply_filters(st, pos, periods, min_ortg, max_drtg, min_net, players_like, lineup_size, drop_empty_segments=drop_empty)

        print(f"[lineup_stints]  _update: st_in={len(st)} → after_filters={len(dff)} "
      f"(periods={periods}, min_ortg={min_ortg}, max_drtg={max_drtg}, min_net={min_net}, "
      f"lineup_size={lineup_size}, drop_empty={drop_empty})")


        if view_mode == "sequence":
            fig = _timeline_figure_sequence(dff, pos)
        else:
            fig = _timeline_figure_line_rows(dff, pos, cmpA_val, cmpB_val)

        if not dff.empty:
            dff_sorted = dff.sort_values(["start_action","end_action","team_name"])
            stint_cols = ["team_name","start_label","end_label","score_start","score_end","time_str","poss_str","pts_str","net","ORtg","DRtg","lineup_pos_mix","lineup_str"]
            stints_rows = dff_sorted[stint_cols].to_dict("records")
        else:
            stints_rows = []

        cum_df = _aggregate_lineups(dff, pos)
        cum_rows = cum_df.to_dict("records") if not cum_df.empty else []

        seq_cols, seq_rows = [], []
        if view_mode == "sequence":
            seq_rows_raw, _ = _build_sequence_rows_general(dff, pos)
            if seq_rows_raw:
                base_cols = [{"name":"Segment","id":"segment"},{"name":"Start","id":"start"},{"name":"End","id":"end"}]
                team_cols = [c for c in seq_rows_raw[0].keys() if c.endswith(" lineup")]
                seq_cols = base_cols + [{"name":k,"id":k} for k in team_cols]
                seq_rows = sorted(seq_rows_raw, key=lambda r: (r["_a0"], r["_a1"]))
                for r in seq_rows:
                    r.pop("_a0", None); r.pop("_a1", None)

        return fig, stints_rows, cum_rows, seq_cols, seq_rows

    return app
