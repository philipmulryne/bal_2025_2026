# dashapps/extractor_ui.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
import os, time, re, unicodedata
from collections import Counter, defaultdict
from flask import request

import dash
from dash import Dash, html, dcc, Input, Output, State

# ---------------- import your extractor ----------------
try:
    from extractor_stints_fixed import (
        process_one_game,
        process_week,
        DEFAULT_OVERRIDE,
    )
except Exception as e:
    process_one_game = process_week = None
    DEFAULT_OVERRIDE = Path("positions_override.csv")
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# ---------------- import your compiler ----------------
try:
    from compile_cumulative import (
        GamePath,
        compile_players,
        compile_pbp,
        compile_games_index,
        write_outputs,
        discover_games,
    )
except Exception as e:
    GamePath = None
    compile_players = compile_pbp = compile_games_index = write_outputs = discover_games = None
    _COMP_IMPORT_ERROR = e
else:
    _COMP_IMPORT_ERROR = None


# ---------------- helpers -------------------------------

def _as_path(x: Optional[str]) -> Optional[Path]:
    if x is None:
        return None
    sx = str(x).strip()
    return Path(sx) if sx else None

def _safe_exists(p: Optional[Path]) -> bool:
    return bool(p and p.exists())

def _human(p: Optional[Path]) -> str:
    return str(p) if p else "(not set)"

def _normalize_out_root(s: Optional[str]) -> Path:
    p = _as_path(s) or Path(os.environ.get("DATA_ROOT") or "out_data")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _default_override() -> str:
    p = Path("positions_override.csv")
    return str(p) if p.exists() else ""

def _read_existing_table(dest_dir: Path, base: str):
    import pandas as pd
    csv_gz = dest_dir / f"{base}.csv.gz"
    csv = dest_dir / f"{base}.csv"
    if csv_gz.exists():
        return pd.read_csv(csv_gz, low_memory=False), True
    if csv.exists():
        return pd.read_csv(csv, low_memory=False), False
    return pd.DataFrame(), False

def _write_table_atomic(df, dest_dir: Path, base: str, gzip: bool):
    """
    Atomic-ish write: write to a temp file then replace.
    """
    if df is None or df.empty:
        return None
    import uuid
    tmp = dest_dir / f".{base}.{uuid.uuid4().hex}.tmp"
    out = dest_dir / (f"{base}.csv.gz" if gzip else f"{base}.csv")
    comp = "gzip" if gzip else None
    df.to_csv(tmp, index=False, compression=comp)
    tmp.replace(out)
    return out

def _parquet_write(df, dest_dir: Path, base: str):
    """
    Always (re)write Parquet for coherence after replace/append.
    Fails silently if pyarrow/fastparquet unavailable.
    """
    if df is None or df.empty:
        return None
    out = dest_dir / f"{base}.parquet"
    try:
        df.to_parquet(out, index=False)
        return out
    except Exception:
        return None

def _union_align(existing_df, new_df):
    """
    Return (aligned_existing, aligned_new, final_columns_order)
    Strategy:
      - Column superset
      - Preserve existing column order; append any new cols at the end
    """
    all_cols = list(existing_df.columns)
    for c in new_df.columns:
        if c not in all_cols:
            all_cols.append(c)
    aligned_existing = existing_df.reindex(columns=all_cols)
    aligned_new = new_df.reindex(columns=all_cols)
    return aligned_existing, aligned_new, all_cols

def _bump_data_version(out_root: Path):
    """
    Touch a small version file to signal data updates to all Dash apps.
    """
    v = out_root / "_cumulative" / ".data_version"
    v.parent.mkdir(parents=True, exist_ok=True
                   )
    try:
        v.write_text(str(int(time.time())), encoding="utf-8")
        os.utime(v, None)
    except Exception:
        pass


# ---------------- team-name canonicalization -------------------------------

_TURKISH = set("çğıöşüÇĞİÖŞÜ")

def _has_tr(s: str) -> bool:
    return any(ch in _TURKISH for ch in (s or ""))

_TR_MAP = str.maketrans({
    "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
    "ç": "c", "ğ": "g", "ı": "i", "i": "i", "ö": "o", "ş": "s", "ü": "u",
    "’": "'", "‘": "'", "´": "'", "`": "'",
    "–": "-", "—": "-", "-": "-",
})

def _asciiish_key(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s_norm = unicodedata.normalize("NFKD", s).translate(_TR_MAP)
    s_ascii = "".join(ch for ch in s_norm if not unicodedata.combining(ch))
    s_up = re.sub(r"[^A-Z0-9 ]+", " ", s_ascii.upper())
    return re.sub(r"\s+", " ", s_up).strip()

def _guess_team_columns(df) -> List[str]:
    import pandas as pd
    cand = re.compile("|".join([
        r"(^|_)team(_|$)",
        r"(^|_)home(_|$)",
        r"(^|_)away(_|$)",
        r"(^|_)opponent(_|$)",
        r"(^|_)opp(_|$)",
        r"(^|_)club(_|$)",
        r"(^|_)franchise(_|$)",
    ]), flags=re.IGNORECASE)
    cols = []
    for c in df.columns:
        if df[c].dtype == "O" and cand.search(c):
            s = pd.Series(df[c].dropna().unique()).astype(str)
            if not s.empty and (s.str.len().ge(3)).mean() >= 0.5:
                cols.append(c)
    return cols

def _choose_canonical(variants_with_counts: List[Tuple[str,int]]) -> str:
    # Prefer Turkish-diacritic variants; tie-break by count desc, then shortest, then alpha.
    pool = [(v,n) for v,n in variants_with_counts if _has_tr(v)] or variants_with_counts
    pool.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
    return pool[0][0] if pool else ""

def _load_team_map_file(p: Path) -> dict[str,str]:
    import pandas as pd
    try:
        m = pd.read_csv(p, low_memory=False)
        cols = {c.lower(): c for c in m.columns}
        old = cols.get("old_name") or cols.get("variant") or list(m.columns)[0]
        can = cols.get("canonical_name") or cols.get("canonical") or list(m.columns)[1]
        out = {}
        for o, c in zip(m[old].astype(str), m[can].astype(str)):
            if o and c:
                out[o] = c
        return out
    except Exception:
        return {}

def _save_team_map_file(p: Path, mapping: dict[str,str]) -> None:
    import pandas as pd
    if not mapping:
        return
    rows = [{"old_name": k, "canonical_name": v} for k, v in sorted(mapping.items())]
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)

def _standardize_team_columns(df, persistent_map: dict[str,str]) -> tuple:
    """Return (df2, updated_map, stats_str)."""
    import pandas as pd
    if df is None or df.empty:
        return df, persistent_map, "no-op (empty)"
    df2 = df.copy()
    cols = _guess_team_columns(df2)
    if not cols:
        return df2, persistent_map, "no team columns detected"

    # Build asciiish clusters from current DF
    clusters: dict[str, Counter] = defaultdict(Counter)
    for col in cols:
        for val, cnt in Counter(df2[col].dropna().astype(str)).items():
            clusters[_asciiish_key(val)][val] += cnt

    # Derive canonical per cluster
    auto_map: dict[str,str] = {}
    for key, cnts in clusters.items():
        variants = list(cnts.items())
        canonical = _choose_canonical(variants)
        for v,_n in variants:
            auto_map[v] = canonical

    # Merge persistent map over auto (persistent wins)
    merged_map = dict(auto_map)
    merged_map.update(persistent_map)

    # Apply
    changed = 0
    for col in cols:
        before = df2[col].copy()
        df2[col] = df2[col].astype(str).map(lambda s: merged_map.get(s, s))
        changed += int((before != df2[col]).sum())

    # Enrich persistent map with any new discoveries
    enriched = dict(persistent_map)
    for v, c in merged_map.items():
        if v != c and v not in enriched:
            enriched[v] = c

    return df2, enriched, f"cols={cols}, changes={changed}"


# ---------------- de-duplication utilities ----------------

def _dedupe_pbp(df):
    """
    Prefer keyed dedupe if we can detect an event key; else exact-row dedupe.
    """
    if df is None or df.empty:
        return df, 0
    before = len(df)
    cols = set(c.lower() for c in df.columns)
    # Find an event key
    key_candidates = ["event_number", "event_no", "event_id", "row_id", "idx", "sequence_no"]
    key = None
    for k in key_candidates:
        if k in cols:
            # map back to the real column name (case-sensitive)
            key = next(c for c in df.columns if c.lower() == k)
            break
    gid_col = next((c for c in df.columns if c.lower() == "game_id"), None)
    if key and gid_col:
        df = df.drop_duplicates(subset=[gid_col, key], keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    return df, removed

def _dedupe_players(df):
    if df is None or df.empty:
        return df, 0
    before = len(df)
    cols = {c.lower(): c for c in df.columns}
    keys = []
    for k in ["game_id", "player_id", "team_id", "stint_no"]:
        if k in cols:
            keys.append(cols[k])
    if len(keys) >= 2:
        df = df.drop_duplicates(subset=keys, keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    return df, removed

def _dedupe_games_index(df):
    if df is None or df.empty:
        return df, 0
    before = len(df)
    gid = next((c for c in df.columns if c.lower() == "game_id"), None)
    if gid:
        df = df.drop_duplicates(subset=[gid], keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    return df, removed

def _apply_inline_dedupe(players_df, pbp_df, idx_df) -> Tuple:
    p, pr = _dedupe_players(players_df)
    b, br = _dedupe_pbp(pbp_df)
    i, ir = _dedupe_games_index(idx_df)
    return (p, b, i, pr, br, ir)


# --------------- REMOVE (purge) one game in cumulative ----------------------

def _remove_game_from_cumulative(out_root: Path, game_id: str, week: Optional[str], strict_week: bool) -> str:
    if _COMP_IMPORT_ERROR:
        return f"Cannot remove: compiler import error → {_COMP_IMPORT_ERROR}"

    import pandas as pd

    out_root = out_root.resolve()
    dest_dir = out_root / "_cumulative"
    dest_dir.mkdir(parents=True, exist_ok=True)

    players_all, gz_players = _read_existing_table(dest_dir, "players_all")
    pbp_all,     gz_pbp     = _read_existing_table(dest_dir, "pbp_all")
    idx_all,     gz_idx     = _read_existing_table(dest_dir, "games_index")

    def _filter(df):
        if df.empty:
            return df, 0
        cols = {c.lower(): c for c in df.columns}
        gid = cols.get("game_id")
        wk  = cols.get("week")
        if not gid:
            return df, 0
        before = len(df)
        if strict_week and wk:
            df = df[~((df[gid].astype(str) == str(game_id)) & (df[wk].astype(str) == str(week or "")))]
        else:
            df = df[df[gid].astype(str) != str(game_id)]
        removed = before - len(df)
        return df, removed

    players_all, r1 = _filter(players_all)
    pbp_all,     r2 = _filter(pbp_all)
    idx_all,     r3 = _filter(idx_all)

    p_csv = _write_table_atomic(players_all, dest_dir, "players_all", gz_players)
    b_csv = _write_table_atomic(pbp_all,     dest_dir, "pbp_all",     gz_pbp)
    i_csv = _write_table_atomic(idx_all,     dest_dir, "games_index", gz_idx)

    _ = _parquet_write(players_all, dest_dir, "players_all")
    _ = _parquet_write(pbp_all,     dest_dir, "pbp_all")
    _ = _parquet_write(idx_all,     dest_dir, "games_index")

    _bump_data_version(out_root)

    total = r1 + r2 + r3
    scope = "week+game" if strict_week else "game_id only"
    return (
        "[REMOVE OK]\n"
        f"scope: {scope}\n"
        f"out_root: {out_root}\n"
        f"removed rows → players_all: {r1}, pbp_all: {r2}, games_index: {r3} (total {total})\n"
        f"players_all → {p_csv}\n"
        f"pbp_all     → {b_csv}\n"
        f"games_index → {i_csv}"
    )


# --------------- REPLACE (upsert) one game in cumulative ---------------------

def _replace_game_in_cumulative(out_root: Path, week: str, game_id: str) -> str:
    if _COMP_IMPORT_ERROR:
        return f"Cannot update cumulative: compiler import error → {_COMP_IMPORT_ERROR}"

    import pandas as pd

    out_root = out_root.resolve()
    dest_dir = out_root / "_cumulative"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build fresh single-game frames
    game_dir = out_root / week / game_id
    gp = GamePath(
        week=week,
        game_id=game_id,
        game_dir=game_dir,
        meta_path=game_dir / "meta.json",
        pbp_path=game_dir / "game_pbp_normalized.csv",
        players_path=game_dir / "game_players_with_positions.csv",
    )
    new_players = compile_players([gp])
    new_pbp     = compile_pbp([gp])
    new_idx     = compile_games_index([gp])

    # ---- Team-name canonicalization on new frames ----
    map_path = (out_root / "_cumulative" / "team_name_map.csv")
    env_map = os.environ.get("TEAM_CANON_MAP")
    if env_map:
        map_path = Path(env_map)
    persistent_map = _load_team_map_file(map_path)

    if new_players is not None and not new_players.empty:
        new_players, persistent_map, _ = _standardize_team_columns(new_players, persistent_map)
    if new_pbp is not None and not new_pbp.empty:
        new_pbp, persistent_map, _ = _standardize_team_columns(new_pbp, persistent_map)
    if new_idx is not None and not new_idx.empty:
        new_idx, persistent_map, _ = _standardize_team_columns(new_idx, persistent_map)

    _save_team_map_file(map_path, persistent_map)

    # Load existing
    players_all, gz_players = _read_existing_table(dest_dir, "players_all")
    pbp_all,     gz_pbp     = _read_existing_table(dest_dir, "pbp_all")
    idx_all,     gz_idx     = _read_existing_table(dest_dir, "games_index")

    # Ensure keys exist if empty/malformed
    for df in (players_all, pbp_all, idx_all):
        if "week" not in df.columns:
            df["week"] = pd.NA
        if "game_id" not in df.columns:
            df["game_id"] = pd.NA

    # Drop old rows for this game (strict on both week and game)
    if not players_all.empty:
        players_all = players_all[~((players_all["week"].astype(str) == str(week)) & (players_all["game_id"].astype(str) == str(game_id)))]
    if not pbp_all.empty:
        pbp_all = pbp_all[~((pbp_all["week"].astype(str) == str(week)) & (pbp_all["game_id"].astype(str) == str(game_id)))]
    if not idx_all.empty:
        idx_all = idx_all[~((idx_all["week"].astype(str) == str(week)) & (idx_all["game_id"].astype(str) == str(game_id)))]

    # Union-align and append
    if new_players is not None and not new_players.empty:
        players_merged = new_players if players_all.empty else \
            (lambda e, n: __import__("pandas").concat([e, n], ignore_index=True))(*_union_align(players_all, new_players)[:2])
    else:
        players_merged = players_all

    if new_pbp is not None and not new_pbp.empty:
        pbp_merged = new_pbp if pbp_all.empty else \
            (lambda e, n: __import__("pandas").concat([e, n], ignore_index=True))(*_union_align(pbp_all, new_pbp)[:2])
    else:
        pbp_merged = pbp_all

    if new_idx is not None and not new_idx.empty:
        idx_merged = new_idx if idx_all.empty else \
            (lambda e, n: __import__("pandas").concat([e, n], ignore_index=True))(*_union_align(idx_all, new_idx)[:2])
    else:
        idx_merged = idx_all

    # Inline de-duplication (schema-aware where possible)
    players_merged, pr = _dedupe_players(players_merged)
    pbp_merged,     br = _dedupe_pbp(pbp_merged)
    idx_merged,     ir = _dedupe_games_index(idx_merged)

    # Write back (atomic-ish), preserve gzip choice, refresh Parquet
    p_csv = _write_table_atomic(players_merged, dest_dir, "players_all", gz_players)
    b_csv = _write_table_atomic(pbp_merged,     dest_dir, "pbp_all",     gz_pbp)
    i_csv = _write_table_atomic(idx_merged,     dest_dir, "games_index", gz_idx)

    _ = _parquet_write(players_merged, dest_dir, "players_all")
    _ = _parquet_write(pbp_merged,     dest_dir, "pbp_all")
    _ = _parquet_write(idx_merged,     dest_dir, "games_index")

    _bump_data_version(out_root)

    return (
        "[REPLACE OK]\n"
        f"out_root: {out_root}\n"
        f"game: {week}/{game_id}\n"
        f"de-duped rows removed → players: {pr}, pbp: {br}, games_index: {ir}\n"
        f"players_all → {p_csv}\n"
        f"pbp_all     → {b_csv}\n"
        f"games_index → {i_csv}"
    )


# --------------- APPEND (blind) one game in cumulative ----------------------

def _append_game_to_cumulative(out_root: Path, week: str, game_id: str) -> str:
    if _COMP_IMPORT_ERROR:
        return f"Cannot append: compiler import error → {_COMP_IMPORT_ERROR}"

    import pandas as pd

    out_root = out_root.resolve()
    dest_dir = out_root / "_cumulative"
    dest_dir.mkdir(parents=True, exist_ok=True)

    game_dir = out_root / week / game_id
    gp = GamePath(
        week=week,
        game_id=game_id,
        game_dir=game_dir,
        meta_path=game_dir / "meta.json",
        pbp_path=game_dir / "game_pbp_normalized.csv",
        players_path=game_dir / "game_players_with_positions.csv",
    )

    new_players = compile_players([gp])
    new_pbp     = compile_pbp([gp])
    new_idx     = compile_games_index([gp])

    # ---- Team-name canonicalization on new frames ----
    map_path = (out_root / "_cumulative" / "team_name_map.csv")
    env_map = os.environ.get("TEAM_CANON_MAP")
    if env_map:
        map_path = Path(env_map)
    persistent_map = _load_team_map_file(map_path)

    if new_players is not None and not new_players.empty:
        new_players, persistent_map, _ = _standardize_team_columns(new_players, persistent_map)
    if new_pbp is not None and not new_pbp.empty:
        new_pbp, persistent_map, _ = _standardize_team_columns(new_pbp, persistent_map)
    if new_idx is not None and not new_idx.empty:
        new_idx, persistent_map, _ = _standardize_team_columns(new_idx, persistent_map)

    _save_team_map_file(map_path, persistent_map)

    players_all, gz_players = _read_existing_table(dest_dir, "players_all")
    pbp_all,     gz_pbp     = _read_existing_table(dest_dir, "pbp_all")
    idx_all,     gz_idx     = _read_existing_table(dest_dir, "games_index")

    # Quick presence check to warn user (we'll still dedupe inline to be safe)
    def _present(df):
        if df.empty:
            return False
        gid = next((c for c in df.columns if c.lower() == "game_id"), None)
        return bool(gid and (df[gid].astype(str) == str(game_id)).any())

    warn = []
    if _present(pbp_all) or _present(idx_all) or _present(players_all):
        warn.append(f"[WARN] Some rows for game_id={game_id} already exist. Append will be auto-deduped.")

    # Union-align and append blindly + dedupe
    if new_players is not None and not new_players.empty:
        players_merged = new_players if players_all.empty else \
            (lambda e, n: __import__("pandas").concat([e, n], ignore_index=True))(*_union_align(players_all, new_players)[:2])
    else:
        players_merged = players_all

    if new_pbp is not None and not new_pbp.empty:
        pbp_merged = new_pbp if pbp_all.empty else \
            (lambda e, n: __import__("pandas").concat([e, n], ignore_index=True))(*_union_align(pbp_all, new_pbp)[:2])
    else:
        pbp_merged = pbp_all

    if new_idx is not None and not new_idx.empty:
        idx_merged = new_idx if idx_all.empty else \
            (lambda e, n: __import__("pandas").concat([e, n], ignore_index=True))(*_union_align(idx_all, new_idx)[:2])
    else:
        idx_merged = idx_all

    players_merged, pr = _dedupe_players(players_merged)
    pbp_merged,     br = _dedupe_pbp(pbp_merged)
    idx_merged,     ir = _dedupe_games_index(idx_merged)

    p_csv = _write_table_atomic(players_merged, dest_dir, "players_all", gz_players)
    b_csv = _write_table_atomic(pbp_merged,     dest_dir, "pbp_all",     gz_pbp)
    i_csv = _write_table_atomic(idx_merged,     dest_dir, "games_index", gz_idx)

    _ = _parquet_write(players_merged, dest_dir, "players_all")
    _ = _parquet_write(pbp_merged,     dest_dir, "pbp_all")
    _ = _parquet_write(idx_merged,     dest_dir, "games_index")

    _bump_data_version(out_root)

    note = "\n".join(warn) + ("\n" if warn else "")
    return (
        "[APPEND OK]\n" + note +
        f"out_root: {out_root}\n"
        f"game: {week}/{game_id}\n"
        f"de-duped rows removed → players: {pr}, pbp: {br}, games_index: {ir}\n"
        f"players_all → {p_csv}\n"
        f"pbp_all     → {b_csv}\n"
        f"games_index → {i_csv}"
    )


# --------------- REBUILD cumulative -----------------------------------------

def _rebuild_cumulative(out_root: Path, weeks: Optional[List[str]], parquet: bool, gzip_csv: bool) -> str:
    if _COMP_IMPORT_ERROR:
        return f"Cannot rebuild: compiler import error → {_COMP_IMPORT_ERROR}"
    games = discover_games(out_root, weeks)
    if not games:
        return f"[ERROR] No games found under {out_root} (weeks={weeks or 'ALL'})."

    players_df = compile_players(games)
    pbp_df     = compile_pbp(games)
    idx_df     = compile_games_index(games)

    # Canonicalize names on compiled frames before write
    map_path = (out_root / "_cumulative" / "team_name_map.csv")
    env_map = os.environ.get("TEAM_CANON_MAP")
    if env_map:
        map_path = Path(env_map)
    persistent_map = _load_team_map_file(map_path)

    players_df, persistent_map, _ = _standardize_team_columns(players_df, persistent_map)
    pbp_df,     persistent_map, _ = _standardize_team_columns(pbp_df,     persistent_map)
    idx_df,     persistent_map, _ = _standardize_team_columns(idx_df,     persistent_map)
    _save_team_map_file(map_path, persistent_map)

    # One last dedupe pass before writing
    players_df, _ = _dedupe_players(players_df)
    pbp_df,     _ = _dedupe_pbp(pbp_df)
    idx_df,     _ = _dedupe_games_index(idx_df)

    dest_dir = out_root / "_cumulative"
    write_outputs(dest_dir, players_df, pbp_df, idx_df, make_parquet=parquet, gzip_csv=gzip_csv)

    _bump_data_version(out_root)

    return (
        "[REBUILD OK]\n"
        f"out_root: {out_root}\n"
        f"weeks: {weeks or 'ALL'}\n"
        f"dest: {dest_dir}"
    )


# --------------- STANDARDIZE existing cumulative (manual action) -------------

def _standardize_cumulative_now(out_root: Path) -> str:
    import pandas as pd
    dest_dir = out_root / "_cumulative"
    players_all, gz_players = _read_existing_table(dest_dir, "players_all")
    pbp_all,     gz_pbp     = _read_existing_table(dest_dir, "pbp_all")
    idx_all,     gz_idx     = _read_existing_table(dest_dir, "games_index")

    map_path = (out_root / "_cumulative" / "team_name_map.csv")
    env_map = os.environ.get("TEAM_CANON_MAP")
    if env_map:
        map_path = Path(env_map)
    persistent_map = _load_team_map_file(map_path)

    players_all, persistent_map, s1 = _standardize_team_columns(players_all, persistent_map)
    pbp_all,     persistent_map, s2 = _standardize_team_columns(pbp_all,     persistent_map)
    idx_all,     persistent_map, s3 = _standardize_team_columns(idx_all,     persistent_map)

    _save_team_map_file(map_path, persistent_map)

    p_csv = _write_table_atomic(players_all, dest_dir, "players_all", gz_players)
    b_csv = _write_table_atomic(pbp_all,     dest_dir, "pbp_all",     gz_pbp)
    i_csv = _write_table_atomic(idx_all,     dest_dir, "games_index", gz_idx)

    _ = _parquet_write(players_all, dest_dir, "players_all")
    _ = _parquet_write(pbp_all,     dest_dir, "pbp_all")
    _ = _parquet_write(idx_all,     dest_dir, "games_index")

    _bump_data_version(out_root)
    return (
        "[STANDARDIZE OK]\n"
        f"players_all → {p_csv}\n"
        f"pbp_all     → {b_csv}\n"
        f"games_index → {i_csv}\n"
        f"notes: players({s1}), pbp({s2}), idx({s3})"
    )


# --------------- fetch by ID -------------------

def _fetch_game_json(game_id: str, base_url_tmpl: str, dest_root: Path, week: str) -> Path:
    try:
        import requests
    except Exception as e:
            raise RuntimeError(f"requests is required to fetch JSON (pip install requests). Underlying error: {e}")

    url = base_url_tmpl.format(id=game_id, gid=game_id, game_id=game_id)
    headers = {"User-Agent": "analytics-hub/1.0 (+fetch-by-id)"}
    r = requests.get(url, headers=headers, timeout=(5, 30))
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} → HTTP {r.status_code}")
    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Response was not valid JSON: {e}")

    out_dir = dest_root / week
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{game_id}.json"

    import json
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_file


# --------------- Dash factory -----------------------------------------------

def create_dash_extractor(server, base_pathname: str = "/extractor/") -> Dash:
    # Register a one-time restart route on the parent Flask server (optional now)
    if not getattr(server, "_has_restart_route", False):
        @server.route("/__restart", methods=["POST"])
        def __restart_route():
            token = os.environ.get("RESTART_TOKEN")
            if token and request.headers.get("X-Restart-Token") != token:
                return "Forbidden", 403
            try:
                os.utime(Path(__file__).resolve(), None)
            except Exception:
                pass
            func = request.environ.get("werkzeug.server.shutdown")
            if func is not None:
                func()
            else:
                os._exit(0)
            return "Restarting…", 200

        server._has_restart_route = True

    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
    )

    env_json_root = os.environ.get("BSL_JSON_ROOT") or os.environ.get("JSON_ROOT") or ""
    env_data_root = os.environ.get("DATA_ROOT") or "out_data"
    env_raw_root  = os.environ.get("RAW_JSON_ROOT") or os.environ.get("BSL_JSON_ARCHIVE") or "raw_json"

    app.layout = html.Div(
        style={"padding": "18px", "maxWidth": "980px", "margin": "0 auto"},
        children=[
            html.H2("Game Extractor"),
            html.P("Fetch JSON by game ID, run single games or whole weeks, and manage cumulative outputs."),

            html.Div(
                id="import-error",
                children=(
                    html.Div(f"Extractor import error: {_IMPORT_ERROR}",
                             style={"color": "#ff6b6b", "marginBottom": "8px"}) if _IMPORT_ERROR else ""
                )
            ),
            html.Div(
                id="comp-error",
                children=(
                    html.Div(f"Compiler import error: {_COMP_IMPORT_ERROR}",
                             style={"color": "#ff6b6b", "marginBottom": "8px"}) if _COMP_IMPORT_ERROR else ""
                )
            ),

            dcc.Store(id="last-game", data=None),
            dcc.Store(id="restart-token", data=os.environ.get("RESTART_TOKEN") or ""),

            dcc.Tabs(id="mode", value="fetch", children=[

                # ----------- Fetch by ID -----------
                dcc.Tab(label="Fetch by ID", value="fetch", children=[
                    html.Br(),
                    html.Label("Game ID"),
                    dcc.Input(id="fetch-id", type="text", placeholder="e.g., 2715220",
                              value="", style={"width": "40%"}),
                    html.Br(), html.Br(),
                    html.Label("Base URL template"),
                    dcc.Input(
                        id="fetch-url",
                        type="text",
                        value="https://fibalivestats.dcd.shared.geniussports.com/data/{id}/data.json",
                        style={"width": "100%"},
                    ),
                    html.Br(), html.Br(),
                    html.Label("Raw JSON archive root (destination)"),
                    dcc.Input(id="fetch-raw-root", type="text", value=env_raw_root,
                              style={"width": "100%"}),
                    html.Br(), html.Br(),
                    html.Label("Week folder (e.g., week01)"),
                    dcc.Input(id="fetch-week", type="text", placeholder="week01", value="",
                              style={"width": "40%"}),
                    html.Br(), html.Br(),
                    dcc.Checklist(
                        id="fetch-then-run",
                        options=[
                            {"label": "Run extractor after download", "value": "run"},
                            {"label": "Replace this game in cumulative after run", "value": "replace"},
                            {"label": "Append this game to cumulative after run (may duplicate)", "value": "append"},
                        ],
                        value=[],
                        style={"marginBottom": "8px"}
                    ),
                    html.Label("Output root (for extractor run)"),
                    dcc.Input(id="fetch-out-root", type="text", value=env_data_root,
                              style={"width": "100%"}),
                    html.Br(),
                    html.Button("Fetch (and optionally run)", id="fetch-btn", n_clicks=0),
                ]),

                # ----------- Single Game -----------
                dcc.Tab(label="Single Game JSON", value="game", children=[
                    html.Br(),
                    html.Label("Path to game JSON file"),
                    dcc.Input(id="src-json", type="text", placeholder="e.g., /data/week01/2714807.json",
                              value="", style={"width": "100%"}),

                    html.Br(), html.Br(),
                    html.Label("Week name (optional)"),
                    dcc.Input(id="week-name", type="text", placeholder="Defaults to parent folder name",
                              value="", style={"width": "100%"}),

                    html.Br(), html.Br(),
                    html.Label("Positions override CSV (optional)"),
                    dcc.Input(id="override-csv", type="text", value=_default_override(),
                              placeholder="positions_override.csv", style={"width": "100%"}),

                    html.Br(), html.Br(),
                    html.Label("Output root"),
                    dcc.Input(id="out-root-game", type="text", value=env_data_root,
                              placeholder="out_data (or $DATA_ROOT)", style={"width": "100%"}),

                    html.Br(), html.Br(),
                    dcc.Checklist(
                        id="after-run",
                        options=[
                            {"label": "Replace this game in cumulative after run", "value": "replace"},
                            {"label": "Append this game to cumulative after run (may duplicate)", "value": "append"},
                        ],
                        value=[],
                        style={"marginBottom": "8px"}
                    ),
                    html.Button("Run single game", id="run-game", n_clicks=0),
                ]),

                # ----------- Whole Week -----------
                dcc.Tab(label="Whole Week Folder", value="week", children=[
                    html.Br(),
                    html.Label("Path to week folder (contains multiple *.json)"),
                    dcc.Input(id="week-dir", type="text", value=env_json_root,
                              placeholder="e.g., /data/BSL/2025_26/week01", style={"width": "100%"}),

                    html.Br(), html.Br(),
                    html.Label("Positions override CSV (optional)"),
                    dcc.Input(id="override-csv-week", type="text", value=_default_override(),
                              placeholder="positions_override.csv", style={"width": "100%"}),

                    html.Br(), html.Br(),
                    html.Label("Output root"),
                    dcc.Input(id="out-root-week", type="text", value=env_data_root,
                              placeholder="out_data (or $DATA_ROOT)", style={"width": "100%"}),

                    html.Br(), html.Br(),
                    dcc.Checklist(
                        id="rebuild-after-week",
                        options=[{"label": "Rebuild cumulative for THIS week after run", "value": "rebuild"}],
                        value=[],
                        style={"marginBottom": "8px"}
                    ),
                    html.Button("Run whole week", id="run-week", n_clicks=0),
                ]),

                # ----------- Compile Cumulative -----------
                dcc.Tab(label="Compile Cumulative", value="compile", children=[
                    html.Br(),
                    html.H4("Replace just one game"),
                    html.Div([
                        html.Div([
                            html.Label("Output root"),
                            dcc.Input(id="out-root-replace", type="text", value=env_data_root,
                                      style={"width": "100%"}),
                        ]),
                        html.Br(),
                        html.Label("Week"),
                        dcc.Input(id="replace-week", type="text", placeholder="e.g., week01",
                                  style={"width": "40%"}),
                        html.Br(), html.Br(),
                        html.Label("Game ID (folder name under the week)"),
                        dcc.Input(id="replace-game-id", type="text", placeholder="e.g., 2714807",
                                  style={"width": "40%"}),
                        html.Br(), html.Br(),
                        html.Button("Replace this game", id="replace-game-btn", n_clicks=0),
                        html.Div(style={"height": "18px"}),
                        html.Button("Use last-run game (replace)", id="replace-last-btn", n_clicks=0,
                                    title="Replace the game you last processed in 'Fetch' or 'Single Game'"),
                    ], style={"marginBottom": "20px"}),

                    html.Hr(),
                    html.H4("Append just one game (auto-dedup)"),
                    html.Div([
                        html.Div([
                            html.Label("Output root"),
                            dcc.Input(id="out-root-append", type="text", value=env_data_root,
                                      style={"width": "100%"}),
                        ]),
                        html.Br(),
                        html.Label("Week"),
                        dcc.Input(id="append-week", type="text", placeholder="e.g., week01",
                                  style={"width": "40%"}),
                        html.Br(), html.Br(),
                        html.Label("Game ID (folder name under the week)"),
                        dcc.Input(id="append-game-id", type="text", placeholder="e.g., 2714807",
                                  style={"width": "40%"}),
                        html.Br(), html.Br(),
                        html.Button("Append this game", id="append-game-btn", n_clicks=0),
                        html.Div(style={"height": "18px"}),
                        html.Button("Use last-run game (append)", id="append-last-btn", n_clicks=0,
                                    title="Append the game you last processed in 'Fetch' or 'Single Game'"),
                    ], style={"marginBottom": "20px"}),

                    html.Hr(),
                    html.H4("Remove/Purge one game from cumulative"),
                    html.Div([
                        html.Div([
                            html.Label("Output root"),
                            dcc.Input(id="out-root-remove", type="text", value=env_data_root,
                                      style={"width": "100%"}),
                        ]),
                        html.Br(),
                        html.Label("Game ID"),
                        dcc.Input(id="remove-game-id", type="text", placeholder="e.g., 2715217",
                                  style={"width": "40%"}),
                        html.Br(), html.Br(),
                        html.Label("Week (optional; enable strict if you want week+game match)"),
                        dcc.Input(id="remove-week", type="text", placeholder="e.g., week01",
                                  style={"width": "40%"}),
                        dcc.Checklist(
                            id="remove-strict",
                            options=[{"label": "Strict week match", "value": "strict"}],
                            value=[]
                        ),
                        html.Br(),
                        html.Button("Remove from cumulative", id="remove-btn", n_clicks=0),
                    ], style={"marginBottom": "20px"}),

                    html.Hr(),
                    html.H4("Dedupe cumulative tables"),
                    html.Div([
                        html.Label("Output root"),
                        dcc.Input(id="out-root-dedupe", type="text", value=env_data_root, style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Button("De-duplicate now", id="dedupe-btn", n_clicks=0),
                    ], style={"marginBottom": "8px"}),

                    html.Hr(),
                    html.H4("Standardize Team Names (cumulative)"),
                    html.Div([
                        html.Label("Output root"),
                        dcc.Input(id="out-root-standardize", type="text", value=env_data_root, style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Button("Standardize now", id="standardize-btn", n_clicks=0),
                    ], style={"marginBottom": "8px"}),

                    html.Hr(),
                    html.H4("Rebuild cumulative (from raw)"),
                    html.Div([
                        html.Label("Output root"),
                        dcc.Input(id="out-root-rebuild", type="text", value=env_data_root, style={"width": "100%"}),
                        html.Br(), html.Br(),
                        html.Label("Weeks (space-separated, leave empty for ALL)"),
                        dcc.Input(id="rebuild-weeks", type="text", value="", style={"width": "100%"}),
                        html.Br(), html.Br(),
                        dcc.Checklist(
                            id="rebuild-options",
                            options=[
                                {"label": "Write Parquet", "value": "parquet"},
                                {"label": "GZip CSV", "value": "gzip"},
                            ],
                            value=[]
                        ),
                        html.Br(),
                        html.Button("Rebuild cumulative (from raw)", id="rebuild-btn", n_clicks=0),
                        html.Div(style={"opacity":0.7, "marginTop":"6px"},
                                 children="(Use rebuild if you suspect structural drift; it re-derives cumulative and writes Parquet/CSV.)"),
                    ]),
                ]),
            ]),

            # Restart control (optional)
            html.Div([
                html.Button("Restart App", id="restart-btn", n_clicks=0,
                            title="Shutdown & restart the server to pick up code/data changes (dev)"),
                html.Span("  (not required for data refresh)", style={"marginLeft": "8px", "opacity": 0.7}),
            ], style={"marginTop": "8px"}),

            html.Hr(),
            html.Pre(id="status", style={
                "whiteSpace": "pre-wrap",
                "background": "rgba(255,255,255,0.06)",
                "padding": "12px",
                "borderRadius": "8px",
                "border": "1px solid rgba(255,255,255,0.12)"
            }),
        ]
    )

    # -------------------- callbacks --------------------

    # Fetch by ID (optional run & replace/append)
    @app.callback(
        Output("status", "children"),
        Output("last-game", "data"),
        Input("fetch-btn", "n_clicks"),
        State("fetch-id", "value"),
        State("fetch-url", "value"),
        State("fetch-raw-root", "value"),
        State("fetch-week", "value"),
        State("fetch-then-run", "value"),
        State("fetch-out-root", "value"),
        prevent_initial_call=True,
    )
    def _fetch_then_optional_run(n_clicks, gid_str, url_tmpl, raw_root_str, week_str, opts, out_root_str):
        gid = (gid_str or "").strip()
        week = (week_str or "").strip() or "adhoc"
        if not gid:
            return "Please provide a Game ID.", None

        raw_root = _as_path(raw_root_str) or Path("raw_json")
        raw_root.mkdir(parents=True, exist_ok=True)

        try:
            saved = _fetch_game_json(gid, url_tmpl, raw_root, week)
        except Exception as e:
            return f"[FETCH ERROR] {type(e).__name__}: {e}", None

        msg = [f"[FETCH OK] saved → {saved}"]
        last_payload = None

        if "run" in (opts or []):
            if _IMPORT_ERROR:
                msg.append(f"Cannot run extractor: import error → {_IMPORT_ERROR}")
            else:
                out_root = _normalize_out_root(out_root_str)
                out_game_dir = out_root / week / Path(saved).stem
                try:
                    meta = process_one_game(
                        src_json=saved,
                        out_dir=out_game_dir,
                        override_csv=None,
                        week_name=week,
                        validate=False,
                    )
                    msg.append(f"[RUN OK] {getattr(meta, 'label', 'game')} → {out_game_dir}")
                    last_payload = {"out_root": str(out_root), "week": week, "game_id": str(Path(saved).stem)}
                    # precedence: replace > append if both ticked
                    if "replace" in (opts or []):
                        msg.append("\n[Replacing this game in cumulative…]")
                        msg.append(_replace_game_in_cumulative(out_root, week, Path(saved).stem))
                    elif "append" in (opts or []):
                        msg.append("\n[Appending this game to cumulative (auto-dedupe)…]")
                        msg.append(_append_game_to_cumulative(out_root, week, Path(saved).stem))
                except Exception as e:
                    msg.append(f"[RUN ERROR] {type(e).__name__}: {e}")

        return "\n".join(msg), last_payload

    # Single game run
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Output("last-game", "data", allow_duplicate=True),
        Input("run-game", "n_clicks"),
        State("src-json", "value"),
        State("week-name", "value"),
        State("override-csv", "value"),
        State("out-root-game", "value"),
        State("after-run", "value"),
        prevent_initial_call=True,
    )
    def _run_single_game(n_clicks, src_json_str, week_name_str, override_str, out_root_str, after_opts):
        if _IMPORT_ERROR:
            return f"Cannot run: extractor import error → {_IMPORT_ERROR}", None

        src = _as_path(src_json_str)
        if not _safe_exists(src):
            return f"Error: game JSON not found → {_human(src)}", None

        out_root = _normalize_out_root(out_root_str)
        week_name = (week_name_str or "").strip() or src.parent.name or "adhoc"
        ov = _as_path(override_str) if override_str else None

        out_game_dir = out_root / week_name / src.stem
        try:
            meta = process_one_game(
                src_json=src,
                out_dir=out_game_dir,
                override_csv=ov,
                week_name=week_name,
                validate=False,
            )
            msg = [f"[OK] {getattr(meta, 'label', 'game')}",
                   f"Week: {getattr(meta, 'week', week_name)}",
                   f"Game ID: {getattr(meta, 'game_id', src.stem)}",
                   f"Wrote: {out_game_dir}"]
        except Exception as e:
            return f"Extraction failed.\nsrc={_human(src)}\nout={_human(out_game_dir)}\nerror={type(e).__name__}: {e}", None

        last_payload = {"out_root": str(out_root), "week": week_name, "game_id": src.stem}
        if "replace" in (after_opts or []):
            msg.append("\n[Replacing this game in cumulative…]")
            msg.append(_replace_game_in_cumulative(out_root, week_name, src.stem))
        elif "append" in (after_opts or []):
            msg.append("\n[Appending this game to cumulative (auto-dedupe)…]")
            msg.append(_append_game_to_cumulative(out_root, week_name, src.stem))

        return "\n".join(msg), last_payload

    # Whole week run
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("run-week", "n_clicks"),
        State("week-dir", "value"),
        State("override-csv-week", "value"),
        State("out-root-week", "value"),
        State("rebuild-after-week", "value"),
        prevent_initial_call=True,
    )
    def _run_whole_week(n_clicks, week_dir_str, override_str, out_root_str, rebuild_opts):
        if _IMPORT_ERROR:
            return f"Cannot run: extractor import error → {_IMPORT_ERROR}"

        week_dir = _as_path(week_dir_str)
        if not _safe_exists(week_dir) or not week_dir.is_dir():
            return f"Error: week folder not found or not a directory → {_human(week_dir)}"

        out_root = _normalize_out_root(out_root_str)
        ov = _as_path(override_str) if override_str else None

        try:
            out_week_dir = process_week(
                week_json_dir=week_dir,
                out_root=out_root,
                override_csv=ov,
                validate=False,
            )
            msg = [f"[OK] Wrote week outputs: {out_week_dir}"]
        except Exception as e:
            return f"Week extraction failed.\nweek={_human(week_dir)}\nout_root={_human(out_root)}\nerror={type(e).__name__}: {e}"

        if "rebuild" in (rebuild_opts or []):
            wk = week_dir.name
            msg.append("\n[Rebuilding cumulative for this week…]")
            msg.append(_rebuild_cumulative(out_root, weeks=[wk], parquet=False, gzip_csv=False))

        return "\n".join(msg)

    # Compile: replace explicit game
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("replace-game-btn", "n_clicks"),
        State("out-root-replace", "value"),
        State("replace-week", "value"),
        State("replace-game-id", "value"),
        prevent_initial_call=True,
    )
    def _replace_explicit(n_clicks, out_root_str, week_str, gid_str):
        out_root = _normalize_out_root(out_root_str)
        week = (week_str or "").strip()
        gid = (gid_str or "").strip()
        if not week or not gid:
            return "Please provide both Week and Game ID to replace."
        return _replace_game_in_cumulative(out_root, week, gid)

    # Compile: replace last-run game
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("replace-last-btn", "n_clicks"),
        State("last-game", "data"),
        prevent_initial_call=True,
    )
    def _replace_last(n_clicks, last_game):
        if not last_game:
            return "No last-run game stored yet. Run a fetch or single game first."
        out_root = Path(last_game["out_root"])
        week = last_game["week"]
        gid = last_game["game_id"]
        return _replace_game_in_cumulative(out_root, week, gid)

    # Compile: append explicit game
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("append-game-btn", "n_clicks"),
        State("out-root-append", "value"),
        State("append-week", "value"),
        State("append-game-id", "value"),
        prevent_initial_call=True,
    )
    def _append_explicit(n_clicks, out_root_str, week_str, gid_str):
        out_root = _normalize_out_root(out_root_str)
        week = (week_str or "").strip()
        gid = (gid_str or "").strip()
        if not week or not gid:
            return "Please provide both Week and Game ID to append."
        return _append_game_to_cumulative(out_root, week, gid)

    # Compile: append last-run game
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("append-last-btn", "n_clicks"),
        State("last-game", "data"),
        prevent_initial_call=True,
    )
    def _append_last(n_clicks, last_game):
        if not last_game:
            return "No last-run game stored yet. Run a fetch or single game first."
        out_root = Path(last_game["out_root"])
        week = last_game["week"]
        gid = last_game["game_id"]
        return _append_game_to_cumulative(out_root, week, gid)

    # Compile: remove/purge game
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("remove-btn", "n_clicks"),
        State("out-root-remove", "value"),
        State("remove-game-id", "value"),
        State("remove-week", "value"),
        State("remove-strict", "value"),
        prevent_initial_call=True,
    )
    def _remove_handler(n_clicks, out_root_str, gid_str, week_str, flags):
        out_root = _normalize_out_root(out_root_str)
        gid = (gid_str or "").strip()
        if not gid:
            return "Please provide Game ID to remove."
        strict = "strict" in (flags or [])
        week = (week_str or "").strip() if strict else None
        return _remove_game_from_cumulative(out_root, gid, week, strict)

    # Compile: dedupe cumulative
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("dedupe-btn", "n_clicks"),
        State("out-root-dedupe", "value"),
        prevent_initial_call=True,
    )
    def _dedupe_now(n_clicks, out_root_str):
        out_root = _normalize_out_root(out_root_str)
        dest_dir = out_root / "_cumulative"
        players_all, gz_players = _read_existing_table(dest_dir, "players_all")
        pbp_all,     gz_pbp     = _read_existing_table(dest_dir, "pbp_all")
        idx_all,     gz_idx     = _read_existing_table(dest_dir, "games_index")

        players_all, pr = _dedupe_players(players_all)
        pbp_all,     br = _dedupe_pbp(pbp_all)
        idx_all,     ir = _dedupe_games_index(idx_all)

        p_csv = _write_table_atomic(players_all, dest_dir, "players_all", gz_players)
        b_csv = _write_table_atomic(pbp_all,     dest_dir, "pbp_all",     gz_pbp)
        i_csv = _write_table_atomic(idx_all,     dest_dir, "games_index", gz_idx)

        _ = _parquet_write(players_all, dest_dir, "players_all")
        _ = _parquet_write(pbp_all,     dest_dir, "pbp_all")
        _ = _parquet_write(idx_all,     dest_dir, "games_index")

        _bump_data_version(out_root)

        return (
            "[DEDUPE OK]\n"
            f"out_root: {out_root}\n"
            f"removed duplicates → players: {pr}, pbp: {br}, games_index: {ir}\n"
            f"players_all → {p_csv}\n"
            f"pbp_all     → {b_csv}\n"
            f"games_index → {i_csv}"
        )

    # Compile: standardize cumulative
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("standardize-btn", "n_clicks"),
        State("out-root-standardize", "value"),
        prevent_initial_call=True,
    )
    def _standardize_now(n_clicks, out_root_str):
        out_root = _normalize_out_root(out_root_str)
        return _standardize_cumulative_now(out_root)

    # Compile: rebuild
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("rebuild-btn", "n_clicks"),
        State("out-root-rebuild", "value"),
        State("rebuild-weeks", "value"),
        State("rebuild-options", "value"),
        prevent_initial_call=True,
    )
    def _rebuild_handler(n_clicks, out_root_str, weeks_str, opts):
        out_root = _normalize_out_root(out_root_str)
        weeks = [w for w in (weeks_str or "").split() if w.strip()] or None
        parquet = "parquet" in (opts or [])
        gzip_csv = "gzip" in (opts or [])
        return _rebuild_cumulative(out_root, weeks=weeks, parquet=parquet, gzip_csv=gzip_csv)

    # Client-side restart (optional)
    app.clientside_callback(
        """
        function(n, token) {
            if (!n) { return window.dash_clientside.no_update; }
            const headers = {};
            if (token) headers['X-Restart-Token'] = token;
            try {
              fetch('/__restart', {method:'POST', headers})
                .then(() => { setTimeout(() => { location.reload(); }, 1500); })
                .catch(() => { setTimeout(() => { location.reload(); }, 2000); });
            } catch (e) {
              setTimeout(() => { location.reload(); }, 2000);
            }
            return "[RESTART] Server is restarting… If the page doesn't reload, refresh manually.";
        }
        """,
        Output("status", "children", allow_duplicate=True),
        Input("restart-btn", "n_clicks"),
        State("restart-token", "data"),
    )

    return app
