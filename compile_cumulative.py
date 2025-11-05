#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Constants (kept aligned with your extractor)
# -----------------------------------------------------------------------------
CSV_NAMES = {
    "pbp": "game_pbp_normalized.csv",
    "players": "game_players_with_positions.csv",
}

META_NAME = "meta.json"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def eprint(*msg):
    print(*msg, file=sys.stderr)

def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        eprint(f"[warn] could not read JSON {p}: {exc}")
        return None

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as exc:
        eprint(f"[warn] could not read CSV {p}: {exc}")
        return pd.DataFrame()

def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series([np.nan] * len(df))
    return df

def _union_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Row-bind with union-of-columns semantics."""
    if not dfs:
        return pd.DataFrame()
    # Collect superset of columns
    all_cols: List[str] = []
    seen = set()
    for d in dfs:
        for c in d.columns:
            if c not in seen:
                seen.add(c)
                all_cols.append(c)
    # Reindex each df to superset
    aligned = [d.reindex(columns=all_cols) for d in dfs]
    return pd.concat(aligned, ignore_index=True) if aligned else pd.DataFrame()

def _cast_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
@dataclass
class GamePath:
    week: str
    game_id: str
    game_dir: Path
    meta_path: Path
    pbp_path: Path
    players_path: Path

def discover_games(out_root: Path, weeks: Optional[List[str]] = None) -> List[GamePath]:
    out_root = out_root.resolve()
    if not out_root.exists():
        raise FileNotFoundError(f"out_root not found: {out_root}")

    week_dirs: List[Path]
    if weeks:
        week_dirs = [out_root / w for w in weeks]
    else:
        # Any subfolder under out_root is fair game (week01, week02, …)
        week_dirs = [p for p in out_root.iterdir() if p.is_dir()]

    games: List[GamePath] = []
    for wk in sorted(week_dirs):
        if not wk.exists() or not wk.is_dir():
            continue
        for gd in sorted(p for p in wk.iterdir() if p.is_dir()):
            gid = gd.name
            meta = gd / META_NAME
            pbp = gd / CSV_NAMES["pbp"]
            players = gd / CSV_NAMES["players"]
            # Require at least meta + one of the CSVs to count as a game
            if meta.exists() and (pbp.exists() or players.exists()):
                games.append(
                    GamePath(
                        week=wk.name,
                        game_id=gid,
                        game_dir=gd,
                        meta_path=meta,
                        pbp_path=pbp,
                        players_path=players,
                    )
                )
    return games

# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------
def compile_players(games: List[GamePath]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    req_cols = ["week", "game_id", "label"]  # we will append these

    for g in games:
        df = _safe_read_csv(g.players_path)
        if df.empty:
            continue
        meta = _read_json(g.meta_path) or {}
        label = meta.get("label")

        df["week"] = g.week
        df["game_id"] = g.game_id
        df["label"] = label

        # Strong but forgiving dtypes
        num_cols = ["team_no", "player_no", "shirtNumber"]
        df = _cast_numeric(df, num_cols)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged = _union_concat(dfs)

    # A helpful, stable column order: metadata first
    meta_first = ["week", "game_id", "label"]
    rest = [c for c in merged.columns if c not in meta_first]
    merged = merged[meta_first + rest]
    return merged


def compile_pbp(games: List[GamePath]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for g in games:
        df = _safe_read_csv(g.pbp_path)
        if df.empty:
            continue

        meta = _read_json(g.meta_path) or {}
        label = meta.get("label")

        df["week"] = g.week
        df["game_id"] = g.game_id
        df["label"] = label

        # Normalize typical numeric columns (present in your extractor output)
        numeric_guess = [
            "period", "actionNumber", "tno", "pno",
            "points", "possession_id", "possession_owner_tno",
        ]
        df = _cast_numeric(df, numeric_guess)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged = _union_concat(dfs)

    # Metadata first
    meta_first = ["week", "game_id", "label"]
    rest = [c for c in merged.columns if c not in meta_first]
    merged = merged[meta_first + rest]
    return merged


def compile_games_index(games: List[GamePath]) -> pd.DataFrame:
    rows = []
    for g in games:
        meta = _read_json(g.meta_path) or {}
        rows.append(
            {
                "week": g.week,
                "game_id": g.game_id,
                "label": meta.get("label"),
                "teams": " / ".join(meta.get("teams") or []),
                "meta_path": str(g.meta_path),
                "pbp_path": str(g.pbp_path) if g.pbp_path.exists() else None,
                "players_path": str(g.players_path) if g.players_path.exists() else None,
                "created_at": meta.get("created_at"),
                "source_json": meta.get("source_json"),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    dest_dir: Path,
    players_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    make_parquet: bool,
    gzip_csv: bool,
):
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _wcsv(df: pd.DataFrame, name: str):
        if df.empty:
            eprint(f"[info] {name}: no rows, skipping write.")
            return None
        suffix = ".csv.gz" if gzip_csv else ".csv"
        out = dest_dir / f"{name}{suffix}"
        df.to_csv(out, index=False, compression="gzip" if gzip_csv else None)
        eprint(f"[ok] wrote {out} ({len(df):,} rows)")
        return out

    def _wparq(df: pd.DataFrame, name: str):
        if not make_parquet or df.empty:
            return None
        out = dest_dir / f"{name}.parquet"
        try:
            df.to_parquet(out, index=False)
            eprint(f"[ok] wrote {out} ({len(df):,} rows)")
            return out
        except Exception as exc:
            eprint(f"[warn] parquet write failed for {name}: {exc}")
            return None

    players_out = _wcsv(players_df, "players_all")
    pbp_out = _wcsv(pbp_df, "pbp_all")
    idx_out = _wcsv(idx_df, "games_index")

    _ = _wparq(players_df, "players_all")
    _ = _wparq(pbp_df, "pbp_all")
    _ = _wparq(idx_df, "games_index")

    # Tiny manifest for provenance
    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "dest_dir": str(dest_dir),
        "counts": {
            "players_all": int(players_df.shape[0]),
            "pbp_all": int(pbp_df.shape[0]),
            "games_index": int(idx_df.shape[0]),
        },
        "csv_paths": {
            "players_all": str(players_out) if players_out else None,
            "pbp_all": str(pbp_out) if pbp_out else None,
            "games_index": str(idx_out) if idx_out else None,
        },
        "schemas": {
            "players_all": list(players_df.columns),
            "pbp_all": list(pbp_df.columns),
            "games_index": list(idx_df.columns),
        },
    }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    eprint(f"[ok] wrote {dest_dir / 'manifest.json'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compile cumulative CSV/Parquet files from per-game outputs (out_data/<week>/<game_id>/…)."
    )
    p.add_argument("--out-root", type=Path, default=Path("out_data"),
                   help="Root where extractor wrote per-game folders (default: ./out_data)")
    p.add_argument("--weeks", nargs="*", default=None,
                   help="Optional list of week folders to include (e.g., week01 week02). If omitted, include all weeks under out_root.")
    p.add_argument("--only", nargs="*", choices=["players", "pbp"], default=None,
                   help="Restrict compilation to certain tables (players, pbp). Default: build both.")
    p.add_argument("--dest", type=Path, default=None,
                   help="Destination folder for cumulative outputs (default: <out-root>/_cumulative)")
    p.add_argument("--parquet", action="store_true",
                   help="Also write Parquet versions.")
    p.add_argument("--gzip-csv", action="store_true",
                   help="Write compressed CSVs (*.csv.gz).")
    return p.parse_args()


def main():
    args = parse_args()
    dest_dir = args.dest or (args.out_root / "_cumulative")

    games = discover_games(args.out_root, args.weeks)
    if not games:
        eprint(f"[error] no games discovered under {args.out_root}. Did you run the extractor?")
        sys.exit(2)

    eprint(f"[info] discovered {len(games)} games under {args.out_root}")

    build_players = (args.only is None) or ("players" in args.only)
    build_pbp = (args.only is None) or ("pbp" in args.only)

    players_df = compile_players(games) if build_players else pd.DataFrame()
    pbp_df = compile_pbp(games) if build_pbp else pd.DataFrame()
    idx_df = compile_games_index(games)

    write_outputs(dest_dir, players_df, pbp_df, idx_df, make_parquet=args.parquet, gzip_csv=args.gzip_csv)


if __name__ == "__main__":
    main()
