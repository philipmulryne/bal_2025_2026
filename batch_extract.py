#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import sys

def run_one(json_path: Path, out_root: Path, week: str, python_exe: str) -> int:
    """Invoke the existing extractor.py CLI for a single JSON file."""
    cmd = [
        python_exe, "extractor.py",
        "--src", str(json_path),
        "--out-root", str(out_root),
        "--week", week,
    ]
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode

def collect_files(src: Path, pattern: str) -> List[Path]:
    if src.is_file() and src.suffix.lower() == ".json":
        return [src]
    files = sorted(src.glob(pattern))
    return [p for p in files if p.is_file() and p.suffix.lower() == ".json"]

def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for extractor.py over a folder or a single file."
    )
    parser.add_argument("--src", required=True,
                        help="Path to a JSON file or a folder containing JSON files (e.g., week02/).")
    parser.add_argument("--out-root", required=True,
                        help="Output root directory (same as extractor).")
    parser.add_argument("--week", required=True,
                        help="Week label to pass through (e.g., week02).")
    parser.add_argument("--glob", default="*.json",
                        help="Glob for files inside a folder (default: *.json).")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable to use (default: current interpreter).")
    args = parser.parse_args()

    src = Path(args.src)
    out_root = Path(args.out_root)
    files = collect_files(src, args.glob)

    if not files:
        print(f"[WARN] No JSON files found under {src} matching {args.glob}.", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Found {len(files)} file(s). Starting batch…")
    failures: List[Path] = []
    for f in files:
        rc = run_one(f, out_root, args.week, args.python)
        if rc != 0:
            print(f"[ERR ] Non-zero exit on {f} (rc={rc})", file=sys.stderr)
            failures.append(f)

    if failures:
        print(f"[DONE] Completed with {len(failures)} failure(s).", file=sys.stderr)
        for ff in failures:
            print(f"  - {ff}", file=sys.stderr)
        sys.exit(1)

    print("[DONE] All files processed successfully.")

if __name__ == "__main__":
    main()
