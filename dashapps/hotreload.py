# dashapps/common/hotreload.py
from __future__ import annotations
import os, threading
from pathlib import Path
from typing import Optional, Sequence, Tuple
import pandas as pd

_LOCK = threading.Lock()
# Simple in-proc cache: path -> (mtime, DataFrame)
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}

def _pick_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p and p.exists():
            return p
    return None

def read_csv_hot(candidates: Sequence[Path], **kwargs) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Return (df, used_path). Re-reads only when the file's mtime changed.
    Works with .csv and .csv.gz (pandas infers compression).
    """
    path = _pick_existing(candidates)
    if not path:
        return None, None
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except Exception:
        return None, path
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1], path
        df = pd.read_csv(path, **kwargs)
        _CACHE[key] = (mtime, df)
        return df, path

def clear_cache():
    with _LOCK:
        _CACHE.clear()

def version_file(out_root: Path) -> Path:
    return (out_root / "_cumulative" / ".data_version").resolve()

def get_version(out_root: Path) -> float:
    vf = version_file(out_root)
    try:
        return vf.stat().st_mtime if vf.exists() else 0.0
    except Exception:
        return 0.0
