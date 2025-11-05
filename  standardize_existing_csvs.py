# standardize_existing_csvs.py
from pathlib import Path
import pandas as pd
import re, unicodedata, shutil

OUT_ROOT = Path("out_data")  # set to your OUT_ROOT
DEST = OUT_ROOT / "_cumulative"
ALIASES = OUT_ROOT / "_config" / "team_aliases.csv"

def _strip_accents(s): 
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
def _norm_key(s): 
    return re.sub(r"[^A-Z0-9]+", "", _strip_accents(s).upper())

def load_aliases():
    m = {}
    if ALIASES.exists():
        df = pd.read_csv(ALIASES)
        for _, r in df.iterrows():
            a, c = str(r["alias"]).strip(), str(r["canonical"]).strip()
            if a and c: m[_norm_key(a)] = c
    return m

TEAMISH = ("team","home","away","opponent","opp")

def standardize(df, alias_map):
    if df is None or df.empty: return df, 0
    df = df.copy()
    cols = [c for c in df.columns 
            if any(t in c.lower() for t in TEAMISH)
            and not any(x in c.lower() for x in ("_id","id_","code","abbr","short","idx","num"))]
    changed = 0
    for c in cols:
        before = df[c].astype(str).tolist()
        df[c] = df[c].apply(lambda v: alias_map.get(_norm_key(v), v))
        after = df[c].astype(str).tolist()
        changed += sum(1 for b,a in zip(before,after) if b!=a)
    # re-key if a name-like col exists
    name_like = next((c for c in cols if "name" in c.lower() or c.lower()=="team"), None)
    if name_like:
        df["team_key"] = df[name_like].apply(lambda s: _norm_key(str(s)))
    return df, changed

def run():
    alias = load_aliases()
    for base in ("players_all","pbp_all","games_index"):
        for ext in (".csv.gz",".csv"):
            src = DEST / f"{base}{ext}"
            if not src.exists(): continue
            print(f"→ {src}")
            df = pd.read_csv(src)
            new, n = standardize(df, alias)
            if n == 0:
                print("   no changes.")
                continue
            # backup once
            bak = DEST / f"{base}.prestd.bak{ext}"
            if not bak.exists():
                shutil.copy2(src, bak)
            # write atomically
            tmp = DEST / f".{base}.tmp{ext}"
            comp = "gzip" if ext.endswith(".gz") else None
            new.to_csv(tmp, index=False, compression=comp)
            tmp.replace(src)
            print(f"   changed cells: {n}  (backup: {bak})")

if __name__ == "__main__":
    run()
