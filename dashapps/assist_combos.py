# assist_combos.py (patched for robust layout on Dash>=2.15)
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.graph_objects as go

# NOTE: On some installs, "from .common import page_shell as PageShell" may import a MODULE
# rather than a callable. We keep this import but add runtime guards below to avoid
# returning None to app.layout.
from .common import page_shell as PageShell  # type: ignore


# -----------------------------------------------------------------------------
# Data discovery / loading
# -----------------------------------------------------------------------------
def _find_pbp_all() -> Optional[Path]:
    # Always check your definitive path first
    default_path = Path("/Users/uygarkaraca/JSON/BSL_2025_26/out_data/_cumulative/pbp_all.csv")
    if default_path.exists():
        return default_path

    # Then check environment variable
    env = os.environ.get("PBP_ALL")
    if env and Path(env).exists():
        return Path(env)

    # Then fall back to DATA_ROOT and other locations
    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    candidates = [
        data_root / "_cumulative" / "pbp_all.csv",
        data_root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Last resort: search recursively
    for c in data_root.rglob("pbp_all.csv"):
        return c
    return None



def _ensure_cols(df: pd.DataFrame, cols: List[str], fill=None) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


def _load_pbp_all() -> pd.DataFrame:
    src = _find_pbp_all()
    if not src or not src.exists():
        raise FileNotFoundError(
            "pbp_all.csv not found. Set PBP_ALL=/path/to/pbp_all.csv or place it under DATA_ROOT/_cumulative/."
        )

    df = pd.read_csv(src)

    # Ensure presence of key columns used later (avoid KeyError in options/UI)
    df = _ensure_cols(df, ["game_id", "week", "label"], fill=np.nan)
    df = _ensure_cols(
        df,
        [
            "player",
            "player_name_from_roster",
            "assist_player",
            "team_name",
            "ev_code",
            "subType",
            "period",
            "actionNumber",
            "success",
            "points",
        ],
        fill=np.nan,
    )

    # Coerce numerics
    for c in ["period", "actionNumber", "points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strings
    df["ev_code"] = df["ev_code"].fillna("").astype(str)
    df["subType"] = df["subType"].fillna("").astype(str)

    # Finisher fallback
    df["finisher"] = df["player"].fillna(df["player_name_from_roster"]).fillna("Unknown Player")

    # Shot type classification
    df["shot_type"] = np.where(
        df["ev_code"] == "P3",
        "3PT",
        np.where(df["ev_code"] == "P2", "2PT", np.where(df["ev_code"] == "FT", "FT", None)),
    )

    # Points fallback (safety)
    if "points" not in df.columns:
        df["points"] = np.where(
            df["ev_code"] == "P3",
            3,
            np.where(df["ev_code"] == "P2", 2, np.where(df["ev_code"] == "FT", 1, 0)),
        )

    # -------- FT assist inference (explicit ASS after FT > heuristic backtrack) --------
    df = _infer_ft_assists(df)

    return df


def _infer_ft_assists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate:
      - df['is_first_ft_made'] -> bool mask for FT 1of1 / 1of2 / 1of3 with success==1
      - df['ft_assist_player'] -> inferred passer using explicit 'ASS' row that comes RIGHT AFTER the FT (authoritative),
                                  with a short forward window as backup. If not found, fall back to previous
                                  heuristic (look back to last P2/P3 attempt/make by shooter).
      - df['assist_player_enriched'] -> native 'assist_player' for P2/P3; for FT-first rows,
                                        use the passer discovered above.
    """
    # normalize assist column and create enriched view
    df["assist_player"] = df["assist_player"].astype(str).replace({"nan": ""})
    df["assist_player_enriched"] = df["assist_player"]

    # case-insensitive subtype check
    sub_lower = df["subType"].astype(str).str.lower()
    df["is_first_ft_made"] = (
        (df["ev_code"] == "FT")
        & (sub_lower.isin(["1of1", "1of2", "1of3"]))
        & (pd.to_numeric(df["success"], errors="coerce") == 1)
    )

    # sort to reason about local neighborhoods
    sorted_df = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    sorted_df = sorted_df.sort_values(["game_id", "period", "actionNumber"], kind="mergesort").reset_index(drop=True)

    inferred: Dict[int, str] = {}

    # indices of FT-first made rows
    cand_idx = sorted_df.index[sorted_df["is_first_ft_made"] == True].tolist()

    # how far forward to look for 'ASS' before bailing out
    FWD_WINDOW = 5

    for pos in cand_idx:
        row = sorted_df.loc[pos]
        shooter = str(row["player"])
        team = str(row["team_name"])
        game = row["game_id"]

        found = None

        # ---------- 1) Prefer explicit ASS right after FT (authoritative) ----------
        if pos + 1 < len(sorted_df):
            nxt = sorted_df.loc[pos + 1]
            if nxt["game_id"] == game and str(nxt["team_name"]) == team and str(nxt["ev_code"]) == "ASS":
                candidates = [
                    str(nxt.get("player", "")).strip(),
                    str(nxt.get("assist_player", "")).strip(),
                    str(nxt.get("player_name_from_roster", "")).strip(),
                ]
                found = next((c for c in candidates if c), None)

        # Short forward scan until next scoring attempt (P2/P3/FT)
        if not found:
            limit = min(len(sorted_df), pos + 1 + FWD_WINDOW)
            for j in range(pos + 1, limit):
                rj = sorted_df.loc[j]
                if rj["game_id"] != game:
                    break
                code = str(rj["ev_code"])
                if code in ("P2", "P3", "FT"):
                    break
                if code == "ASS" and str(rj["team_name"]) == team:
                    candidates = [
                        str(rj.get("player", "")).strip(),
                        str(rj.get("assist_player", "")).strip(),
                        str(rj.get("player_name_from_roster", "")).strip(),
                    ]
                    tmp = next((c for c in candidates if c), None)
                    if tmp:
                        found = tmp
                        break

        # ---------- 2) Heuristic fallback ----------
        if not found:
            ft_kind = str(row["subType"]).lower()
            back_limit = max(0, pos - 8)

            if ft_kind == "1of1":
                for j in range(pos - 1, back_limit - 1, -1):
                    r = sorted_df.loc[j]
                    if r["game_id"] != game:
                        break
                    if (str(r["team_name"]) == team) and (str(r["player"]) == shooter) and (str(r["ev_code"]) in ("P2", "P3")):
                        if pd.to_numeric(r.get("success", 0), errors="coerce") == 1:
                            ap = str(r.get("assist_player", "")).strip()
                            if ap:
                                found = ap
                            break
            else:
                for j in range(pos - 1, back_limit - 1, -1):
                    r = sorted_df.loc[j]
                    if r["game_id"] != game:
                        break
                    if (str(r["team_name"]) == team) and (str(r["player"]) == shooter) and (str(r["ev_code"]) in ("P2", "P3")):
                        ap = str(r.get("assist_player", "")).strip()
                        if ap:
                            found = ap
                        break

        if found:
            inferred[int(row["_orig_idx"])] = found

    # Map back to original df
    df["ft_assist_player"] = ""
    if inferred:
        ix = pd.Index(inferred.keys())
        df.loc[ix, "ft_assist_player"] = pd.Series(inferred)

    # Enrich assist for leaderboards: for FT-first rows, use the discovered passer if present
    use_mask = (df["is_first_ft_made"]) & (df["ft_assist_player"].astype(str).str.len() > 0)
    df.loc[use_mask, "assist_player_enriched"] = df.loc[use_mask, "ft_assist_player"]

    return df


_DF_PBP: Optional[pd.DataFrame] = None

def _pbp() -> pd.DataFrame:
    global _DF_PBP
    if _DF_PBP is None:
        _DF_PBP = _load_pbp_all()
    return _DF_PBP


# -----------------------------------------------------------------------------
# Filtering & aggregation
# -----------------------------------------------------------------------------
def _filtered(
    df: pd.DataFrame,
    teams: List[str],
    games: List[str],
    periods: Tuple[int, int],
    event_shot_types: List[str],
) -> pd.DataFrame:
    """Combos view filter — ONLY 2PT/3PT assisted made FGs."""
    f = df.copy()

    mask = (
        (f["assist_player"].astype(str).str.len() > 0)
        & (f["ev_code"].isin(["P2", "P3"]))
        & (pd.to_numeric(f["success"], errors="coerce") == 1)
    )
    f = f.loc[mask]

    if teams:
        f = f[f["team_name"].isin(teams)]
    if games:
        f = f[f["game_id"].astype(str).isin(games)]
    if periods:
        lo, hi = periods
        per = pd.to_numeric(f["period"], errors="coerce")
        f = f[(per >= lo) & (per <= hi)]

    if event_shot_types:
        f = f[f["shot_type"].isin(event_shot_types)]  # "2PT" / "3PT" only for combos

    return f


def _aggregate_split(f: pd.DataFrame, weight_by_points: bool) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame()

    agg_map = {"actionNumber": "count", "points": "sum", "game_id": pd.Series.nunique}
    if "r" in f.columns:
        agg_map["r"] = "mean"
    if {"x", "y"}.issubset(f.columns):
        agg_map["x"] = "mean"; agg_map["y"] = "mean"

    g = (
        f.groupby(["team_name", "assist_player", "finisher", "shot_type"], dropna=False)
         .agg(agg_map)
         .rename(columns={"actionNumber": "makes", "game_id": "games"})
         .reset_index()
    )

    # Share within finisher × shot_type
    tot = g.groupby(["team_name", "finisher", "shot_type"])["makes"].transform("sum")
    g["finisher_share"] = np.where(tot > 0, g["makes"] / tot, np.nan)
    g["finisher_share_pct"] = (100.0 * g["finisher_share"]).round(1)
    g["value"] = g["points"] if weight_by_points else g["makes"]

    return g.sort_values(["team_name", "value", "makes"], ascending=[True, False, False])


def _aggregate_all(f: pd.DataFrame, weight_by_points: bool) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame()

    agg_map = {"actionNumber": "count", "points": "sum", "game_id": pd.Series.nunique}
    if "r" in f.columns:
        agg_map["r"] = "mean"
    if {"x", "y"}.issubset(f.columns):
        agg_map["x"] = "mean"; agg_map["y"] = "mean"

    g = (
        f.groupby(["team_name", "assist_player", "finisher"], dropna=False)
         .agg(agg_map)
         .rename(columns={"actionNumber": "makes", "game_id": "games"})
         .reset_index()
    )
    g["shot_type"] = "ALL"

    # Share within finisher across ALL (2PT+3PT)
    tot = g.groupby(["team_name", "finisher"])["makes"].transform("sum")
    g["finisher_share"] = np.where(tot > 0, g["makes"] / tot, np.nan)
    g["finisher_share_pct"] = (100.0 * g["finisher_share"]).round(1)
    g["value"] = g["points"] if weight_by_points else g["makes"]

    return g.sort_values(["team_name", "value", "makes"], ascending=[True, False, False])


# ---------------- Leaderboards (assist / receive, incl. FT-first) ----------------
def _dominant_team(series: pd.Series) -> Optional[str]:
    if series.empty:
        return None
    mode = series.mode(dropna=True)
    return None if mode.empty else mode.iloc[0]


def _leaderboard_base(
    f: pd.DataFrame, key_col: str, topn: int, search: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a leaderboard for either assisters (key_col='assist_player_enriched')
    or receivers (key_col='finisher') with 2PT / 3PT / FT columns and an ALL total.
    """
    if f.empty:
        return pd.DataFrame(columns=[key_col, "team", "ALL", "2PT", "3PT", "FT"])

    f = f[~f[key_col].isna()].copy()
    f[key_col] = f[key_col].astype(str)

    # shot-type counts
    g = (
        f.groupby([key_col, "shot_type"], dropna=False)["actionNumber"]
        .count()
        .rename("makes")
        .reset_index()
    )
    p = g.pivot_table(
        index=key_col, columns="shot_type", values="makes", aggfunc="sum", fill_value=0
    )

    for col in ["2PT", "3PT", "FT"]:
        if col not in p.columns:
            p[col] = 0

    p["ALL"] = p[["2PT", "3PT", "FT"]].sum(axis=1)
    p = p.sort_values("ALL", ascending=False)

    # Dominant team
    team_map = f.groupby(key_col)["team_name"].apply(_dominant_team).rename("team").to_frame()

    out = p.merge(team_map, left_index=True, right_index=True, how="left").reset_index()

    # search
    if search and search.strip():
        s = search.strip().lower()
        out = out[out[key_col].str.lower().str.contains(s, na=False)]

    out = out.sort_values(["ALL", "3PT", "2PT", "FT"], ascending=[False, False, False, False]).head(int(topn))
    out = out[[key_col, "team", "ALL", "2PT", "3PT", "FT"]]
    return out


def _filtered_for_leaders(
    df: pd.DataFrame,
    teams: List[str],
    games: List[str],
    periods: Tuple[int, int],
    include_codes: List[str],
) -> pd.DataFrame:
    """
    Leaderboards filter — allows 2PT/3PT assisted *made* + FT-first *made* (1of1/1of2/1of3).
    Uses 'assist_player_enriched' so FT rows can carry inferred passers (from explicit ASS or fallback).
    """
    f = df.copy()

    # Field goals: must be made and have a native assister
    fg_mask = (
        f["ev_code"].isin(["P2", "P3"])
        & (pd.to_numeric(f["success"], errors="coerce") == 1)
        & (f["assist_player"].astype(str).str.len() > 0)
    )

    # First FT: must be first-of and made; assister may be enriched from explicit ASS
    ft_mask = f["is_first_ft_made"]

    mask = False
    if "P2" in include_codes or "P3" in include_codes:
        mask = mask | fg_mask
    if "FT" in include_codes:
        mask = mask | ft_mask

    f = f.loc[mask].copy()

    if teams:
        f = f[f["team_name"].isin(teams)]
    if games:
        f = f[f["game_id"].astype(str).isin(games)]
    if periods:
        lo, hi = periods
        per = pd.to_numeric(f["period"], errors="coerce")
        f = f[(per >= lo) & (per <= hi)]

    # Keep only 2PT/3PT/FT labels
    f = f[f["shot_type"].isin(["2PT", "3PT", "FT"])]

    return f


# -----------------------------------------------------------------------------
# NEW: Assist network robustness (TEAM)
# -----------------------------------------------------------------------------
def _safe_entropy(values: np.ndarray) -> float:
    """Normalized entropy in [0,1]; 0 if degenerate/empty."""
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if v.size <= 1:
        return 0.0
    p = v / v.sum()
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


def _team_robustness(f: pd.DataFrame, use_points: bool) -> pd.DataFrame:
    """
    Per-team robustness (0–100). Components (0–1):
      Diversity = mean(sender_entropy, receiver_entropy)
      PairEntropy = entropy over (assister,receiver) pairs
      Density = unique_pairs / (unique_assisters * unique_receivers)
      Reciprocity = share of unordered dyads with both directions (A→B & B→A)
      AntiConcentration = 1 - top_assister_share
    Score = 0.30*Diversity + 0.20*PairEntropy + 0.15*Density + 0.20*Reciprocity + 0.15*AntiConc
    """
    if f.empty:
        return pd.DataFrame()

    df = f.copy()
    df["assister"] = df["assist_player_enriched"].fillna("").astype(str).str.strip()
    df["receiver"] = df["finisher"].fillna("").astype(str).str.strip()
    df = df[(df["assister"] != "") & (df["receiver"] != "")]
    if df.empty:
        return pd.DataFrame()

    w = (pd.to_numeric(df["points"], errors="coerce").fillna(0.0) if use_points
         else pd.Series(1.0, index=df.index))
    df["_w"] = w

    out = []
    for team, g in df.groupby("team_name", dropna=False):
        a_counts = g.groupby("assister")["_w"].sum().sort_values(ascending=False)
        r_counts = g.groupby("receiver")["_w"].sum().sort_values(ascending=False)
        pr_counts = g.groupby(["assister", "receiver"])["_w"].sum().sort_values(ascending=False)

        A, R, E = a_counts.size, r_counts.size, pr_counts.shape[0]
        density = float(E / (A * R)) if (A > 0 and R > 0) else 0.0

        Hs = _safe_entropy(a_counts.values)
        Hr = _safe_entropy(r_counts.values)
        Hp = _safe_entropy(pr_counts.values)
        diversity = 0.5 * (Hs + Hr)

        total_w = float(a_counts.sum()) if a_counts.sum() > 0 else 1.0
        top_ass_share = float(a_counts.iloc[0] / total_w) if A > 0 else 0.0
        anti_conc = 1.0 - top_ass_share

        pairs = set((a, b) for a, b in pr_counts.index if a != b)
        dyads = set(frozenset({a, b}) for (a, b) in pairs)
        mutual = 0
        mutual_weight = 0.0
        for d in dyads:
            if len(d) != 2:
                continue
            a, b = tuple(d)
            ab = pr_counts.get((a, b), 0.0)
            ba = pr_counts.get((b, a), 0.0)
            if ab > 0 and ba > 0:
                mutual += 1
                mutual_weight += min(ab, ba)
        reciprocity = float(mutual / len(dyads)) if dyads else 0.0
        partner_share = float(mutual_weight / total_w) if total_w > 0 else 0.0

        score = (0.30 * diversity + 0.20 * Hp + 0.15 * density + 0.20 * reciprocity + 0.15 * anti_conc)

        out.append({
            "team_name": team,
            "robustness": round(100.0 * score, 1),
            "assisters": int(A),
            "receivers": int(R),
            "pairs": int(E),
            "density_pct": round(100.0 * density, 1),
            "reciprocity_pct": round(100.0 * reciprocity, 1),
            "partner_share_pct": round(100.0 * partner_share, 1),
            "top_assister_share_pct": round(100.0 * top_ass_share, 1),
            "sender_entropy_pct": round(100.0 * Hs, 1),
            "receiver_entropy_pct": round(100.0 * Hr, 1),
            "pair_entropy_pct": round(100.0 * Hp, 1),
        })

    res = pd.DataFrame(out)
    if res.empty:
        return res
    return res.sort_values(["robustness", "density_pct", "reciprocity_pct"], ascending=[False, False, False]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def _sankey(df: pd.DataFrame, team: Optional[str], top_n: int, use_points: bool) -> go.Figure:
    if df.empty:
        return go.Figure()

    if team:
        df = df[df["team_name"] == team]
    df = df.sort_values("value", ascending=False).head(top_n)

    passers = list(pd.unique(df["assist_player"]))
    finishers = list(pd.unique(df["finisher"]))
    p_index = {p: i for i, p in enumerate(passers)}
    f_index = {f: i + len(passers) for i, f in enumerate(finishers)}

    src, tgt, val, lab = [], [], [], []
    for _, r in df.iterrows():
        src.append(p_index.get(r["assist_player"], 0))
        tgt.append(f_index.get(r["finisher"], 0))
        val.append(float(r["points"] if use_points else r["makes"]))
        lab.append(f'{r["assist_player"]} → {r["finisher"]} ({r["shot_type"]})')

    fig = go.Figure([go.Sankey(
        node=dict(label=passers + finishers, pad=16, thickness=16),
        link=dict(source=src, target=tgt, value=val, label=lab),
        arrangement="snap",
    )])
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Sankey: Top {top_n} Assist Combos" + (f" – {team}" if team else "")
    )
    return fig


def _bars(df: pd.DataFrame, team: Optional[str], top_n: int, use_points: bool) -> go.Figure:
    if df.empty:
        return go.Figure()

    if team:
        df = df[df["team_name"] == team]
    df = df.sort_values("value", ascending=False).head(top_n)

    labels = df.apply(lambda r: f'{r["assist_player"]} → {r["finisher"]} ({r["shot_type"]})', axis=1)
    y = (df["points"] if use_points else df["makes"]).astype(float)

    fig = go.Figure(go.Bar(x=y, y=labels, orientation="h"))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Top {top_n} Assist Combos (by {'Points' if use_points else 'Makes'})" + (f" – {team}" if team else ""),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# -----------------------------------------------------------------------------
# Dash factory (with sidebar shell) — hardened against None layouts
# -----------------------------------------------------------------------------

def _safe_pageshell(shell_obj, content, nav_items, base_pathname):
    """Try various ways to render the shell and never return None."""
    try:
        # Most common: callable(PageShell)
        if callable(shell_obj):
            out = shell_obj(content, nav_items, current_endpoint=base_pathname)
            if out is not None:
                return out
        # If imported as a module with an attribute
        if hasattr(shell_obj, "page_shell") and callable(shell_obj.page_shell):
            out = shell_obj.page_shell(content, nav_items, current_endpoint=base_pathname)
            if out is not None:
                return out
        if hasattr(shell_obj, "build") and callable(shell_obj.build):
            out = shell_obj.build(content, nav_items, current_endpoint=base_pathname)
            if out is not None:
                return out
    except Exception as e:
        return html.Div([
            html.H2("Assist Combos"),
            html.Div("Page shell error; using fallback."),
            html.Pre(str(e)),
            content,
        ])

    # Fallback layout if shell returned None
    return html.Div([
        html.H2("Assist Combos"),
        html.Div("No page shell available; using minimal wrapper."),
        content,
    ])


def create_dash_assist_combos(server, base_pathname: str = "/assist_combos/") -> Dash:
    app = dash.Dash(
        __name__,
        server=server,
        # 'url_base_pathname' still works, but if your Dash is very new, you can switch to:
        # requests_pathname_prefix=base_pathname,
        url_base_pathname=base_pathname,
        suppress_callback_exceptions=True,
        title="Assist Combos",
    )

    # Set a placeholder *immediately* to satisfy Dash>=2.15 validation
    app.layout = html.Div("Initializing Assist Combos…")

    try:
        df = _pbp()

        _games_view = df[["week", "game_id", "label"]].copy()
        _games_view["week"] = _games_view["week"].fillna("")
        _games_view["game_id"] = _games_view["game_id"].astype(str).fillna("")
        _games_view["label"] = _games_view["label"].fillna("")

        team_opts = [{"label": t, "value": t} for t in sorted([t for t in df["team_name"].dropna().unique()])]
        game_opts = [{"label": f'{w} • {gid} • {lab}', "value": str(gid)}
                     for w, gid, lab in _games_view.drop_duplicates().sort_values(["week", "game_id"]).itertuples(index=False)]

        period_max = int(pd.to_numeric(df["period"], errors="coerce").max() or 4)

        # --- Inner content (right column) ---
        content = html.Div(children=[
            html.H2("Assist Combos & Leaderboards", style={"marginBottom": "8px"}),
            html.P(
                "FG combos use assisted 2PT/3PT makes. Leaderboards include FT-first made (1of1/1of2/1of3) with inferred passers.",
                style={"color": "#555"},
            ),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "12px"}, children=[
                html.Div(children=[html.Label("Team(s)"), dcc.Dropdown(id="ac-teams", options=team_opts, multi=True, placeholder="Select teams")]),
                html.Div(children=[html.Label("Game(s)"), dcc.Dropdown(id="ac-games", options=game_opts, multi=True, placeholder="Filter by game")]),
                html.Div(children=[html.Label("Shot Type"),
                    dcc.Checklist(id="ac-shot-types",
                        options=[{"label": "All", "value": "ALL"}, {"label": "2PT", "value": "2PT"}, {"label": "3PT", "value": "3PT"}, {"label": "FT", "value": "FT"}],
                        value=["ALL"], inputStyle={"marginRight": "6px", "marginLeft": "12px"})]),
                html.Div(children=[html.Label("Periods"),
                    dcc.RangeSlider(id="ac-periods", min=1, max=period_max, step=1, value=[1, period_max],
                                    marks={i: str(i) for i in range(1, period_max + 1)})]),
            ]),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "12px", "marginTop": "6px"}, children=[
                html.Div(children=[html.Label("Min makes (combos)"),
                    dcc.Slider(id="ac-min-makes", min=1, max=10, step=1, value=2, marks={i: str(i) for i in range(1, 11)})]),
                html.Div(children=[html.Label("Rank by (combos)"),
                    dcc.RadioItems(id="ac-rank-by",
                        options=[{"label": "Makes", "value": "makes"}, {"label": "Points (2/3 weighted)", "value": "points"}],
                        value="points", inputStyle={"marginRight": "6px", "marginLeft": "12px"})]),
                html.Div(children=[html.Label("Top N (charts)"),
                    dcc.Slider(id="ac-topn", min=5, max=30, step=1, value=15, marks={5: "5", 10: "10", 15: "15", 20: "20", 25: "25", 30: "30"})]),
                html.Div(children=[html.Label("Top N (leaders)"),
                   dcc.Slider(
                        id="ac-topn-leaders",
                        min=10, max=100, step=5, value=50,
                        marks={10: "10", 25: "25", 50: "50", 75: "75", 100: "100"}
                   ),]),
            ]),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "6px"}, children=[
                html.Div(children=[html.Label("Search (passer/finisher/leaderboards)"),
                    dcc.Input(id="ac-search", type="text", placeholder="e.g., Hale or Whittaker", style={"width": "100%"})]),
                html.Div(),
            ]),

            html.Hr(),
            html.Div(id="ac-kpis", style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(140px, 1fr))", "gap": "12px"}),

            html.H3("Top Assist Combos (table)", style={"marginTop": "16px"}),

            dash_table.DataTable(
                id="ac-table",
                columns=[
                    {"name": "Team", "id": "team_name"},
                    {"name": "Passer", "id": "assist_player"},
                    {"name": "Finisher", "id": "finisher"},
                    {"name": "Shot", "id": "shot_type"},
                    {"name": "Makes", "id": "makes", "type": "numeric"},
                    {"name": "Points", "id": "points", "type": "numeric"},
                    {"name": "% of Finisher’s Assisted Makes", "id": "finisher_share_pct", "type": "numeric"},
                    {"name": "Games", "id": "games", "type": "numeric"},
                ],
                data=[], sort_action="native", filter_action="native", page_size=20,
                style_table={"overflowX": "auto"}, style_header={"fontWeight": "600"},
                style_cell={"padding": "6px 8px"},
                export_format="csv", export_headers="display",
            ),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "18px"}, children=[
                dcc.Graph(id="ac-sankey"),
                dcc.Graph(id="ac-bars"),
            ]),

            html.Hr(),
            html.H3("Leaderboards"),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
                html.Div(children=[
                    html.H4("Top Assisters (ALL / 2PT / 3PT / FT)"),
                    dash_table.DataTable(
                        id="ac-assist-leaders",
                        columns=[
                            {"name": "Assister", "id": "assist_player_enriched"},
                            {"name": "Team", "id": "team"},
                            {"name": "ALL", "id": "ALL", "type": "numeric"},
                            {"name": "2PT", "id": "2PT", "type": "numeric"},
                            {"name": "3PT", "id": "3PT", "type": "numeric"},
                            {"name": "FT", "id": "FT", "type": "numeric"},
                        ],
                        data=[], sort_action="native", page_size=25,
                        style_table={"overflowX": "auto"}, style_header={"fontWeight": "600"},
                        style_cell={"padding": "6px 8px"},
                        export_format="csv", export_headers="display",
                    ),
                ]),
                html.Div(children=[
                    html.H4("Top Receivers (ALL / 2PT / 3PT / FT)"),
                    dash_table.DataTable(
                        id="ac-receiver-leaders",
                        columns=[
                            {"name": "Receiver", "id": "finisher"},
                            {"name": "Team", "id": "team"},
                            {"name": "ALL", "id": "ALL", "type": "numeric"},
                            {"name": "2PT", "id": "2PT", "type": "numeric"},
                            {"name": "3PT", "id": "3PT", "type": "numeric"},
                            {"name": "FT", "id": "FT", "type": "numeric"},
                        ],
                        data=[], sort_action="native", page_size=25,
                        style_table={"overflowX": "auto"}, style_header={"fontWeight": "600"},
                        style_cell={"padding": "6px 8px"},
                        export_format="csv", export_headers="display",
                    ),
                ]),
            ]),

            html.Hr(),
            html.H3("Team Assist Robustness (0–100)"),
            dash_table.DataTable(
                id="ac-team-robust",
                columns=[
                    {"name": "Team", "id": "team_name"},
                    {"name": "Robustness", "id": "robustness", "type": "numeric"},
                    {"name": "Assisters", "id": "assisters", "type": "numeric"},
                    {"name": "Receivers", "id": "receivers", "type": "numeric"},
                    {"name": "Pairs", "id": "pairs", "type": "numeric"},
                    {"name": "Density %", "id": "density_pct", "type": "numeric"},
                    {"name": "Reciprocity %", "id": "reciprocity_pct", "type": "numeric"},
                    {"name": "Partner Share %", "id": "partner_share_pct", "type": "numeric"},
                    {"name": "Top Assister Share %", "id": "top_assister_share_pct", "type": "numeric"},
                    {"name": "Sender Entropy %", "id": "sender_entropy_pct", "type": "numeric"},
                    {"name": "Receiver Entropy %", "id": "receiver_entropy_pct", "type": "numeric"},
                    {"name": "Pair Entropy %", "id": "pair_entropy_pct", "type": "numeric"},
                ],
                data=[], sort_action="native", page_size=25,
                style_table={"overflowX": "auto"}, style_header={"fontWeight": "600"},
                style_cell={"padding": "6px 8px"},
                export_format="csv", export_headers="display",
            ),

            html.Div(style={"marginTop": "12px", "fontSize": "12px", "color": "#666"}, children=[
                html.Span("Combos tables/charts use FG assists only (2PT & 3PT). Leaderboards: ALL = 2PT+3PT+FT-first."),
            ])
        ])

        nav_items = server.config.get("NAV", [])
        # Always end with a non-None layout
        app.layout = _safe_pageshell(PageShell, content, nav_items, base_pathname)

        # -------------------- Callbacks --------------------
        @app.callback(
            Output("ac-table", "data"),
            Output("ac-kpis", "children"),
            Output("ac-sankey", "figure"),
            Output("ac-bars", "figure"),
            Output("ac-assist-leaders", "data"),
            Output("ac-receiver-leaders", "data"),
            Output("ac-team-robust", "data"),   # <-- NEW
            Input("ac-teams", "value"),
            Input("ac-games", "value"),
            Input("ac-shot-types", "value"),
            Input("ac-periods", "value"),
            Input("ac-min-makes", "value"),
            Input("ac-rank-by", "value"),
            Input("ac-topn", "value"),
            Input("ac-topn-leaders", "value"),
            Input("ac-search", "value"),
        )
        def update_view(teams, games, shot_modes, periods, min_makes, rank_by, topn, topn_leaders, q):
            df_local = _pbp()

            teams = teams or []
            games = games or []
            periods = periods or [1, int(pd.to_numeric(df_local["period"], errors="coerce").max() or 4)]
            use_points = (rank_by == "points")

            shot_modes = shot_modes or ["ALL"]

            # --------- COMBOS (2PT/3PT only) ----------
            event_types_for_combos = [s for s in shot_modes if s in ("2PT", "3PT")]
            if not event_types_for_combos:
                event_types_for_combos = ["2PT", "3PT"]

            f_combos = _filtered(
                df_local,
                teams=teams,
                games=games,
                periods=tuple(periods),
                event_shot_types=event_types_for_combos,
            )

            # free-text search on combos (passer/finisher)
            if q and q.strip():
                part = q.strip().lower()
                f_combos = f_combos[
                    (f_combos["assist_player"].astype(str).str.lower().str.contains(part, na=False))
                    | (f_combos["finisher"].astype(str).str.lower().str.contains(part, na=False))
                ]

            # Build aggregates for combos table
            frames = []
            if "ALL" in shot_modes:
                frames.append(_aggregate_all(f_combos, weight_by_points=use_points))
            if any(s in shot_modes for s in ("2PT", "3PT")):
                split = _aggregate_split(f_combos, weight_by_points=use_points)
                keep = [s for s in shot_modes if s in ("2PT", "3PT")]
                if keep:
                    split = split[split["shot_type"].isin(keep)]
                frames.append(split)

            agg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            if min_makes and not agg.empty:
                agg = agg[agg["makes"] >= int(min_makes)]

            # KPIs from FG events
            total_makes = int(len(f_combos)) if not f_combos.empty else 0
            total_points = int(f_combos["points"].sum()) if "points" in f_combos.columns and not f_combos.empty else 0
            n_rows = int(len(agg))
            n_games = int(f_combos["game_id"].nunique()) if not f_combos.empty else 0

            kpis = [
                _kpi("Rows (combos shown)", f"{n_rows:,}"),
                _kpi("Total Makes (events)", f"{total_makes:,}"),
                _kpi("Total Points (events)", f"{total_points:,}"),
                _kpi("Games Covered", f"{n_games:,}"),
            ]

            chart_df = agg.copy().sort_values("value", ascending=False).head(int(topn) if topn else 15)
            team_for_charts = teams[0] if len(teams) == 1 else None
            sankey_fig = _sankey(chart_df, team_for_charts, int(topn or 15), use_points)
            bars_fig = _bars(chart_df, team_for_charts, int(topn or 15), use_points)

            table_cols = ["team_name", "assist_player", "finisher", "shot_type", "makes", "points", "finisher_share_pct", "games"]
            rows = (
                agg.sort_values(["value", "makes"], ascending=[False, False])[table_cols]
                .to_dict("records")
                if not agg.empty
                else []
            )

            # --------- LEADERBOARDS (2PT/3PT + FT-first) ----------
            include_codes = set()
            if "ALL" in shot_modes:
                include_codes.update(["P2", "P3", "FT"])
            if "2PT" in shot_modes:
                include_codes.add("P2")
            if "3PT" in shot_modes:
                include_codes.add("P3")
            if "FT" in shot_modes:
                include_codes.add("FT")
            if not include_codes:
                include_codes = {"P2", "P3", "FT"}

            f_lead = _filtered_for_leaders(
                df_local, teams=teams, games=games, periods=tuple(periods), include_codes=list(include_codes)
            )

            search_val = (q or "").strip()

            assist_leaders = _leaderboard_base(
                f_lead.rename(columns={"assist_player_enriched": "assist_player_enriched"}),
                key_col="assist_player_enriched",
                topn=int(topn_leaders or 50),
                search=search_val
            )

            receiver_leaders = _leaderboard_base(
                f_lead, key_col="finisher", topn=int(topn_leaders or 50), search=search_val
            )

            # --------- NEW: TEAM ROBUSTNESS ----------
            team_robust = _team_robustness(f_lead, use_points=use_points)

            return rows, kpis, sankey_fig, bars_fig, \
                   assist_leaders.to_dict("records"), receiver_leaders.to_dict("records"), \
                   team_robust.to_dict("records")

        return app

    except Exception as e:
        # Hard fallback layout if anything fails during init (incl. PageShell or data)
        app.logger.exception("assist_combos initialization failed: %s", e)
        app.layout = html.Div([
            html.H2("Assist Combos – unavailable"),
            html.P("The module failed to initialize. Check logs and ensure pbp_all.csv exists."),
            html.Pre(str(e)),
        ])
        return app


def _kpi(title: str, value: str):
    return html.Div(className="kpi", children=[
        html.Div(title, style={"fontSize": "12px", "color": "#667"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": 700, "marginTop": "2px"}),
    ])
