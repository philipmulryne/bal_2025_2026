# dashapps/shot_chart.py
from __future__ import annotations

import os, re, ast
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict

import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.graph_objects as go
from sklearn.cluster import KMeans

from .common import page_shell as PageShell


# =============================================================================
# IO
# =============================================================================
def _find_csv(env_key: str, candidates: Iterable[Path]) -> Optional[Path]:
    env = os.environ.get(env_key)
    if env and Path(env).exists():
        return Path(env)
    for c in candidates:
        if c and c.exists():
            return c
    return None


def _find_pbp_all() -> Optional[Path]:
    data_root = Path(os.environ.get("DATA_ROOT", "out_data"))
    p = _find_csv("PBP_ALL", [
        data_root / "_cumulative" / "pbp_all.csv",
        data_root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),
    ])
    if p:
        return p
    for c in data_root.rglob("pbp_all.csv"):
        return c
    return None


def _load_pbp() -> pd.DataFrame:
    src = _find_pbp_all()
    if not src or not src.exists():
        raise FileNotFoundError("pbp_all.csv not found. Set PBP_ALL or place it under /mnt/data.")
    df = pd.read_csv(src)

    # normalize minimal fields
    for c in ["ev_code", "ev_result", "actionNumber", "period", "game_id",
              "team_name", "player", "player_name_from_roster", "qualifier"]:
        if c not in df.columns:
            df[c] = np.nan
    df["ev_code"] = df["ev_code"].astype(str).str.lower()
    df["ev_result"] = df["ev_result"].astype(str).str.lower()
    for c in ["actionNumber", "period", "game_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "player" not in df.columns and "player_name_from_roster" in df.columns:
        df["player"] = df["player_name_from_roster"]
    df["player"] = df["player"].fillna("Unknown Player")

    # FG attempts only
    df["is_fg_attempt"] = df["ev_code"].isin(["p2", "p3"])
    df = df[df["is_fg_attempt"]].copy()
    return df


# =============================================================================
# Hover helpers (opponent/week/score/clock) — tailored to your CSV
# =============================================================================
def _fmt_mmss(total_seconds: Optional[int]) -> str:
    try:
        if total_seconds is None or (isinstance(total_seconds, float) and np.isnan(total_seconds)):
            return "—"
        total_seconds = int(total_seconds)
        if total_seconds < 0:
            total_seconds = 0
        m = total_seconds // 60
        s = total_seconds % 60
        return f"{m}:{s:02d}"
    except Exception:
        return "—"


def _coerce_mmss(s: str) -> str:
    """
    Accepts strings like '09:43' or '09:43:00' and returns '9:43'.
    """
    if pd.isna(s):
        return "—"
    s = str(s).strip()
    m = re.match(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$", s)
    if not m:
        return "—"
    mins, secs = int(m.group(1)), int(m.group(2))
    return f"{mins}:{secs:02d}"


def _build_context_columns(df: pd.DataFrame, spec: str) -> Dict[str, pd.Series]:
    """
    Robustly produce:
      opp (opponent name)  — inferred from the other team in the game
      week                 — 'week' column if present
      score_str            — from cum_1 / cum_2 when available
      q_clock              — per-period clock (MM:SS)
      g_clock              — game-time elapsed (MM:SS), computed from q_clock + period
    Works directly with your CSV schema.
    """
    out: Dict[str, pd.Series] = {}

    # Week / Round
    if "week" in df.columns and df["week"].notna().any():
        out["week"] = df["week"].astype(str)
    else:
        out["week"] = pd.Series(["—"] * len(df), index=df.index)

    # Opponent: infer from the other team appearing under the same game_id
    if {"game_id", "team_name"}.issubset(df.columns):
        teams_by_game = (
            df[["game_id", "team_name"]]
            .dropna()
            .drop_duplicates()
            .groupby("game_id")["team_name"]
            .apply(list)
            .to_dict()
        )

        def _opp_row(gid, team):
            if pd.isna(gid) or pd.isna(team):
                return "—"
            pool = [t for t in teams_by_game.get(gid, []) if str(t) != str(team)]
            return pool[0] if pool else "—"

        out["opp"] = df.apply(lambda r: _opp_row(r.get("game_id"), r.get("team_name")), axis=1)
    else:
        out["opp"] = pd.Series(["—"] * len(df), index=df.index)

    # Score at event time — prefer cum_1/cum_2
    if {"cum_1", "cum_2"}.issubset(df.columns):
        s1 = pd.to_numeric(df["cum_1"], errors="coerce").fillna(0).astype(int).astype(str)
        s2 = pd.to_numeric(df["cum_2"], errors="coerce").fillna(0).astype(int).astype(str)
        out["score_str"] = s1 + "–" + s2
    elif {"s1", "s2"}.issubset(df.columns):
        s1 = pd.to_numeric(df["s1"], errors="coerce").fillna(0).astype(int).astype(str)
        s2 = pd.to_numeric(df["s2"], errors="coerce").fillna(0).astype(int).astype(str)
        out["score_str"] = s1 + "–" + s2
    else:
        out["score_str"] = pd.Series(["—"] * len(df), index=df.index)

    # Period clock string
    if "gt" in df.columns and df["gt"].notna().any():
        q_clock = df["gt"].apply(_coerce_mmss)
    elif "clock" in df.columns and df["clock"].notna().any():
        q_clock = df["clock"].apply(_coerce_mmss)
    else:
        q_clock = pd.Series(["—"] * len(df), index=df.index)
    out["q_clock"] = q_clock

    # Derive game-time elapsed
    per_len = 12 * 60 if str(spec).upper() == "NBA" else 10 * 60
    ot_len = 5 * 60
    period_num = pd.to_numeric(df.get("period", np.nan), errors="coerce")

    def _gclock_row(qc: str, p) -> str:
        m = re.match(r"^\s*(\d+):([0-5]\d)\s*$", str(qc))
        if not m or pd.isna(p):
            return "—"
        mins, secs = int(m.group(1)), int(m.group(2))
        p = int(p)
        # Interpret qc as REMAINING time in the period (matches your CSV)
        elapsed = 0
        if p <= 4:
            elapsed += (p - 1) * per_len
            elapsed += (per_len - (mins * 60 + secs))
        else:
            elapsed += 4 * per_len
            elapsed += (p - 5) * ot_len
            elapsed += (ot_len - (mins * 60 + secs))
        return _fmt_mmss(elapsed)

    out["g_clock"] = [
        _gclock_row(qc, p) for qc, p in zip(out["q_clock"], period_num)
    ]
    return out


# =============================================================================
# XY detection (strict)
# =============================================================================
def _xy_candidates(cols: List[str], patterns: List[str]) -> List[str]:
    out = []
    for c in cols:
        lc = c.lower()
        if any(re.fullmatch(p, lc) or re.search(p, lc) for p in patterns):
            out.append(c)
    return out


def _detect_xy(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    x_patterns = [r"^x$", r"^x_coord$", r"^x[-_ ]?pos$", r"^locx$", r"^x[_-]?loc$",
                  r"^shot[_-]?x$", r".*[_-]x$"]
    y_patterns = [r"^y$", r"^y_coord$", r"^y[-_ ]?pos$", r"^locy$", r"^y[_-]?loc$",
                  r"^shot[_-]?y$", r".*[_-]y$"]
    xs = _xy_candidates(cols, x_patterns)
    ys = _xy_candidates(cols, y_patterns)
    if not xs or not ys:
        raise ValueError("Could not detect shot location columns. Please ensure x/y or locX/locY exist.")
    best = (-1, None, None)
    for x in xs:
        xnum = pd.to_numeric(df[x], errors="coerce")
        for y in ys:
            ynum = pd.to_numeric(df[y], errors="coerce")
            ov = int((xnum.notna() & ynum.notna()).sum())
            if ov > best[0]:
                best = (ov, x, y)
    return best[1], best[2]


# =============================================================================
# Court specs + transform
# =============================================================================
def _parse_qualifier(val) -> set:
    if pd.isna(val):
        return set()
    s = str(val).strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return {str(x).lower() for x in obj}
        return {str(obj).lower()}
    except Exception:
        toks = [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]
        return set(toks)


def _spec_params(spec: str) -> Dict[str, float]:
    s = (spec or "NBA").upper()
    if s == "FIBA":
        return dict(
            C_W=91.86, C_H=49.21,
            hoop_x=5.17, backboard_x=3.94,
            lane_w=16.40, ft_depth=19.00,
            restricted_r=4.10, arc_r=22.15,
            corner_x=22.15, corner_y=14.0,
        )
    # NBA default
    return dict(
        C_W=94.0, C_H=50.0,
        hoop_x=5.25, backboard_x=4.0,
        lane_w=16.0, ft_depth=19.0,
        restricted_r=4.0, arc_r=23.75,
        corner_x=22.0, corner_y=14.0,
    )


def _estimate_transform(df: pd.DataFrame, xcol: str, ycol: str, spec: str = "NBA") -> Dict[str, float]:
    const = _spec_params(spec)
    C_W, C_H = const["C_W"], const["C_H"]
    hoop_x = const["hoop_x"]

    X = pd.to_numeric(df[xcol], errors="coerce")
    Y = pd.to_numeric(df[ycol], errors="coerce")
    shots = df.loc[X.notna() & Y.notna(), [xcol, ycol, "qualifier"]].copy()
    shots[xcol], shots[ycol] = X.loc[shots.index].astype(float), Y.loc[shots.index].astype(float)

    if "qualifier" in shots.columns:
        Q = shots["qualifier"].apply(_parse_qualifier)
        shots["is_paint"] = Q.apply(lambda s: "pointsinthepaint" in s)
    else:
        shots["is_paint"] = False

    paint = shots.loc[shots["is_paint"], [xcol, ycol]]
    if len(paint) >= 10:
        spread_x = paint[xcol].max() - paint[xcol].min()
        spread_y = paint[ycol].max() - paint[ycol].min()
        primary_axis = "x" if spread_x >= spread_y else "y"
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(
            paint[[xcol]] if primary_axis == "x" else paint[[ycol]]
        )
        tmp = paint.assign(cluster=km.labels_)
        hoop_centers = tmp.groupby("cluster")[[xcol, ycol]].median().sort_values(
            by=xcol if primary_axis == "x" else ycol
        ).reset_index(drop=True)
        x_left, y_left = float(hoop_centers.iloc[0][xcol]), float(hoop_centers.iloc[0][ycol])
        x_right, y_right = float(hoop_centers.iloc[1][xcol]), float(hoop_centers.iloc[1][ycol])
    else:
        x_left, x_right = float(shots[xcol].quantile(0.02)), float(shots[xcol].quantile(0.98))
        y_left = y_right = float(shots[ycol].median())

    hoop_span = (C_W - 2 * hoop_x)
    bx = (x_right - x_left) / hoop_span
    ax = x_left - bx * hoop_x

    y_center = np.median([y_left, y_right])
    y_low, y_high = float(shots[ycol].quantile(0.01)), float(shots[ycol].quantile(0.99))
    by = (y_high - y_low) / C_H if (y_high > y_low) else 1.0
    ay = y_center - by * (C_H / 2.0)

    return {
        "ax": ax, "bx": bx, "ay": ay, "by": by,
        "x_left": x_left, "x_right": x_right, "y_center": y_center,
        "spec": spec, **const
    }


def _court_to_data(cx: np.ndarray, cy: np.ndarray, tr: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    X = tr["ax"] + tr["bx"] * cx
    Y = tr["ay"] + tr["by"] * cy
    return X, Y


def _data_to_court(X: np.ndarray, Y: np.ndarray, tr: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    Cx = (X - tr["ax"]) / tr["bx"]
    Cy = (Y - tr["ay"]) / tr["by"]
    return Cx, Cy


def _fold_to_halfcourt(Cx: np.ndarray, Cy: np.ndarray, tr: Dict[str, float], left_half: bool) -> Tuple[np.ndarray, np.ndarray]:
    C_W, C_H = tr["C_W"], tr["C_H"]
    Cx2, Cy2 = Cx.copy(), Cy.copy()
    if left_half:
        mask = Cx2 > (C_W / 2.0)
        Cx2[mask] = C_W - Cx2[mask]
        Cy2[mask] = C_H - Cy2[mask]
    else:
        mask = Cx2 < (C_W / 2.0)
        Cx2[mask] = C_W - Cx2[mask]
        Cy2[mask] = C_H - Cy2[mask]
    return Cx2, Cy2


# =============================================================================
# Zones + grouping
# =============================================================================
def _zone_labels_fine() -> List[str]:
    return [
        "Rim (0–4ft)",
        "Paint (4–14ft)",
        "Short Mid (14–16ft)",
        "Long Mid (16–22ft)",
        "Left Corner 3",
        "Right Corner 3",
        "Left Wing 3",
        "Center 3",
        "Right Wing 3",
        "Heave (40+ ft)",
    ]


def _zone_labels_aggregate() -> List[str]:
    return ["Corners (All)", "Wings (All)", "Threes (All)"]


def _zone_filter_options() -> List[Dict[str, str]]:
    opts = [{"label": z, "value": z} for z in _zone_labels_fine()]
    opts += [{"label": z, "value": z} for z in _zone_labels_aggregate()]
    return opts


def _expand_zone_filter(sel: Iterable[str]) -> List[str]:
    base = set(sel or [])
    fine = set()
    for z in base:
        if z in _zone_labels_fine():
            fine.add(z)
        elif z == "Corners (All)":
            fine.update(["Left Corner 3", "Right Corner 3"])
        elif z == "Wings (All)":
            fine.update(["Left Wing 3", "Center 3", "Right Wing 3"])
        elif z == "Threes (All)":
            fine.update(["Left Corner 3", "Right Corner 3", "Left Wing 3", "Center 3", "Right Wing 3"])
    if not fine:
        fine.update(_zone_labels_fine())
    return sorted(fine)


def _assign_zones_and_groups(
    cx: np.ndarray,
    cy: np.ndarray,
    tr: Dict[str, float],
    side_top_is_left: bool = True,
    center_band: float = 6.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zone labeling that mirrors left/right names per attacking hoop,
    so Full-court naming is correct on both halves.

    side_top_is_left:
      - True  => in the right-hoop half, TOP side is called "Left".
      - False => in the right-hoop half, TOP side is called "Right".
    On the left-hoop half, this mapping is inverted automatically.
    """
    C_W, C_H = tr["C_W"], tr["C_H"]
    hoop_x = tr["hoop_x"]
    arc_r = tr["arc_r"]
    corner_x = tr["corner_x"]
    corner_y = tr["corner_y"]

    hx_left, hy = hoop_x, C_H / 2.0
    hx_right = C_W - hoop_x

    dl = np.hypot(cx - hx_left,  cy - hy)
    dr = np.hypot(cx - hx_right, cy - hy)
    use_left = (dl <= dr)  # True → nearer the left basket

    dx = np.where(use_left, np.abs(cx - hx_left),  np.abs(cx - hx_right))
    dy = np.abs(cy - hy)
    r  = np.where(use_left, dl, dr)

    behind_arc = r >= arc_r
    in_corner_band  = dy >= (C_H / 2.0 - corner_y)
    in_corner_depth = dx <= corner_x
    is_corner3 = behind_arc & in_corner_band & in_corner_depth

    is_above_break = behind_arc & ~is_corner3
    is_center3 = is_above_break & (dy <= float(center_band))

    # ----- key fix: mirror naming on left-hoop half -----
    is_left_side_base = (cy >= hy) if side_top_is_left else (cy < hy)
    is_left_side = np.where(use_left, ~is_left_side_base, is_left_side_base)
    # -----------------------------------------------------

    zones  = np.full_like(cx, fill_value="", dtype=object)
    groups = np.full_like(cx, fill_value="", dtype=object)

    zones[r <= 4.0] = "Rim (0–4ft)";                groups[r <= 4.0] = "Rim (0–4ft)"
    m = (r > 4.0) & (r <= 14.0);                    zones[m] = "Paint (4–14ft)";       groups[m] = "Paint (4–14ft)"
    m = (r > 14.0) & (r <= 16.0);                   zones[m] = "Short Mid (14–16ft)";  groups[m] = "Short Mid (14–16ft)"
    m = (r > 16.0) & (r <= 22.0);                   zones[m] = "Long Mid (16–22ft)";   groups[m] = "Long Mid (16–22ft)"

    lc = is_corner3 & is_left_side
    rc = is_corner3 & ~is_left_side
    zones[lc] = "Left Corner 3";   groups[lc] = "Corner 3"
    zones[rc] = "Right Corner 3";  groups[rc] = "Corner 3"

    zones[is_center3] = "Center 3"; groups[is_center3] = "Above-Break 3"
    wing_mask = is_above_break & ~is_center3
    lw = wing_mask & is_left_side
    rw = wing_mask & ~is_left_side
    zones[lw] = "Left Wing 3";     groups[lw] = "Above-Break 3"
    zones[rw] = "Right Wing 3";    groups[rw] = "Above-Break 3"

    zones[r >= 40.0] = "Heave (40+ ft)"; groups[r >= 40.0] = "Heave (40+ ft)"

    m = (zones == "") & (r < arc_r)
    zones[m]  = "Long Mid (16–22ft)"
    groups[m] = "Long Mid (16–22ft)"
    m = (zones == "")
    zones[m]  = "Right Wing 3"
    groups[m] = "Above-Break 3"

    return zones, groups


def _fg_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return dict(FGM=0, FGA=0, PCT=0.0, PTS=0, eFG=0.0, P3M=0, P3A=0)
    fga = len(frame)
    fgm = int(frame["is_make"].sum())
    p3a = int((frame["ev_code"] == "p3").sum())
    p3m = int(((frame["ev_code"] == "p3") & frame["is_make"]).sum())
    pct = fgm / fga if fga else 0.0
    efg = (fgm + 0.5 * p3m) / fga if fga else 0.0
    pts = int((fgm - p3m) * 2 + p3m * 3)
    return dict(FGM=fgm, FGA=fga, PCT=pct, PTS=pts, eFG=efg, P3M=p3m, P3A=p3a)


# =============================================================================
# Shapes + plotting
# =============================================================================
def _court_shapes(tr: Dict[str, float], halfcourt: bool = False, left_half: bool = True) -> List[dict]:
    C_W, C_H = tr["C_W"], tr["C_H"]
    hoop_x = tr["hoop_x"]
    lane_w, ft_depth = tr["lane_w"], tr["ft_depth"]
    backboard_x = tr["backboard_x"]
    arc_r = tr["arc_r"]
    corner_y = tr["corner_y"]

    # region in court coords
    if halfcourt:
        if left_half:
            cx0, cx1 = 0.0, C_W / 2.0
        else:
            cx0, cx1 = C_W / 2.0, C_W
    else:
        cx0, cx1 = 0.0, C_W
    cy0, cy1 = 0.0, C_H

    def rect(x0, y0, x1, y1, lw=1.2):
        X0, Y0 = _court_to_data(np.array([x0]), np.array([y0]), tr)
        X1, Y1 = _court_to_data(np.array([x1]), np.array([y1]), tr)
        return dict(type="rect", x0=X0[0], y0=Y0[0], x1=X1[0], y1=Y1[0],
                    line=dict(color="#EEEEEE", width=lw), layer="below")

    def line(x0, y0, x1, y1, lw=1.2):
        X0, Y0 = _court_to_data(np.array([x0]), np.array([y0]), tr)
        X1, Y1 = _court_to_data(np.array([x1]), np.array([y1]), tr)
        return dict(type="line", x0=X0[0], y0=Y0[0], x1=X1[0], y1=Y1[0],
                    line=dict(color="#EEEEEE", width=lw), layer="below")

    def circle(cx, cy, r, lw=1.2):
        X0, Y0 = _court_to_data(np.array([cx - r]), np.array([cy - r]), tr)
        X1, Y1 = _court_to_data(np.array([cx + r]), np.array([cy + r]), tr)
        return dict(type="circle", x0=X0[0], y0=Y0[0], x1=X1[0], y1=Y1[0],
                    line=dict(color="#EEEEEE", width=lw), layer="below")

    shapes = [rect(cx0, cy0, cx1, cy1, lw=2.0)]

    def half_elements(is_left: bool):
        hx = hoop_x if is_left else (C_W - hoop_x)
        board_x = backboard_x if is_left else (C_W - backboard_x)
        lane_x0 = 0.0 if is_left else (C_W - ft_depth)
        lane_x1 = ft_depth if is_left else C_W
        shapes.extend([
            rect(lane_x0, (C_H - lane_w) / 2, lane_x1, (C_H + lane_w) / 2),
            line(board_x, (C_H / 2) - 3.0, board_x, (C_H / 2) + 3.0, lw=3.0),  # backboard
            circle(hx, C_H / 2, 0.75, lw=3.0),                                   # rim
            circle(hx, C_H / 2, tr["restricted_r"]),                             # restricted
            # three-point approx
            line(0.0 if is_left else C_W, 3.0, 0.0 if is_left else C_W, 3.0 + corner_y),
            line(0.0 if is_left else C_W, C_H - 3.0 - corner_y, 0.0 if is_left else C_W, C_H - 3.0),
            circle(hx, C_H / 2, arc_r),
        ])

    if halfcourt:
        half_elements(left_half)
    else:
        half_elements(True)
        half_elements(False)
        shapes.append(line(C_W / 2, 0, C_W / 2, C_H, lw=2.0))  # midcourt

    return shapes


def _scatter(df: pd.DataFrame, xcol: str, ycol: str, name: str, is_make: bool) -> go.Scattergl:
    color = "#1f9e49" if is_make else "#c0392b"
    sym = "circle" if is_make else "x"
    return go.Scattergl(
        x=df[xcol], y=df[ycol], mode="markers", name=name,
        marker=dict(symbol=sym, size=9 if is_make else 10,
                    line=dict(width=0.5, color="rgba(0,0,0,0.4)"), color=color),
        opacity=0.9 if is_make else 0.85,
        hovertemplate="%{text}<extra></extra>",
        text=df.get("hover", None),
    )


def _heat(df: pd.DataFrame, xcol: str, ycol: str, nbins=60) -> go.Histogram2d:
    return go.Histogram2d(x=df[xcol], y=df[ycol], nbinsx=nbins, nbinsy=nbins,
                          colorscale="YlOrRd", opacity=0.9, name="Density", showscale=True)


def _zone_overlay_shapes(tr: Dict[str, float], zone_fg_group: Dict[str, float], alpha: float = 0.28) -> List[dict]:
    C_W, C_H = tr["C_W"], tr["C_H"]
    hoop_x = tr["hoop_x"]
    arc_r = tr["arc_r"]
    corner_x = tr["corner_x"]
    corner_y = tr["corner_y"]

    def rgba_for_pct(p: float) -> str:
        p = max(0.30, min(0.70, p))
        t = (p - 0.30) / 0.40  # 0..1
        r = int(255 * (1 - t))
        g = int(255 * t)
        b = 60
        return f"rgba({r},{g},{b},{alpha})"

    shapes: List[dict] = []

    def ring(cx, cy, r0, r1, fill_opacity_color):
        X0, Y0 = _court_to_data(np.array([cx - r1]), np.array([cy - r1]), tr)
        X1, Y1 = _court_to_data(np.array([cx + r1]), np.array([cy + r1]), tr)
        return dict(type="circle", x0=X0[0], y0=Y0[0], x1=X1[0], y1=Y1[0],
                    line=dict(color="rgba(0,0,0,0)"), fillcolor=fill_opacity_color, layer="below")

    for hx in (hoop_x, C_W - hoop_x):
        for z, (r0, r1) in {
            "Rim (0–4ft)": (0.0, 4.0),
            "Paint (4–14ft)": (4.0, 14.0),
            "Short Mid (14–16ft)": (14.0, 16.0),
            "Long Mid (16–22ft)": (16.0, 22.0),
            "Above-Break 3": (22.0, arc_r + 2.0),
        }.items():
            p = zone_fg_group.get(z)
            if p is not None:
                shapes.append(ring(hx, C_H / 2, r0, r1, rgba_for_pct(p)))
        # Corners (rect bands)
        p = zone_fg_group.get("Corner 3")
        if p is not None:
            # bottom
            X0, Y0 = _court_to_data(np.array([0.0 if hx < C_W / 2 else (C_W - corner_x)]),
                                    np.array([3.0]), tr)
            X1, Y1 = _court_to_data(np.array([corner_x if hx < C_W / 2 else C_W]),
                                    np.array([3.0 + corner_y]), tr)
            shapes.append(dict(type="rect", x0=min(X0[0], X1[0]), y0=Y0[0], x1=max(X0[0], X1[0]), y1=Y1[0],
                               line=dict(color="rgba(0,0,0,0)"), fillcolor=rgba_for_pct(p), layer="below"))
            # top
            X0, Y0 = _court_to_data(np.array([0.0 if hx < C_W / 2 else (C_W - corner_x)]),
                                    np.array([C_H - 3.0 - corner_y]), tr)
            X1, Y1 = _court_to_data(np.array([corner_x if hx < C_W / 2 else C_W]),
                                    np.array([C_H - 3.0]), tr)
            shapes.append(dict(type="rect", x0=min(X0[0], X1[0]), y0=Y0[0], x1=max(X0[0], X1[0]), y1=Y1[0],
                               line=dict(color="rgba(0,0,0,0)"), fillcolor=rgba_for_pct(p), layer="below"))
    return shapes


def _zone_annotations_from_data(f_fold: pd.DataFrame, xcol: str, ycol: str, text_by_zone: Dict[str, str]) -> List[dict]:
    ann: List[dict] = []
    for z, grp in f_fold.groupby("zone"):
        if z not in text_by_zone or grp.empty:
            continue
        x_med = float(np.median(grp[xcol].to_numpy(dtype=float)))
        y_med = float(np.median(grp[ycol].to_numpy(dtype=float)))
        ann.append(dict(
            x=x_med, y=y_med, text=text_by_zone[z],
            showarrow=False, font=dict(size=13, color="#111"),
            bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1
        ))
    return ann


# =============================================================================
# Hex overlay helpers
# =============================================================================
def _rgba_for_val(v: float, vmin: float, vmax: float, alpha: float = 0.66) -> str:
    if vmax <= vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    r = int(255 * (1 - t))
    g = int(255 * t)
    b = 60
    return f"rgba({r},{g},{b},{alpha})"


def _hex_vertices(cx: float, cy: float, R: float) -> List[Tuple[float, float]]:
    ang = np.deg2rad([30, 90, 150, 210, 270, 330])
    return [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in ang]


def _hexbin_shapes(
    f_plot: pd.DataFrame,
    xcol: str,
    ycol: str,
    tr: Dict[str, float],
    metric: str = "efg",
    R: float = 3.0,
    min_fga: int = 4
) -> List[dict]:
    if f_plot.empty:
        return []

    Cx = f_plot[xcol].to_numpy(dtype=float)
    Cy = f_plot[ycol].to_numpy(dtype=float)
    Cx, Cy = _data_to_court(Cx, Cy, tr)

    dx = 1.5 * R
    dy = np.sqrt(3.0) * R

    col = np.rint(Cx / dx).astype(int)
    row = np.rint((Cy - (col % 2) * (dy / 2.0)) / dy).astype(int)

    tmp = pd.DataFrame({
        "col": col,
        "row": row,
        "is_make": f_plot["is_make"].to_numpy(),
        "ev_code": f_plot["ev_code"].to_numpy()
    })

    def _agg(g: pd.DataFrame) -> pd.Series:
        fga = len(g)
        fgm = int(g["is_make"].sum())
        p3m = int(((g["ev_code"] == "p3") & g["is_make"]).sum())
        if fga == 0:
            return pd.Series(dict(FGA=0, PCT=0.0, eFG=0.0))
        return pd.Series(dict(FGA=fga, PCT=fgm / fga, eFG=(fgm + 0.5 * p3m) / fga))

    bins = tmp.groupby(["col", "row"]).apply(_agg).reset_index()
    bins = bins[bins["FGA"] >= min_fga]
    if bins.empty:
        return []

    if metric == "fg":
        val = bins["PCT"].to_numpy(dtype=float)
        vmin, vmax = 0.30, 0.70
    elif metric == "fga":
        val = bins["FGA"].to_numpy(dtype=float)
        vmin, vmax = float(val.min()), float(val.max())
    else:  # efg
        val = bins["eFG"].to_numpy(dtype=float)
        vmin, vmax = 0.30, 0.70

    shapes: List[dict] = []
    for c, r, v in zip(bins["col"], bins["row"], val):
        cx = c * dx
        cy = r * dy + (c % 2) * (dy / 2.0)
        poly_court = _hex_vertices(cx, cy, R)
        xs, ys = _court_to_data(np.array([p[0] for p in poly_court]),
                                np.array([p[1] for p in poly_court]), tr)
        path = "M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys)) + " Z"
        shapes.append(dict(
            type="path", path=path,
            line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            fillcolor=_rgba_for_val(float(v), vmin, vmax, alpha=0.60),
            layer="below"
        ))
    return shapes


# =============================================================================
# Page / Layout
# =============================================================================
def _page_content(df: pd.DataFrame, period_max: int) -> html.Div:
    team_opts = [{"label": t, "value": t} for t in sorted(df["team_name"].dropna().unique())]
    player_opts = [{"label": p, "value": p} for p in sorted(df["player"].dropna().unique())]
    games_view = df[["game_id"]].dropna().drop_duplicates().sort_values("game_id")
    game_opts = [{"label": str(int(g)) if pd.notna(g) else "", "value": str(int(g))} for g in games_view["game_id"]]

    return html.Div(children=[
        html.H2("Shot Chart (Data-Calibrated)"),
        html.P("Fine zones include Left/Right Corners, Left/Right/Center Wings. Aggregates (All Corners/Wings/Threes) are supported in filters and tables."),

        # Row 1 — core filters
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(6,1fr)", "gap": "12px"}, children=[
            html.Div(children=[html.Label("Teams"), dcc.Dropdown(id="sc-teams", options=team_opts, multi=True)]),
            html.Div(children=[html.Label("Players"), dcc.Dropdown(id="sc-players", options=player_opts, multi=True)]),
            html.Div(children=[html.Label("Games"), dcc.Dropdown(id="sc-games", options=game_opts, multi=True)]),
            html.Div(children=[html.Label("Periods"),
                dcc.RangeSlider(id="sc-periods", min=1, max=period_max, step=1, value=[1, period_max],
                                marks={i: str(i) for i in range(1, period_max + 1)})]),
            html.Div(children=[html.Label("Court View"),
                dcc.RadioItems(id="sc-court", options=[
                    {"label": " Full", "value": "full"},
                    {"label": " Half (Left)", "value": "half_left"},
                    {"label": " Half (Right)", "value": "half_right"},
                ], value="full", inputStyle={"marginRight": "6px", "marginLeft": "8px"})]),
            html.Div(children=[html.Label("Spec"),
                dcc.RadioItems(id="sc-spec", options=[{"label": " NBA", "value": "NBA"}, {"label": " FIBA", "value": "FIBA"}],
                               value="NBA", inputStyle={"marginRight": "6px", "marginLeft": "8px"})]),
        ]),

        # Row 2 — display toggles (compact), with collapsible Zone Filter
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(9,1fr)", "gap": "12px", "marginTop": "6px"},
            children=[
                html.Div(children=[html.Label("Include"),
                    dcc.Checklist(id="sc-include", options=[{"label": " Makes", "value": "make"}, {"label": " Misses", "value": "miss"}],
                                  value=["make", "miss"])]),
                html.Div(children=[html.Label("Flip Axes"),
                    dcc.Checklist(id="sc-flips", options=[{"label": " Flip X", "value": "flipx"}, {"label": " Flip Y", "value": "flipy"}], value=[])]),
                html.Div(children=[html.Label("Render Mode"),
                    dcc.RadioItems(id="sc-mode", options=[
                        {"label": " Scatter", "value": "scatter"},
                        {"label": " Heatmap", "value": "heat"},
                        {"label": " Hex (eFG)", "value": "hex"},
                    ], value="scatter", inputStyle={"marginRight": "6px", "marginLeft": "8px"})]),
                html.Div(children=[html.Label("Zone Overlay"),
                    dcc.Checklist(id="sc-zone-overlay", options=[{"label": " Show FG% overlay", "value": "on"}], value=[])]),
                html.Div(children=[html.Label("Zone Labels"),
                    dcc.Checklist(id="sc-zone-labels", options=[{"label": " Show % labels (half-court)", "value": "on"}], value=["on"])]),
                html.Div(children=[html.Label("Normalize to Hoop"),
                    dcc.Checklist(id="sc-fold", options=[{"label": " Fold to selected half", "value": "on"}], value=["on"])]),
                html.Div(children=[html.Label("Side Naming"),
                    dcc.RadioItems(
                        id="sc-side-map",
                        options=[
                            {"label": " Top → Left",  "value": "top_left"},
                            {"label": " Top → Right", "value": "top_right"},
                        ],
                        value="top_left",
                        inputStyle={"marginRight": "6px", "marginLeft": "8px"}
                    )]),
                # Collapsible Zone Filter (so it doesn't eat space)
                html.Div(children=[
                    html.Details(open=False, children=[
                        html.Summary("Zone Filter"),
                        html.Div(style={"maxHeight": "240px", "overflowY": "auto", "padding": "6px", "border": "1px solid #ddd", "borderRadius": "8px"},
                                 children=[
                                     dcc.Checklist(id="sc-zone-filter",
                                                   options=_zone_filter_options(),
                                                   value=_zone_labels_fine(),
                                                   labelStyle={"display": "block"})
                                 ])
                    ])
                ]),
                html.Div(children=[html.Label("Diagnostics"),
                    dcc.Checklist(id="sc-diag", options=[{"label": " Show", "value": "on"}], value=[])]),
            ]
        ),

        html.Hr(style={"margin":"8px 0"}),
        dcc.Graph(id="sc-fig", style={"height": "720px"}),

        html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "alignItems": "start"}, children=[
            html.Div(children=[
                html.H3("Zones — Current Selection"),
                dash_table.DataTable(
                    id="sc-zone-table",
                    columns=[
                        {"name": "Zone", "id": "zone"},
                        {"name": "FGA", "id": "FGA", "type": "numeric"},
                        {"name": "FGM", "id": "FGM", "type": "numeric"},
                        {"name": "FG%", "id": "FG%", "type": "numeric"},
                        {"name": "3PA", "id": "3PA", "type": "numeric"},
                        {"name": "3PM", "id": "3PM", "type": "numeric"},
                        {"name": "eFG%", "id": "eFG%", "type": "numeric"},
                        {"name": "PTS", "id": "PTS", "type": "numeric"},
                    ],
                    data=[],
                    page_size=12,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "6px", "fontSize": "14px"},
                ),
                dcc.Graph(id="sc-zone-bar", style={"height": "320px", "marginTop": "8px"}),
            ]),
            html.Div(children=[
                html.H3("Leaderboards (Aggregate)"),
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2,1fr)", "gap": "8px"}, children=[
                    html.Div(children=[html.Label("Group By"),
                        dcc.RadioItems(id="sc-leader-dim", options=[{"label": " Player", "value": "player"}, {"label": " Team", "value": "team"}],
                                       value="player", inputStyle={"marginRight": "6px", "marginLeft": "8px"})]),
                    html.Div(children=[html.Label("Sort"),
                        dcc.Dropdown(id="sc-leader-sort", options=[
                            {"label": "eFG%", "value": "eFG"},
                            {"label": "FG%", "value": "PCT"},
                            {"label": "FGA", "value": "FGA"},
                            {"label": "Points", "value": "PTS"},
                        ], value="eFG")]),
                ]),
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2,1fr)", "gap": "8px", "marginTop": "6px"}, children=[
                    html.Div(children=[html.Label("Top-N"),
                        dcc.Slider(id="sc-leader-topn", min=5, max=50, step=1, value=15,
                                   marks={5: "5", 15: "15", 30: "30", 50: "50"})]),
                    html.Div(children=[html.Label("Min FGA"),
                        dcc.Input(id="sc-leader-minfga", type="number", value=10, min=0, step=1, style={"width": "100%"})]),
                ]),
                dash_table.DataTable(
                    id="sc-leaderboard",
                    columns=[
                        {"name": "Entity", "id": "Entity"},
                        {"name": "FGA", "id": "FGA", "type": "numeric"},
                        {"name": "FGM", "id": "FGM", "type": "numeric"},
                        {"name": "FG%", "id": "FG%", "type": "numeric"},
                        {"name": "3PA", "id": "3PA", "type": "numeric"},
                        {"name": "3PM", "id": "3PM", "type": "numeric"},
                        {"name": "eFG%", "id": "eFG%", "type": "numeric"},  # <-- FIXED stray quote here
                        {"name": "PTS", "id": "PTS", "type": "numeric"},
                    ],
                    data=[],
                    page_size=15,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "6px", "fontSize": "14px"},
                ),
            ]),
        ]),

        html.H3("Leaderboards (By Zone)"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)", "gap": "8px"}, children=[
            html.Div(children=[html.Label("By"),
                dcc.RadioItems(id="sc-zlead-dim",
                    options=[{"label":" Player","value":"player"},{"label":" Team","value":"team"}],
                    value="player", inputStyle={"marginRight":"6px","marginLeft":"8px"})]),
            html.Div(children=[html.Label("Zone"),
                dcc.Dropdown(id="sc-zlead-zone",
                    options=[{"label":z,"value":z} for z in _zone_labels_fine()],
                    value="Left Corner 3")]),
            html.Div(children=[html.Label("Sort"),
                dcc.Dropdown(id="sc-zlead-sort", options=[
                    {"label":"eFG%","value":"eFG"},
                    {"label":"FG%","value":"PCT"},
                    {"label":"FGA","value":"FGA"},
                    {"label":"Points","value":"PTS"}
                ], value="eFG")]),
            html.Div(children=[html.Label("Top-N"),
                dcc.Slider(id="sc-zlead-topn", min=5, max=50, step=1, value=15,
                           marks={5:"5",15:"15",30:"30",50:"50"})]),
            html.Div(children=[html.Label("Min FGA"),
                dcc.Input(id="sc-zlead-minfga", type="number", value=8, min=0, step=1, style={"width":"100%"})]),
        ]),
        dash_table.DataTable(
            id="sc-zone-entity-leaderboard",
            columns=[
                {"name":"Entity","id":"Entity"},
                {"name":"Zone","id":"zone"},
                {"name":"FGA","id":"FGA","type":"numeric"},
                {"name":"FGM","id":"FGM","type":"numeric"},
                {"name":"FG%","id":"FG%","type":"numeric"},
                {"name":"3PA","id":"3PA","type":"numeric"},
                {"name":"3PM","id":"3PM","type":"numeric"},
                {"name":"eFG%","id":"eFG%","type":"numeric"},
                {"name":"PTS","id":"PTS","type":"numeric"},
            ],
            data=[], page_size=15, sort_action="native",
            style_table={"overflowX":"auto"},
            style_cell={"padding":"6px","fontSize":"14px"},
        ),

        html.Div(id="sc-diag-panel", style={"marginTop":"10px"}),
    ])


# =============================================================================
# App
# =============================================================================
def create_dash_shot_chart(server, base_pathname: str = "/shot_chart/") -> Dash:
    app = dash.Dash(__name__, server=server, url_base_pathname=base_pathname,
                    suppress_callback_exceptions=True, title="Shot Chart")

    df = _load_pbp()
    xcol, ycol = _detect_xy(df)

    df = df.assign(**{
        xcol: pd.to_numeric(df[xcol], errors="coerce"),
        ycol: pd.to_numeric(df[ycol], errors="coerce"),
    })
    df["is_make"] = df["ev_result"].eq("made")
    df["is_miss"] = df["ev_result"].eq("missed")

    TRANSFORMS = {
        "NBA": _estimate_transform(df, xcol, ycol, spec="NBA"),
        "FIBA": _estimate_transform(df, xcol, ycol, spec="FIBA"),
    }

    period_max = int(pd.to_numeric(df["period"], errors="coerce").max() or 4)
    content = _page_content(df, period_max)
    nav_items = server.config.get("NAV", [])
    app.layout = PageShell(content, nav_items, current_endpoint=base_pathname)

    @app.callback(
        Output("sc-fig","figure"),
        Output("sc-diag-panel","children"),
        Output("sc-zone-table","data"),
        Output("sc-leaderboard","data"),
        Output("sc-zone-bar","figure"),
        Output("sc-zone-entity-leaderboard","data"),
        # filters
        Input("sc-teams","value"),
        Input("sc-players","value"),
        Input("sc-games","value"),
        Input("sc-periods","value"),
        # view settings
        Input("sc-court","value"),
        Input("sc-spec","value"),
        Input("sc-include","value"),
        Input("sc-flips","value"),
        Input("sc-mode","value"),
        Input("sc-zone-overlay","value"),
        Input("sc-zone-labels","value"),
        Input("sc-fold","value"),
        Input("sc-side-map","value"),
        Input("sc-zone-filter","value"),
        Input("sc-diag","value"),
        # aggregate leaders
        Input("sc-leader-dim","value"),
        Input("sc-leader-sort","value"),
        Input("sc-leader-topn","value"),
        Input("sc-leader-minfga","value"),
        # zone leaders
        Input("sc-zlead-dim","value"),
        Input("sc-zlead-zone","value"),
        Input("sc-zlead-sort","value"),
        Input("sc-zlead-topn","value"),
        Input("sc-zlead-minfga","value"),
        prevent_initial_call=False,
    )
    def _update(teams, players, games, periods,
                court_sel, spec_sel, include_sel, flips_sel, mode_sel,
                zone_overlay_sel, zone_labels_sel, fold_sel, side_map_val, zone_filter, diag_sel,
                leader_dim, leader_sort, leader_topn, leader_minfga,
                zlead_dim, zlead_zone, zlead_sort, zlead_topn, zlead_minfga):

        f = df.copy()
        if teams:   f = f[f["team_name"].isin(teams)]
        if players: f = f[f["player"].isin(players)]
        if games:   f = f[f["game_id"].astype(str).isin(games)]
        if periods:
            lo, hi = periods or [1, period_max]
            f = f[(pd.to_numeric(f["period"], errors="coerce") >= lo) &
                  (pd.to_numeric(f["period"], errors="coerce") <= hi)]

        inc_make = "make" in (include_sel or [])
        inc_miss = "miss" in (include_sel or [])
        f = f[((f["is_make"] & inc_make) | (f["is_miss"] & inc_miss))]
        f = f[f[xcol].notna() & f[ycol].notna()]

        TR0 = TRANSFORMS.get((spec_sel or "NBA").upper(), TRANSFORMS["NBA"]).copy()
        ax, bx, ay, by = TR0["ax"], TR0["bx"], TR0["ay"], TR0["by"]
        if "flipx" in (flips_sel or []):
            cx0 = TR0["C_W"] / 2.0
            x_mid = TR0["ax"] + TR0["bx"] * cx0
            f[xcol] = 2*x_mid - f[xcol]; bx = -bx
        if "flipy" in (flips_sel or []):
            cy0 = TR0["C_H"] / 2.0
            y_mid = TR0["ay"] + TR0["by"] * cy0
            f[ycol] = 2*y_mid - f[ycol]; by = -by
        TR = {**TR0, "ax": ax, "bx": bx, "ay": ay, "by": by}

        # Court coords + zones
        side_top_is_left = (side_map_val == "top_left")
        if not f.empty:
            Cx_all, Cy_all = _data_to_court(f[xcol].to_numpy(dtype=float), f[ycol].to_numpy(dtype=float), TR)
            zones_fine, zones_group = _assign_zones_and_groups(
                Cx_all, Cy_all, TR,
                side_top_is_left=side_top_is_left,
                center_band=6.0
            )
            f = f.assign(_Cx=Cx_all, _Cy=Cy_all, zone=zones_fine, zone_group=zones_group)
        else:
            f = f.assign(_Cx=np.array([]), _Cy=np.array([]), zone=np.array([]), zone_group=np.array([]))

        # Zone filter (fine expansion)
        fine_sel = set(_expand_zone_filter(zone_filter))
        f_z = f[f["zone"].isin(fine_sel)]

        # ---- Build plotting frame with optional half-court fold
        do_fold = ("on" in (fold_sel or [])) or (court_sel != "full")
        half = (court_sel != "full") or do_fold
        left_half = False if court_sel == "half_right" else True

        f_plot = f.copy()
        if do_fold and not f_plot.empty:
            Cx_fold, Cy_fold = _fold_to_halfcourt(f_plot["_Cx"].to_numpy(dtype=float), f_plot["_Cy"].to_numpy(dtype=float), TR, left_half)
            X_fold, Y_fold = _court_to_data(Cx_fold, Cy_fold, TR)
            f_plot = f_plot.assign(**{xcol: X_fold, ycol: Y_fold})

        # ---- Build context columns (opponent/week/score/clock) for hover
        ctx = _build_context_columns(f_plot, spec_sel or "NBA")

        # ---- COURT coords of the *plotted* points (for hover XY after any folding)
        if not f_plot.empty:
            Cx_plot, Cy_plot = _data_to_court(
                f_plot[xcol].to_numpy(dtype=float),
                f_plot[ycol].to_numpy(dtype=float),
                TR
            )
        else:
            Cx_plot, Cy_plot = np.array([]), np.array([])

        # ---- Compose per-shot hover text (includes opponent, score, week)
        if not f_plot.empty:
            def _safe_int(x):
                try:    return int(x)
                except: return "–"

            hov = []
            for ec, imk, pl, tm, zn, cxv, cyv, per, qc, gc, opp, wk, sc, gid, anum in zip(
                f_plot["ev_code"].astype(str),
                f_plot["is_make"],
                f_plot.get("player", pd.Series([""] * len(f_plot))),
                f_plot.get("team_name", pd.Series([""] * len(f_plot))),
                f_plot["zone"],
                Cx_plot, Cy_plot,
                f_plot.get("period", pd.Series([np.nan] * len(f_plot))),
                ctx["q_clock"], ctx["g_clock"],
                ctx["opp"], ctx["week"], ctx["score_str"],
                f_plot.get("game_id", pd.Series([np.nan] * len(f_plot))),
                f_plot.get("actionNumber", pd.Series([np.nan] * len(f_plot))),
            ):
                tag = ("3PT" if ec == "p3" else "2PT")
                res = ("MAKE" if bool(imk) else "MISS")
                hov.append(
                    (
                        f"<b>{tag} {res}</b> — {pl} ({tm})"
                        f"<br><b>Zone:</b> {zn}"
                        f"<br><b>XY (court ft):</b> {float(cxv):.1f}, {float(cyv):.1f}"
                        f"<br><b>Q{_safe_int(per)}</b> {qc} &nbsp;|&nbsp; <b>GTime:</b> {gc}"
                        f"<br><b>Opp:</b> {opp} &nbsp;|&nbsp; <b>Week:</b> {wk}"
                        f"<br><b>Score:</b> {sc}"
                        f"<br><b>Game:</b> {_safe_int(gid)} &nbsp;|&nbsp; <b>Action #</b> {_safe_int(anum)}"
                    )
                )
            f_plot = f_plot.assign(hover=hov)
        else:
            f_plot = f_plot.assign(hover=[])

        # ---- Figure + overlays
        base_shapes = _court_shapes(TR, halfcourt=half, left_half=left_half)

        if "on" in (zone_overlay_sel or []):
            zfg_group = {}
            for gname, grp in f_z.groupby("zone_group"):
                m = _fg_metrics(grp)
                zfg_group[gname] = m["PCT"]
            base_shapes += _zone_overlay_shapes(TR, zfg_group, alpha=0.28)

        hex_shapes = []
        if (mode_sel or "scatter") == "hex":
            hex_shapes = _hexbin_shapes(f_plot, xcol, ycol, TR, metric="efg", R=3.0, min_fga=4)

        fig = go.Figure()
        fig.update_layout(shapes=base_shapes + hex_shapes,
                          hovermode="closest",
                          hoverlabel=dict(bgcolor="#ffffff", font_size=12))

        if (mode_sel or "scatter") == "heat":
            if not f_plot.empty:
                fig.add_trace(_heat(f_plot, xcol, ycol))
        elif (mode_sel or "scatter") == "hex":
            pass
        else:
            misses = f_plot[f_plot["is_miss"]]
            makes  = f_plot[f_plot["is_make"]]
            if not misses.empty: fig.add_trace(_scatter(misses, xcol, ycol, "Miss", is_make=False))
            if not makes.empty:  fig.add_trace(_scatter(makes,  xcol, ycol, "Make", is_make=True))

        # Zone labels (FG% text) — show whenever folding is active
        if (("on" in (zone_labels_sel or [])) and not f_plot.empty
            and (("on" in (fold_sel or [])) or court_sel != "full")):
            text_by_zone = {z: f"{round(_fg_metrics(grp)['PCT']*100,1)}%" for z, grp in f_z.groupby("zone")}
            fig.update_layout(annotations=_zone_annotations_from_data(f_plot, xcol, ycol, text_by_zone))

        # axes/legend cosmetics
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10),
                          legend=dict(orientation="h", y=1.02, x=1.0, xanchor="right", title=None),
                          title=f"Shot Chart (Data-Calibrated, {spec_sel})",
                          plot_bgcolor="#f5f5f5", paper_bgcolor="white")

        # ---- Zone table (fine)
        zone_rows = []
        if len(f_z):
            for z, grp in sorted(f_z.groupby("zone"), key=lambda kv: _fg_metrics(kv[1])["FGA"], reverse=True):
                m = _fg_metrics(grp)
                zone_rows.append(dict(
                    zone=z,
                    FGA=m["FGA"], FGM=m["FGM"], **{"FG%": round(m["PCT"]*100, 1)},
                    **{"3PA": m["P3A"], "3PM": m["P3M"], "eFG%": round(m["eFG"]*100,1), "PTS": m["PTS"]}
                ))

        # ---- Aggregate leaderboards (entity over selected *fine* zones)
        dim_col = "player" if (leader_dim or "player") == "player" else "team_name"
        leader_rows = []
        if len(f_z):
            def agg_frame(grp):
                m = _fg_metrics(grp)
                return pd.Series(dict(FGA=m["FGA"], FGM=m["FGM"], PCT=m["PCT"],
                                      P3A=m["P3A"], P3M=m["P3M"], eFG=m["eFG"], PTS=m["PTS"]))
            tbl = f_z.groupby(dim_col, dropna=False).apply(agg_frame).reset_index().rename(columns={dim_col:"Entity"})
            tbl = tbl[tbl["FGA"] >= (leader_minfga or 0)]
            sort_key = (leader_sort or "eFG")
            tbl = tbl.sort_values(sort_key, ascending=False).head(int(leader_topn or 15))
            for _, r in tbl.iterrows():
                leader_rows.append(dict(
                    Entity=r["Entity"],
                    FGA=int(r["FGA"]), FGM=int(r["FGM"]),
                    **{"FG%": round(float(r["PCT"])*100,1)},
                    **{"3PA": int(r["P3A"]), "3PM": int(r["P3M"]),
                       "eFG%": round(float(r["eFG"])*100,1), "PTS": int(r["PTS"])}
                ))

        # ---- Zone × Entity leaderboards (per *fine* zone)
        z_dim_col = "player" if (zlead_dim or "player") == "player" else "team_name"
        zone_entity_rows = []
        if len(f):
            g = f.copy()
            def agg_z(grp):
                m = _fg_metrics(grp)
                return pd.Series(dict(FGA=m["FGA"], FGM=m["FGM"], PCT=m["PCT"],
                                      P3A=m["P3A"], P3M=m["P3M"], eFG=m["eFG"], PTS=m["PTS"]))
            ztbl = g.groupby([z_dim_col, "zone"], dropna=False).apply(agg_z).reset_index()
            ztbl = ztbl[ztbl["zone"] == (zlead_zone or "Left Corner 3")]
            ztbl = ztbl[ztbl["FGA"] >= (zlead_minfga or 0)]
            ztbl = ztbl.sort_values((zlead_sort or "eFG"), ascending=False).head(int(zlead_topn or 15))
            for _, r in ztbl.iterrows():
                zone_entity_rows.append(dict(
                    Entity=r[z_dim_col],
                    zone=r["zone"],
                    FGA=int(r["FGA"]), FGM=int(r["FGM"]),
                    **{"FG%": round(float(r["PCT"])*100,1)},
                    **{"3PA": int(r["P3A"]), "3PM": int(r["P3M"]),
                       "eFG%": round(float(r["eFG"])*100,1), "PTS": int(r["PTS"])}
                ))

        # ---- Zone bar chart (eFG% by fine zone for current selection)
        bar_fig = go.Figure()
        if zone_rows:
            znames = [r["zone"] for r in zone_rows]
            zefg   = [r["eFG%"] for r in zone_rows]
            bar_fig.add_bar(x=znames, y=zefg, name="eFG%")
            bar_fig.update_layout(title="Zone eFG% (selection)", xaxis_title="", yaxis_title="eFG%")
        bar_fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor="#fafafa")

        # ---- Diagnostics
        diag = []
        if "on" in (diag_sel or []):
            diag = [
                html.H4("Diagnostics"),
                html.Ul(children=[
                    html.Li(f"Detected X/Y: {xcol} / {ycol}"),
                    html.Li(f"Spec: {TR['spec']} — Court (W×H): {TR['C_W']:.2f}×{TR['C_H']:.2f} ft"),
                    html.Li(f"Hoop centers (data est.): left≈({TR['x_left']:.2f}, {TR['y_center']:.2f}), right≈({TR['x_right']:.2f}, {TR['y_center']:.2f})"),
                    html.Li(f"Transform: X = {TR['ax']:.3f} + {TR['bx']:.3f} * Cx ; Y = {TR['ay']:.3f} + {TR['by']:.3f} * Cy"),
                    html.Li(f"Side map: TOP → {'Left' if side_top_is_left else 'Right'} (mirrored on left hoop)"),
                    html.Li(f"Filtered shots: {len(f):,} | For tables/leaderboards (zone-filtered): {len(f_z):,}"),
                    html.Li(f"Fine zone options: {', '.join(_zone_labels_fine())}"),
                    html.Li(f"Aggregates understood: {', '.join(_zone_labels_aggregate())}"),
                ])
            ]

        return fig, diag, zone_rows, leader_rows, bar_fig, zone_entity_rows

    return app
