# dashapps/player_profile.py
from __future__ import annotations

import re, urllib.parse, unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table

# Reuse the shot-chart engine you already have
from .shot_chart import (
    _load_pbp, _detect_xy, _estimate_transform,
    _court_shapes, _hexbin_shapes, _fg_metrics,
    _data_to_court, _court_to_data,
    _assign_zones_and_groups, _zone_annotations_from_data,
    _expand_zone_filter, _zone_filter_options, _build_context_columns
)

# Optional: use players list to resolve id-based keys -> display names if needed
try:
    from .players import _PLAYERS_RAW  # may be empty but import should succeed
except Exception:
    _PLAYERS_RAW = pd.DataFrame(columns=["name", "team_display", "team_key"])


# -----------------------------
# small text helpers (local)
# -----------------------------
def _clean_text(s: object) -> str:
    if s is None:
        return ""
    t = str(s).replace("\u00A0", " ")
    t = unicodedata.normalize("NFKC", t).strip()
    return re.sub(r"\s+", " ", t)

def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def _resolve_team_from_slug(pbp_df: pd.DataFrame, team_key: Optional[str]) -> Optional[str]:
    """Map a team_key (slug) -> team_name seen in pbp."""
    if not team_key:
        return None
    tslug = _slugify(_clean_text(team_key))
    if "team_name" not in pbp_df.columns or pbp_df["team_name"].isna().all():
        return None
    # build slug map from pbp team names
    tmap = {}
    for nm in pbp_df["team_name"].dropna().unique():
        tmap[_slugify(str(nm))] = str(nm)
    return tmap.get(tslug)


def _resolve_player_from_key(
    pbp_df: pd.DataFrame,
    player_key: Optional[str]
) -> Optional[str]:
    """
    Accepts:
      - 'name-<slug>'  (from players.py)
      - '<free text name>'
      - 'id-<something>' (try to map from _PLAYERS_RAW if present)
    Returns display name present under pbp_df['player'].
    """
    if not player_key:
        return None

    raw = _clean_text(player_key)
    # If players.py sent name-based key (preferred)
    if raw.startswith("name-"):
        slug = raw[len("name-"):]
    elif raw.startswith("id-"):
        # Try to map id-… via players dataframe
        pid_slug = raw[len("id-"):]
        if not _PLAYERS_RAW.empty:
            # search any column that looked like an id when building keys
            cand_cols = ["player_id", "playerId", "personId", "player_code", "id", "externalId"]
            tmp = _PLAYERS_RAW.copy()
            for c in cand_cols:
                if c in tmp.columns:
                    tmp[c] = tmp[c].astype(str).map(lambda v: _slugify(v))
            # find row whose any id column slug equals pid_slug
            row = tmp.loc[(tmp[cand_cols].astype(str) == pid_slug).any(axis=1)]
            if not row.empty:
                nm = str(row.iloc[0].get("name", "")).strip()
                if nm:
                    slug = _slugify(nm)
                else:
                    slug = pid_slug  # fallback to try direct slug match
            else:
                slug = pid_slug
        else:
            slug = pid_slug
    else:
        # Treat as plain name
        slug = _slugify(raw)

    # Build slug map from pbp_df
    if "player" not in pbp_df.columns or pbp_df["player"].isna().all():
        return None
    pmap = {}
    for nm in pbp_df["player"].dropna().unique():
        pmap[_slugify(str(nm))] = str(nm)

    return pmap.get(slug)


# -----------------------------
# Lightweight figure builder
# -----------------------------
def _build_player_figure(
    df: pd.DataFrame, xcol: str, ycol: str,
    player_name: str,
    team_name: Optional[str],
    spec_sel: str = "NBA",
    court_sel: str = "half_left",   # default: half (like your reference img)
    mode_sel: str = "hex",          # default: hex eFG
    include_sel: Optional[List[str]] = None,
    zone_overlay: bool = True,
    zone_labels: bool = True,
) -> Tuple["go.Figure", List[dict], List[dict], Dict[str, float]]:
    include_sel = include_sel or ["make", "miss"]

    # 1) Filter
    f = df[(df["player"] == player_name)]
    if team_name:
        f = f[f["team_name"] == team_name]

    f = f[((f["ev_result"].eq("made") & ("make" in include_sel)) |
           (f["ev_result"].eq("missed") & ("miss" in include_sel)))]
    f = f[f[xcol].notna() & f[ycol].notna()]

    # 2) Transform & court coords
    TR0 = _estimate_transform(df, xcol, ycol, spec=spec_sel or "NBA")
    TR = TR0.copy()

    Cx_all, Cy_all = _data_to_court(
        f[xcol].to_numpy(dtype=float),
        f[ycol].to_numpy(dtype=float),
        TR
    ) if not f.empty else (np.array([]), np.array([]))

    zones_fine, zones_group = _assign_zones_and_groups(Cx_all, Cy_all, TR, side_top_is_left=True, center_band=6.0) if len(Cx_all) else (np.array([]), np.array([]))
    f = f.assign(_Cx=Cx_all, _Cy=Cy_all, zone=zones_fine, zone_group=zones_group)

    # Half / Full
    half = court_sel != "full"
    left_half = (court_sel != "half_right")
    if half and not f.empty:
        # Fold to selected half
        from .shot_chart import _fold_to_halfcourt  # local import to avoid clutter
        Cx_fold, Cy_fold = _fold_to_halfcourt(f["_Cx"].to_numpy(dtype=float), f["_Cy"].to_numpy(dtype=float), TR, left_half)
        X_fold, Y_fold  = _court_to_data(Cx_fold, Cy_fold, TR)
        f = f.assign(**{xcol: X_fold, ycol: Y_fold})

    # Context for hover
    ctx = _build_context_columns(f, spec_sel or "NBA")

    # Hover strings
    hov = []
    for ec, imk, tm, zn, per, qc, gc, opp, wk, sc in zip(
        f["ev_code"].astype(str),
        f["ev_result"].eq("made"),
        f.get("team_name", pd.Series([""]*len(f))),
        f["zone"],
        f.get("period", pd.Series([np.nan]*len(f))),
        ctx["q_clock"], ctx["g_clock"], ctx["opp"], ctx["week"], ctx["score_str"]
    ):
        tag = "3PT" if ec == "p3" else "2PT"
        res = "MAKE" if imk else "MISS"
        per_s = f"Q{int(per)}" if pd.notna(per) else "Q–"
        hov.append(
            f"<b>{tag} {res}</b> — {player_name} ({tm})"
            f"<br><b>Zone:</b> {zn}"
            f"<br><b>{per_s}</b> {qc} &nbsp;|&nbsp; <b>GTime:</b> {gc}"
            f"<br><b>Opp:</b> {opp} &nbsp;|&nbsp; <b>Week:</b> {wk}"
            f"<br><b>Score:</b> {sc}"
        )
    f = f.assign(hover=hov)

    # 3) Base court
    import plotly.graph_objects as go
    fig = go.Figure()
    base_shapes = _court_shapes(TR, halfcourt=half, left_half=left_half)

    # 4) Mode
    if mode_sel == "heat":
        if not f.empty:
            fig.add_trace(go.Histogram2d(x=f[xcol], y=f[ycol], nbinsx=60, nbinsy=60,
                                         colorscale="YlOrRd", opacity=0.9, name="Density", showscale=True))
    elif mode_sel == "hex":
        base_shapes += _hexbin_shapes(f, xcol, ycol, TR, metric="efg", R=3.0, min_fga=4)
    else:  # scatter
        makes  = f[f["ev_result"].eq("made")]
        misses = f[f["ev_result"].eq("missed")]
        if not misses.empty:
            fig.add_trace(go.Scattergl(
                x=misses[xcol], y=misses[ycol], mode="markers", name="Miss",
                marker=dict(symbol="x", size=10, line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
                            color="#c0392b"),
                opacity=0.85, hovertemplate="%{text}<extra></extra>", text=misses["hover"]
            ))
        if not makes.empty:
            fig.add_trace(go.Scattergl(
                x=makes[xcol], y=makes[ycol], mode="markers", name="Make",
                marker=dict(symbol="circle", size=9, line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
                            color="#1f9e49"),
                opacity=0.9, hovertemplate="%{text}<extra></extra>", text=makes["hover"]
            ))

    # 5) Zone overlay + labels
    if zone_overlay and not f.empty:
        # aggregate by zone_group for overlay color (FG%)
        zfg_group: Dict[str, float] = {}
        for gname, grp in f.groupby("zone_group"):
            m = _fg_metrics(grp)
            zfg_group[gname] = m["PCT"]
        from .shot_chart import _zone_overlay_shapes
        base_shapes += _zone_overlay_shapes(TR, zfg_group, alpha=0.28)

    fig.update_layout(shapes=base_shapes,
                      hovermode="closest",
                      hoverlabel=dict(bgcolor="#ffffff", font_size=12),
                      margin=dict(l=10, r=10, t=48, b=10),
                      title=f"{player_name} — Shot Chart ({spec_sel})",
                      plot_bgcolor="#f5f5f5", paper_bgcolor="white",
                      legend=dict(orientation="h", y=1.02, x=1.0, xanchor="right", title=None))
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    # Labels (half-court typical)
    zone_rows, kpis = [], dict(FGA=0, FGM=0, eFG=0.0, PTS=0, P3A=0, P3M=0)
    if not f.empty:
        if zone_labels and half:
            text_by_zone = {z: f"{round(_fg_metrics(g)['PCT']*100,1)}%" for z, g in f.groupby("zone")}
            fig.update_layout(annotations=_zone_annotations_from_data(f, xcol, ycol, text_by_zone))
        # zone table + KPIs
        for z, grp in sorted(f.groupby("zone"), key=lambda kv: _fg_metrics(kv[1])["FGA"], reverse=True):
            m = _fg_metrics(grp)
            zone_rows.append(dict(
                zone=z, FGA=m["FGA"], FGM=m["FGM"],
                **{"FG%": round(m["PCT"]*100, 1)},
                **{"3PA": m["P3A"], "3PM": m["P3M"], "eFG%": round(m["eFG"]*100,1), "PTS": m["PTS"]}
            ))
        m_all = _fg_metrics(f)
        kpis = dict(FGA=m_all["FGA"], FGM=m_all["FGM"], eFG=round(m_all["eFG"]*100,1),
                    PTS=m_all["PTS"], P3A=m_all["P3A"], P3M=m_all["P3M"])

    return fig, zone_rows, [{"label": z, "value": z} for z in sorted(f["zone"].unique())], kpis


# -----------------------------
# Page layout
# -----------------------------
def create_dash_player_profile(server, base_pathname: str = "/player_profile/") -> Dash:
    app = dash.Dash(
        __name__,
        server=server,
        routes_pathname_prefix=base_pathname,
        requests_pathname_prefix=base_pathname,
        suppress_callback_exceptions=True,
        title="Player Profile"
    )

    df = _load_pbp()
    xcol, ycol = _detect_xy(df)
    # normalize critical columns
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df["ev_result"] = df["ev_result"].astype(str).str.lower()
    if "player" not in df.columns:
        df["player"] = df.get("player_name_from_roster", "Unknown Player")

    # URL and content
    app.layout = html.Div([
        dcc.Location(id="pp-url"),
        html.Div(id="pp-header"),
        html.Div(style={"display":"grid", "gridTemplateColumns":"repeat(6,1fr)", "gap":"10px"}, children=[
            html.Div([html.Label("Court"), dcc.RadioItems(id="pp-court", options=[
                {"label":" Full","value":"full"},
                {"label":" Half (Left)","value":"half_left"},
                {"label":" Half (Right)","value":"half_right"},
            ], value="half_left", inputStyle={"marginRight":"6px","marginLeft":"8px"})]),
            html.Div([html.Label("Spec"), dcc.RadioItems(id="pp-spec",
                options=[{"label":" NBA","value":"NBA"},{"label":" FIBA","value":"FIBA"}],
                value="NBA", inputStyle={"marginRight":"6px","marginLeft":"8px"})]),
            html.Div([html.Label("Render"),
                dcc.RadioItems(id="pp-mode",
                    options=[{"label":" Scatter","value":"scatter"},{"label":" Heat","value":"heat"},{"label":" Hex (eFG)","value":"hex"}],
                    value="hex", inputStyle={"marginRight":"6px","marginLeft":"8px"})]),
            html.Div([html.Label("Include"),
                dcc.Checklist(id="pp-include",
                    options=[{"label":" Makes","value":"make"},{"label":" Misses","value":"miss"}],
                    value=["make","miss"]) ]),
            html.Div([html.Label("Zone overlay"), dcc.Checklist(id="pp-zoverlay",
                options=[{"label":" Show FG% overlay","value":"on"}], value=["on"])]),
            html.Div([html.Label("Zone labels"), dcc.Checklist(id="pp-zlabels",
                options=[{"label":" Show % on half-court","value":"on"}], value=["on"])]),
        ]),
        html.Hr(style={"margin":"8px 0"}),
        dcc.Graph(id="pp-fig", style={"height":"720px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr", "gap":"16px", "alignItems":"start"}, children=[
            html.Div(children=[
                html.H3("Zones — Player Selection"),
                dash_table.DataTable(
                    id="pp-zone-table",
                    columns=[
                        {"name":"Zone","id":"zone"},
                        {"name":"FGA","id":"FGA","type":"numeric"},
                        {"name":"FGM","id":"FGM","type":"numeric"},
                        {"name":"FG%","id":"FG%","type":"numeric"},
                        {"name":"3PA","id":"3PA","type":"numeric"},
                        {"name":"3PM","id":"3PM","type":"numeric"},
                        {"name":"eFG%","id":"eFG%","type":"numeric"},
                        {"name":"PTS","id":"PTS","type":"numeric"},
                    ],
                    data=[], page_size=12, sort_action="native",
                    style_table={"overflowX":"auto"},
                    style_cell={"padding":"6px","fontSize":"14px"},
                ),
            ]),
            html.Div(id="pp-kpis", children=[]),
        ]),
        html.Div(id="pp-deeplink", style={"marginTop":"10px"}),
    ])

    # ---- parse URL + render everything
    @app.callback(
        Output("pp-header","children"),
        Output("pp-fig","figure"),
        Output("pp-zone-table","data"),
        Output("pp-kpis","children"),
        Output("pp-deeplink","children"),
        Input("pp-url","search"),
        Input("pp-court","value"),
        Input("pp-spec","value"),
        Input("pp-mode","value"),
        Input("pp-include","value"),
        Input("pp-zoverlay","value"),
        Input("pp-zlabels","value"),
        prevent_initial_call=False
    )
    def _render(search, court_sel, spec_sel, mode_sel, include_sel, zoverlay, zlabels):
        qs = urllib.parse.parse_qs((search or "").lstrip("?"))
        q_player = (qs.get("player") or [""])[0]
        q_team   = (qs.get("team") or [""])[0]

        # Resolve player/team
        player_name = _resolve_player_from_key(df, q_player) or _clean_text(q_player)
        team_name   = _resolve_team_from_slug(df, q_team)

        if not player_name:
            header = html.Div([
                html.H2("Player Profile — Shot Chart"),
                html.P("No player selected. Append ?player=<name-or-key>&team=<team-key> to the URL "
                       "(this page also works with your /players/ links).",
                       style={"color":"#a00"})
            ])
            empty_fig = dash.no_update
            return header, empty_fig, [], [], ""

        title_bits = [player_name]
        if team_name:
            title_bits.append(f"({team_name})")
        header = html.H2(" ".join(title_bits))

        fig, zone_rows, _zn_opts, kpis = _build_player_figure(
            df, xcol, ycol,
            player_name=player_name, team_name=team_name,
            spec_sel=(spec_sel or "NBA"),
            court_sel=(court_sel or "half_left"),
            mode_sel=(mode_sel or "hex"),
            include_sel=(include_sel or ["make","miss"]),
            zone_overlay=("on" in (zoverlay or [])),
            zone_labels=("on" in (zlabels or [])),
        )

        # KPIs card
        kpi = html.Div([
            html.H3("Overall (current selection)"),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":"8px"}, children=[
                _kpi("FG", f"{kpis.get('FGM',0)}/{kpis.get('FGA',0)}"),
                _kpi("eFG%", f"{kpis.get('eFG',0.0):.1f}"),
                _kpi("3P", f"{kpis.get('P3M',0)}/{kpis.get('P3A',0)}"),
                _kpi("PTS", f"{kpis.get('PTS',0)}"),
            ])
        ], style={"border":"1px solid #eee","borderRadius":"10px","padding":"10px"})

        # Deep link to the full shot-chart app (pre-filter by readable name)
        shot_url = f"/shot_chart/?{urllib.parse.urlencode({'player': player_name, 'team': team_name or ''})}"
        deeplink = html.A("Open in full Shot Chart app →", href=shot_url, style={"fontWeight":"600"})

        return header, fig, zone_rows, kpi, deeplink

    return app


def _kpi(label: str, value: str) -> html.Div:
    return html.Div([
        html.Div(str(label), style={"color":"#666","fontSize":"12px"}),
        html.Div(str(value), style={"fontSize":"18px","fontWeight":"700"})
    ], style={"background":"#fafbff","borderRadius":"10px","padding":"8px"} )
