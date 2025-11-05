from __future__ import annotations
import os
from importlib import import_module
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from flask import (
    Flask, render_template, url_for, request, abort, make_response
)
import pandas as pd
import inspect

# -----------------------------------------------------------------------------
# NAV
# -----------------------------------------------------------------------------
NAV = [
    {
        "name": "Lineup Stints",
        "endpoint": "/lineup_stints/",
        "module": "dashapps.lineup_stints",
        "factory": "create_dash_lineup_stints",
        "desc": "Visualize lineup sequences, stints, and plus-minus in context.",
        "emoji": "🧩",
    },
    {
        "name": "Assist Combos",
        "endpoint": "/assist_combos/",
        "module": "dashapps.assist_combos",
        "factory": "create_dash_assist_combos",
        "desc": "Find high-frequency passer–scorer tandems and hotspots.",
        "emoji": "🤝",
    },


    {
        "name": "Player Game Query",
        "endpoint": "/player_game_query/",
        "module": "dashapps.player_game_query",
        "factory": "create_dash_player_game_query",
        "desc": "Query player game-by-game data across seasons.",
        "emoji": "🧮",
    },
    {
        "name": "All-Time Leaderboards",
        "endpoint": "/all_time/",
        "module": "dashapps.all_time",
        "factory": "create_dash_all_time",
        "desc": "Career totals, per-game & per-36 with min-attempt filters.",
        "emoji": "📚",
    },
    {
        "name": "Scoring Leaderboards",
        "endpoint": "/scoring_leaders/",
        "module": "dashapps.scoring_leaders",
        "factory": "create_dash_scoring_leaders",
        "desc": "Leaders by points, efficiency, and scoring profiles.",
        "emoji": "🏆",
    },
    {
        "name": "Shot Chart",
        "endpoint": "/shot_chart/",
        "module": "dashapps.shot_chart",
        "factory": "create_dash_shot_chart",
        "desc": "XY shot maps with clustering and make–miss overlays.",
        "emoji": "🎯",
    },
    {
        "name": "Shooting Performance",
        "endpoint": "/shooting_performance/",
        "module": "dashapps.shooting_performance",
        "factory": "create_dash_shooting_performance",
        "desc": "Team/Player FG%, eFG%, PIP/FB/2ndCh shares, and subtypes.",
        "emoji": "📊",
    },
    {
        "name": "Players",
        "endpoint": "/players/",
        "module": "dashapps.players",
        "factory": "create_dash_players",
        "emoji": "👤",
    },
    {
        "name": "Team Game Query",
        "endpoint": "/team_game_query/",
        "module": "dashapps.team_game_query",
        "factory": "create_dash_team_game_query",
        "desc": "Query team game stats across seasons with flexible filters.",
        "emoji": "🧪",
    },
    {
        "name": "Game Extractor",
        "endpoint": "/extractor/",
        "module": "dashapps.extractor_ui",
        "factory": "create_dash_extractor",
        "desc": "Run stints/possessions extraction for a game or a whole week.",
        "emoji": "🛠️",
    },
]

# -----------------------------------------------------------------------------
# Helpers to robustly patch Dash apps whose layout is None (Dash>=2.15)
# -----------------------------------------------------------------------------
def _patch_dash_layouts_in_module(mod, module_name: str) -> List[str]:
    patched: List[str] = []
    try:
        from dash import Dash, html
    except Exception:
        return patched

    for name, obj in vars(mod).items():
        if isinstance(obj, Dash):
            try:
                if getattr(obj, "layout", None) is None:
                    obj.layout = html.Div([
                        html.H3(f"{module_name}.{name} – placeholder layout"),
                        html.P("This dashboard did not set a layout during init. Placeholder attached by launcher."),
                    ])
                    patched.append(name)
            except Exception:
                continue
    return patched


def _mount_dash(app: Flask, module_name: str, factory_name: str, endpoint: str) -> Tuple[bool, Optional[str]]:
    try:
        mod = import_module(module_name)
        factory = getattr(mod, factory_name)
        dash_app = None

        tried_sigs: List[str] = []
        try:
            tried_sigs.append("(server=app, base_pathname=endpoint)")
            dash_app = factory(server=app, base_pathname=endpoint)
        except TypeError:
            try:
                tried_sigs.append("(server=app)")
                dash_app = factory(server=app)
            except TypeError:
                try:
                    tried_sigs.append("(app)")
                    dash_app = factory(app)
                except TypeError:
                    tried_sigs.append("()")
                    dash_app = factory()

        if dash_app is not None:
            try:
                from dash import html
                if getattr(dash_app, "layout", None) is None:
                    dash_app.layout = html.Div("Placeholder layout")
            except Exception:
                pass

        patched_names = _patch_dash_layouts_in_module(mod, module_name)
        if patched_names:
            app.logger.info("Patched missing layouts in %s: %s", module_name, ", ".join(patched_names))

        return True, None

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _file_health(data_root: str) -> Dict[str, Any]:
    root = Path(data_root)
    candidates = [
        root / "_cumulative" / "pbp_all.csv",
        root / "pbp_all.csv",
        Path("/mnt/data/pbp_all.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {"found": False, "path": None, "size_kb": 0}

    try:
        sz = path.stat().st_size
    except Exception:
        sz = 0
    return {"found": True, "path": str(path), "size_kb": int(sz / 1024)}


# -----------------------------------------------------------------------------
# App factory + routes
# -----------------------------------------------------------------------------
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
    app.url_map.strict_slashes = False

    # Ensure a default data root; accessible in the UI
    os.environ.setdefault("DATA_ROOT", "out_data")
    data_root = os.environ.get("DATA_ROOT", "out_data")

    app.config["NAV"] = NAV
    app.config["MOUNT_STATUS"] = {}

    # Eagerly mount all Dash apps; record status for the landing page badges.
    for it in NAV:
        ok, err = _mount_dash(app, it["module"], it["factory"], it["endpoint"])
        app.config["MOUNT_STATUS"][it["endpoint"]] = {"ok": ok, "error": err}

    @app.route("/")
    def index():
        items: List[Dict[str, Any]] = []
        for i, it in enumerate(app.config["NAV"], start=1):
            st = app.config["MOUNT_STATUS"].get(it["endpoint"], {"ok": False, "error": "Unknown"})
            items.append({
                "name": it["name"],
                "endpoint": it["endpoint"],
                "desc": it.get("desc") or "Open dashboard",
                "emoji": it.get("emoji") or "📈",
                "ok": st["ok"],
                "error": st["error"],
                "shortcut": i if i <= 9 else None,
            })

        health = _file_health(data_root)
        try:
            import dash  # type: ignore
            dash_version = getattr(dash, "__version__", "unknown")
        except Exception:
            dash_version = "not installed"

        return render_template(
            "index.html",
            items=items,
            data_root=data_root,
            health=health,
            dash_version=dash_version,
        )

    # ---------------------- Player Profile page ----------------------
    # ---------------------- Player Profile page ----------------------
    # ---------------------- Player Profile page ----------------------
# ---------------------- Player Profile page ----------------------
    # ---------------------- Player Profile page ----------------------
    @app.route("/player_profile/")
    def player_profile():
        """
        /player_profile/?player=<player_key>&team=<team_key>&wlo=1&whi=6&download=1
                          &scope=last5|last10|season|last&n=7
        Adds 2FG, eFG%, PF, FD, AST/TOV% to the averages panel.
        Also renders a Scoring Style table (fast break, second chance, paint,
        assisted/unassisted, layup & jumpshot subtypes) scoped to the same window.
        """
        from flask import request, abort, make_response, render_template
        import pandas as pd, numpy as np

        player_key = (request.args.get("player") or "").strip()
        team_key   = (request.args.get("team") or "").strip()
        wlo        = request.args.get("wlo", type=int)
        whi        = request.args.get("whi", type=int)
        want_csv   = request.args.get("download")
        scope      = (request.args.get("scope") or "season").strip().lower()
        n_custom   = request.args.get("n", type=int)

        if not player_key:
            abort(400, "Missing ?player=<player_key>")

        try:
            from dashapps import players as P  # reuse loaded dataset + helpers
        except Exception as e:
            abort(500, f"Could not import dashapps.players: {e}")

        df = getattr(P, "_PLAYERS_RAW", pd.DataFrame())
        if df is None or df.empty:
            abort(500, "Players dataset is empty or not loaded.")

        # Prefer specific team if provided; otherwise any team row for that player
        m = (df["player_key"] == player_key)
        if team_key:
            m &= (df.get("team_key", "") == team_key)

        rows = df.loc[m].copy()
        if rows.empty:
            rows = df[df["player_key"] == player_key].copy()
        if rows.empty:
            abort(404, "Player not found in dataset.")

        name = rows.get("name", pd.Series([player_key])).dropna().iloc[-1]
        team_disp = rows.get("team_display", pd.Series([team_key])).dropna().iloc[-1]
        team_key_eff = rows.get("team_key", pd.Series([team_key])).dropna().iloc[-1]
        uid = f"{team_key_eff}|{player_key}"

        # ========= Build expanded game log (uses P._build_game_log) =========
        gl = P._build_game_log(uid, limit=9999)

        # Optional week slice → defines "Season" bounds for this profile
        if isinstance(gl, pd.DataFrame) and not gl.empty and ("Week" in gl.columns):
            if (wlo is not None) and (whi is not None):
                gl = gl[gl["Week"].between(wlo, whi, inclusive="both")]

        # ========= CSV =========
        if want_csv and isinstance(gl, pd.DataFrame):
            csv = gl.to_csv(index=False)
            resp = make_response(csv)
            safe_name = str(name).replace(" ", "_")
            safe_team = str(team_disp).replace(" ", "_")
            resp.headers["Content-Type"] = "text/csv; charset=utf-8"
            resp.headers["Content-Disposition"] = f'attachment; filename="{safe_name}_{safe_team}_gamelog.csv"'
            return resp

        # ========= HTML table (keep FG/2P/3P/FT on one line) =========
        if isinstance(gl, pd.DataFrame) and not gl.empty:
            gl.columns = [str(c) for c in gl.columns]
            for col in ["Date", "Week", "Opponent", "H/A", "MIN",
                        "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV",
                        "PF", "FD",
                        "FG", "2P", "3P", "FT", "+/-"]:
                if col not in gl.columns:
                    gl[col] = ""

            for sp in ["FG", "2P", "3P", "FT"]:
                if sp in gl.columns:
                    gl[sp] = gl[sp].astype(str).map(lambda s: f"<span class='nowrap'>{s}</span>")

            order = ["Date", "Week", "Opponent", "H/A", "MIN",
                    "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV",
                    "PF", "FD",
                    "FG", "2P", "3P", "FT", "+/-"]
            gl = gl.loc[:, [c for c in order if c in gl.columns]]

            table_html = gl.to_html(
                index=False, header=True, border=0,
                classes=["pl-log", "tbl", "tbl-compact"], na_rep="",
                escape=False  # keep <span class='nowrap'> intact
            )
        else:
            table_html = "<p>No games found.</p>"

        # ========= Averages panel (2FG, eFG%, PF, FD, AST/TOV%) =========
        def _parse_ma(s: pd.Series) -> tuple[pd.Series, pd.Series]:
            if s is None or s.empty:
                z = pd.Series([], dtype=float)
                return z, z
            x = s.astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
            x = x.fillna(0).astype(float)
            if x.shape[1] < 2:
                made = pd.to_numeric(x.iloc[:, 0], errors="coerce").fillna(0.0) if x.shape[1] >= 1 else pd.Series(0.0, index=s.index)
                att  = pd.Series(0.0, index=s.index)
                return made, att
            return x.iloc[:, 0], x.iloc[:, 1]

        def _to_num(col): return pd.to_numeric(col, errors="coerce").fillna(0.0)

        def _scope_slice(gl_df: pd.DataFrame, scope: str, n_custom: int | None) -> tuple[pd.DataFrame, str, int]:
            if gl_df.empty:
                return gl_df, "Season", 0
            _dt = pd.to_datetime(gl_df["Date"], errors="coerce")
            tmp = gl_df.assign(_dt=_dt).sort_values("_dt", ascending=False, na_position="last").drop(columns=["_dt"])
            if scope == "last5":  return tmp.head(5),  "Last 5",  min(5,  len(tmp))
            if scope == "last10": return tmp.head(10), "Last 10", min(10, len(tmp))
            if scope == "last":
                n = int(n_custom) if (n_custom and n_custom > 0) else 5
                return tmp.head(n), f"Last {n}", min(n, len(tmp))
            return tmp, "Season", len(tmp)

        def _compute_avgs(gl_df: pd.DataFrame) -> dict[str, str]:
            if gl_df.empty:
                return {k: "—" for k in
                        ["MIN","PTS","REB","OREB","DREB","AST","STL","BLK","TOV","PF","FD",
                        "2FGM","2FGA","2FG%","FG%","3P%","FT%","eFG%","AST/TOV%"]}

            n_games = max(1, len(gl_df))

            # Volume per game
            MIN  = _to_num(gl_df.get("MIN", 0.0)).mean()
            PTS  = _to_num(gl_df.get("PTS", 0.0)).mean()
            REB  = _to_num(gl_df.get("REB", 0.0)).mean()
            OREB = _to_num(gl_df.get("OREB", 0.0)).mean()
            DREB = _to_num(gl_df.get("DREB", 0.0)).mean()
            AST  = _to_num(gl_df.get("AST", 0.0)).mean()
            STL  = _to_num(gl_df.get("STL", 0.0)).mean()
            BLK  = _to_num(gl_df.get("BLK", 0.0)).mean()
            TOV  = _to_num(gl_df.get("TOV", 0.0)).mean()
            PF   = _to_num(gl_df.get("PF", 0.0)).mean() if "PF" in gl_df.columns else float("nan")
            FD   = _to_num(gl_df.get("FD", 0.0)).mean() if "FD" in gl_df.columns else float("nan")

            # Shooting splits
            FGM, FGA = _parse_ma(gl_df.get("FG", pd.Series(dtype=object)))
            TPM, TPA = _parse_ma(gl_df.get("3P", pd.Series(dtype=object)))
            FTM, FTA = _parse_ma(gl_df.get("FT", pd.Series(dtype=object)))

            sFGM, sFGA, sTPM, sTPA, sFTM, sFTA = FGM.sum(), FGA.sum(), TPM.sum(), TPA.sum(), FTM.sum(), FTA.sum()

            # 2FG totals and rates
            TWO_M = max(0.0, sFGM - sTPM)
            TWO_A = max(0.0, sFGA - sTPA)
            TWO_M_pg = TWO_M / n_games
            TWO_A_pg = TWO_A / n_games
            TWO_pct  = float("nan") if TWO_A <= 0 else 100.0 * (TWO_M / TWO_A)

            FG_pct = float("nan") if sFGA <= 0 else 100.0 * (sFGM / sFGA)
            TP_pct = float("nan") if sTPA <= 0 else 100.0 * (sTPM / sTPA)
            FT_pct = float("nan") if sFTA <= 0 else 100.0 * (sFTM / sFTA)

            # eFG% = (FGM + 0.5 * 3PM) / FGA
            eFG_pct = float("nan") if sFGA <= 0 else 100.0 * ((sFGM + 0.5 * sTPM) / sFGA)

            # Assist-to-Turnover % (sum-based)
            sAST = _to_num(gl_df.get("AST", 0.0)).sum()
            sTOV = _to_num(gl_df.get("TOV", 0.0)).sum()
            AT_pct = float("nan") if sTOV <= 0 else 100.0 * (sAST / sTOV)

            def fmt1(x): return f"{x:.1f}" if np.isfinite(x) else "—"
            def fmt2(x): return f"{x:.2f}" if np.isfinite(x) else "—"

            return {
                "MIN": fmt1(MIN),
                "PTS": fmt1(PTS),
                "REB": fmt1(REB),
                "OREB": fmt1(OREB),
                "DREB": fmt1(DREB),
                "AST": fmt1(AST),
                "STL": fmt1(STL),
                "BLK": fmt1(BLK),
                "TOV": fmt1(TOV),
                "PF":  fmt1(PF) if np.isfinite(PF) else "—",
                "FD":  fmt1(FD) if np.isfinite(FD) else "—",
                "2FGM": fmt1(TWO_M_pg),
                "2FGA": fmt1(TWO_A_pg),
                "2FG%": fmt2(TWO_pct),
                "FG%":  fmt2(FG_pct),
                "3P%":  fmt2(TP_pct),
                "FT%":  fmt2(FT_pct),
                "eFG%": fmt2(eFG_pct),
                "AST/TOV%": fmt2(AT_pct),
            }

        scoped, scope_label, n_games = _scope_slice(gl, scope, n_custom)
        avgs = _compute_avgs(scoped)

        avg_rows = "".join(
            f"<div class='avg-item'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
            for k, v in avgs.items()
        )
        averages_html = f"""
          <div class="avg-head">
            <strong>Averages</strong>
            <span class="muted">({scope_label}, {n_games} game{'s' if n_games != 1 else ''})</span>
          </div>
          <div class="avg-grid">{avg_rows}</div>
        """

        # ========= Scoring Style table (same scope) =========
        scoped_dates = []
        if isinstance(scoped, pd.DataFrame) and not scoped.empty and ("Date" in scoped.columns):
            scoped_dates = [d for d in scoped["Date"].dropna().astype(str).tolist() if d]

        try:
            style_df = P._build_scoring_style(uid, include_dates=scoped_dates)
        except Exception:
            style_df = pd.DataFrame(columns=["Category","Subtype","FGM","3PM","PTS","Share%"])

        if style_df is None or style_df.empty:
            scoring_style_html = "<p class='muted'>No scoring-style breakdown available for this scope.</p>"
        else:
            sdf = style_df.copy()
            sdf["Share%"] = sdf["Share%"].map(lambda x: f"{x:.2f}%")
            scoring_style_html = sdf.to_html(
                index=False, border=0, classes=["tbl","tbl-compact","tbl-tight"], na_rep=""
            )

        # ========= Render =========
        return render_template(
            "player_profile.html",
            name=name,
            team=team_disp,
            table_html=table_html,
            averages_html=averages_html,
            scoring_style_html=scoring_style_html,  # <- NEW
            q_player=player_key,
            q_team=team_key,
            q_wlo=wlo,
            q_whi=whi,
            q_scope=scope,
        )



    return app


app = create_app()

if __name__ == "__main__":
     app.run(host="127.0.0.1", port=8001, debug=True, use_reloader=False)
