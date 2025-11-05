from __future__ import annotations
from typing import List, Dict
from dash import html

def build_sidebar(nav_items: List[Dict], current_endpoint: str) -> html.Div:
    """
    Render a left sidebar using app.config['NAV'].
    Highlights the currently active Dash page by matching endpoint.
    """
    links = []
    for it in nav_items:
        endpoint = it.get("endpoint", "")
        name = it.get("name", endpoint.strip("/")) or endpoint.strip("/")
        cls = "nav-item" + (" active" if endpoint == current_endpoint else "")
        links.append(html.A(name, href=endpoint, className=cls))

    return html.Div(className="sidebar", children=[
        html.Div(className="brand", children="Analytics Hub"),
        html.Nav(className="nav", children=links),
        html.Div(className="foot", children=[
            html.Span("DATA_ROOT aware • Dash+Flask")
        ])
    ])

def page_shell(children, nav_items: List[Dict], current_endpoint: str):
    """
    2-column responsive layout: sticky left nav + main content on the right.
    Styles come from assets/app_shell.css.
    """
    return html.Div(className="app-shell", children=[
        build_sidebar(nav_items, current_endpoint),
        html.Div(className="main", children=children),
    ])
