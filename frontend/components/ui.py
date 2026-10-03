"""Reusable UI building blocks + Plotly house style."""
from __future__ import annotations

import html
import math
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------- palette
ACCENT = "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#b9b8b0"
INK, INK2, MUTED_INK, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#ecebe6", "#c3c2b7"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
TIER_STATUS = {"Low": "good", "Medium": "warning", "High": "serious", "Very High": "critical",
               "Critical": "critical", "Normal": "good", "Watch": "warning", "Suspicious": "critical",
               "Stable": "good", "Warning": "warning", "PASS": "good", "WARN": "warning", "FAIL": "critical",
               "Healthy": "good", "MEDIUM": "warning", "HIGH": "serious", "CRITICAL": "critical"}
TIER_COLOR = {k: STATUS[v] for k, v in TIER_STATUS.items()}
ICON = {"good": "●", "warning": "▲", "serious": "◆", "critical": "■", "neutral": "○"}

pio.templates["risklens"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK2),
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    colorway=SERIES,
    margin=dict(l=8, r=16, t=44, b=36),
    title=dict(font=dict(size=14, color=INK, weight=600), x=0.01, xanchor="left", y=0.97),
    xaxis=dict(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="", zeroline=False,
               title=dict(font=dict(size=11, color=MUTED_INK))),
    yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor="rgba(0,0,0,0)", rangemode="tozero",
               title=dict(font=dict(size=11, color=MUTED_INK))),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", font=dict(size=11), title=None),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d9d8d2", font=dict(color=INK, size=12)),
    hovermode="closest", bargap=0.35,
))
pio.templates.default = "risklens"
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def load_css() -> None:
    css = (Path(__file__).resolve().parent.parent / "styles" / "theme.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------- formatting
def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_money(v, digits: int = 1) -> str:
    if _nan(v):
        return "—"
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.{digits}f}{suf}"
    return f"{v:,.0f}"


def fmt_pct(v, digits: int = 1) -> str:
    return "—" if _nan(v) else f"{v * 100:.{digits}f}%"


def fmt_int(v) -> str:
    return "—" if _nan(v) else f"{int(v):,}"


def fmt_num(v, digits: int = 2) -> str:
    return "—" if _nan(v) else f"{v:,.{digits}f}"


# ----------------------------------------------------------------- components
def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='rl-eyebrow'>{html.escape(eyebrow)}</div>"
                f"<div class='rl-title'>{html.escape(title)}</div>"
                + (f"<div class='rl-sub'>{subtitle}</div>" if subtitle else ""), unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='rl-section'>{html.escape(title)}</div>"
                + (f"<div class='rl-section-sub'>{sub}</div>" if sub else ""), unsafe_allow_html=True)


def tag(kind: str) -> str:
    return f"<span class='rl-tag rl-tag-{kind}'>{kind}</span>"


def kpi(label: str, value: str, foot: str = "", kind: str | None = None) -> None:
    st.markdown(
        f"<div class='rl-kpi'><div class='rl-kpi-label'>{html.escape(label)}</div>"
        f"<div class='rl-kpi-value'>{value}</div>"
        f"<div class='rl-kpi-foot'>{tag(kind) + ' ' if kind else ''}{foot}</div></div>",
        unsafe_allow_html=True)


def pill(label: str, status: str | None = None) -> str:
    s = status or TIER_STATUS.get(str(label), "neutral")
    return f"<span class='rl-pill rl-pill-{s}'>{ICON.get(s, '○')} {html.escape(str(label))}</span>"


def empty_state(msg: str) -> None:
    st.markdown(f"<div class='rl-card' style='color:#898781;font-size:13.5px'>{msg}</div>", unsafe_allow_html=True)


def chart(fig: go.Figure, height: int = 300) -> None:
    fig.update_layout(height=height, template="risklens", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, theme=None)


def api_error(e: Exception) -> None:
    st.error(f"{e}", icon="⚠️")
