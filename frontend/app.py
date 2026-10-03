"""
RiskLens — Streamlit frontend (presentation layer only).

    streamlit run streamlit_app.py

All data comes from the FastAPI backend (API_URL). Pages live in frontend/pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from frontend.components import ui  # noqa: E402
from frontend.utils import api  # noqa: E402

st.set_page_config(page_title="RiskLens", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
ui.load_css()

PAGES = ROOT / "frontend" / "pages"


def _page(file: str, title: str, icon: str, default: bool = False):
    path = PAGES / file
    return st.Page(str(path), title=title, icon=icon, default=default) if path.exists() else None


pages = {
    "Monitor": [
        _page("overview.py", "Overview", ":material/monitoring:", default=True),
        _page("portfolio.py", "Portfolio", ":material/account_balance:"),
        _page("credit_risk.py", "Credit Risk", ":material/query_stats:"),
        _page("early_warnings.py", "Early Warnings", ":material/notifications_active:"),
    ],
    "Investigate": [
        _page("investigations.py", "Investigations", ":material/person_search:"),
        _page("ai_analyst.py", "AI Risk Analyst", ":material/smart_toy:"),
    ],
    "Govern": [
        _page("model_monitor.py", "Model Monitor", ":material/model_training:"),
        _page("data_quality.py", "Data & Audit", ":material/fact_check:"),
    ],
}
# Only register pages that exist (built phase by phase).
pages = {g: [p for p in ps if p is not None] for g, ps in pages.items()}
pages = {g: ps for g, ps in pages.items() if ps}

st.logo(str(ROOT / "frontend" / "styles" / "logo.svg"), size="large")

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    try:
        h = api.get("/health", cache=False)
        ok = h.get("database")
        st.markdown(f"<div style='font-size:12px;color:#8b95a1;margin-top:18px'>"
                    f"{'🟢' if ok else '🔴'} API {'connected' if ok else 'degraded — database offline'}</div>",
                    unsafe_allow_html=True)
    except api.APIError:
        st.markdown("<div style='font-size:12px;color:#e8a0a0;margin-top:18px'>🔴 API offline</div>",
                    unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#6b7580;margin-top:6px'>Lending Risk Intelligence · internal tool<br>"
                "Decision support only — analysts make final decisions.</div>",
                unsafe_allow_html=True)

nav.run()
