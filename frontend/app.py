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
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/ayushjhanginiya"
AUTHOR_EMAIL = "ayushkhandelwal.connect@gmail.com"


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
    st.markdown(f"""<div class="rl-credit">
  <div class="rl-credit-name">Built by <b>Ayush Khandelwal</b></div>
  <div class="rl-credit-role">MBA, Business Analytics · BITS Pilani</div>
  <div class="rl-credit-links">
    <a href="{AUTHOR_LINKEDIN}" target="_blank" rel="noopener noreferrer" title="LinkedIn" aria-label="LinkedIn profile">
      <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z"/></svg>
    </a>
    <a href="mailto:{AUTHOR_EMAIL}" title="{AUTHOR_EMAIL}" aria-label="Email">
      <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" d="M3 5.5h18v13H3z M3.5 6l8.5 7 8.5-7"/></svg>
    </a>
  </div>
</div>""", unsafe_allow_html=True)

nav.run()
