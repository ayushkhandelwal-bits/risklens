"""Global portfolio filters — stored in session_state so every page (and the AI
Risk Analyst) works on the same selection."""
from __future__ import annotations

import streamlit as st

from frontend.utils import api

FILTER_LABELS = {
    "vintage": "Vintage",
    "product": "Product",
    "income_band": "Income band",
    "region": "Region",
    "customer_segment": "Customer segment",
    "risk_tier": "Risk tier",
    "credit_score_band": "Credit score band",
}


def current() -> dict:
    return dict(st.session_state.get("rl_filters", {}))


def as_params(extra: dict | None = None) -> dict:
    p = {k: v for k, v in current().items() if v}
    p.setdefault("population", "portfolio")
    if extra:
        p.update(extra)
    return p


def describe() -> str:
    f = {k: v for k, v in current().items() if v}
    if not f:
        return "Entire booked portfolio"
    return " · ".join(f"{FILTER_LABELS.get(k, k)}: {', '.join(v)}" for k, v in f.items())


def filter_bar(keys: list[str] | None = None) -> dict:
    """Render the filter row. Returns the active filters."""
    keys = keys or list(FILTER_LABELS)
    options = api.get("/portfolio/filters")
    options["vintage"] = [v for v in options.get("vintage", []) if v != "2026-Q1"]  # intake is not booked
    state = st.session_state.setdefault("rl_filters", {})
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        with col:
            opts = options.get(k, [])
            default = [v for v in state.get(k, []) if v in opts]
            state[k] = st.multiselect(FILTER_LABELS[k], opts, default=default, key=f"flt_{k}",
                                      placeholder="All", disabled=not opts)
    active = {k: v for k, v in state.items() if v}
    c1, c2 = st.columns([6, 1])
    with c1:
        st.caption(f"Showing: **{describe()}**")
    with c2:
        if active and st.button("Clear filters", use_container_width=True):
            st.session_state["rl_filters"] = {}
            for k in keys:
                st.session_state.pop(f"flt_{k}", None)
            st.rerun()
    return active
