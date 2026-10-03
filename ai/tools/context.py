"""Per-request analyst context (the dashboard filter selection) shared with tools."""
from __future__ import annotations

from contextvars import ContextVar

from backend.schemas.filters import FILTER_COLUMNS, PortfolioFilters

_current: ContextVar[dict] = ContextVar("risklens_filters", default={})


def set_filters(filters: dict | None) -> None:
    _current.set({k: v for k, v in (filters or {}).items() if k in FILTER_COLUMNS and v})


def resolve(filters: dict | None = None) -> PortfolioFilters:
    """Explicit tool filters override the dashboard context; booked portfolio by default."""
    base = dict(_current.get())
    if filters:
        base.update({k: v for k, v in filters.items() if k in FILTER_COLUMNS})
    return PortfolioFilters(population="portfolio", **{k: v if isinstance(v, list) else [v] for k, v in base.items() if v})
