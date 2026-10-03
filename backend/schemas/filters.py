"""Portfolio filter contract shared by every analytical endpoint."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# API field name -> v_portfolio column. Only these columns can ever be filtered on.
FILTER_COLUMNS = {
    "vintage": "vintage",
    "product": "product",
    "income_band": "income_band",
    "region": "region",
    "customer_segment": "customer_segment",
    "risk_tier": "risk_tier",
    "credit_score_band": "credit_score_band",
    "ews_band": "ews_band",
    "anomaly_status": "anomaly_status",
}
POPULATIONS = ("portfolio", "intake", "all")


class PortfolioFilters(BaseModel):
    population: str = Field("portfolio", description="portfolio | intake | all")
    vintage: list[str] = []
    product: list[str] = []
    income_band: list[str] = []
    region: list[str] = []
    customer_segment: list[str] = []
    risk_tier: list[str] = []
    credit_score_band: list[str] = []
    ews_band: list[str] = []
    anomaly_status: list[str] = []

    def active(self) -> dict[str, list[str]]:
        return {k: getattr(self, k) for k in FILTER_COLUMNS if getattr(self, k)}

    def describe(self) -> str:
        parts = [f"{k.replace('_', ' ')} = {', '.join(v)}" for k, v in self.active().items()]
        return "; ".join(parts) if parts else "entire portfolio"


class InvalidFilter(ValueError):
    pass


def build_where(f: PortfolioFilters, allowed_values: dict[str, set[str]] | None = None,
                alias: str = "") -> tuple[str, dict[str, Any]]:
    """Return a parameterised WHERE clause. Values are validated against the
    known domain when `allowed_values` is supplied (unknown value -> InvalidFilter)."""
    if f.population not in POPULATIONS:
        raise InvalidFilter(f"Unknown population '{f.population}'. Use one of {POPULATIONS}.")
    p = f"{alias}." if alias else ""
    clauses, params = [], {}
    if f.population != "all":
        clauses.append(f"{p}population = :population")
        params["population"] = f.population
    for key, values in f.active().items():
        if allowed_values is not None and key in allowed_values:
            bad = [v for v in values if v not in allowed_values[key]]
            if bad:
                raise InvalidFilter(f"Invalid value(s) for {key}: {bad}")
        names = []
        for i, v in enumerate(values):
            name = f"{key}_{i}"
            params[name] = v
            names.append(f":{name}")
        clauses.append(f"{p}{FILTER_COLUMNS[key]} IN ({', '.join(names)})")
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params
