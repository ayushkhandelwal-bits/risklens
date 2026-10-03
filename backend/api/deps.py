from __future__ import annotations

from fastapi import Query

from backend.schemas.filters import PortfolioFilters


def portfolio_filters(
    population: str = Query("portfolio", description="portfolio | intake | all"),
    vintage: list[str] = Query([]),
    product: list[str] = Query([]),
    income_band: list[str] = Query([]),
    region: list[str] = Query([]),
    customer_segment: list[str] = Query([]),
    risk_tier: list[str] = Query([]),
    credit_score_band: list[str] = Query([]),
    ews_band: list[str] = Query([]),
    anomaly_status: list[str] = Query([]),
) -> PortfolioFilters:
    return PortfolioFilters(population=population, vintage=vintage, product=product, income_band=income_band,
                            region=region, customer_segment=customer_segment, risk_tier=risk_tier,
                            credit_score_band=credit_score_band, ews_band=ews_band, anomaly_status=anomaly_status)
