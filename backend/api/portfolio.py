from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.deps import portfolio_filters
from backend.schemas.filters import PortfolioFilters
from backend.services import portfolio_service as svc

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/filters")
def filters():
    """Valid values for every filter dimension."""
    return svc.filter_options()


@router.get("/summary")
def summary(f: PortfolioFilters = Depends(portfolio_filters)):
    return svc.summary(f)


@router.get("/trend")
def trend(f: PortfolioFilters = Depends(portfolio_filters)):
    return svc.trend(f)


@router.get("/risk-distribution")
def risk_distribution(f: PortfolioFilters = Depends(portfolio_filters)):
    return svc.risk_distribution(f)


@router.get("/breakdown/{dimension}")
def breakdown(dimension: str, f: PortfolioFilters = Depends(portfolio_filters)):
    return svc.breakdown(dimension, f)


@router.get("/customers")
def customers(f: PortfolioFilters = Depends(portfolio_filters), sort: str = "expected_loss",
              limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), search: str | None = None):
    return svc.customers(f, sort, limit, offset, search)


@router.get("/behaviour-trend")
def behaviour_trend(f: PortfolioFilters = Depends(portfolio_filters)):
    return svc.behaviour_trend(f)


@router.get("/top-drivers")
def top_drivers(f: PortfolioFilters = Depends(portfolio_filters), limit: int = Query(8, ge=1, le=30)):
    return svc.top_drivers(f, limit)


@router.get("/data-quality")
def data_quality():
    return svc.data_quality()


@router.get("/insight")
def insight(f: PortfolioFilters = Depends(portfolio_filters)):
    from backend.services.insight_service import portfolio_insight
    return portfolio_insight(f)


@router.get("/concentration")
def concentration(f: PortfolioFilters = Depends(portfolio_filters)):
    return svc.concentration(f)
