from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.deps import portfolio_filters
from backend.schemas.filters import PortfolioFilters
from backend.services import ews_service

router = APIRouter(tags=["early-warning"])


@router.get("/customers/{customer_id}/warnings")
def customer_warnings(customer_id: str):
    return ews_service.customer_warnings(customer_id)


@router.get("/customers/{customer_id}/anomaly")
def customer_anomaly(customer_id: str):
    return ews_service.customer_anomaly(customer_id)


@router.get("/early-warnings/summary")
def summary(f: PortfolioFilters = Depends(portfolio_filters)):
    return ews_service.summary(f)


@router.get("/early-warnings/alerts")
def alerts(f: PortfolioFilters = Depends(portfolio_filters), band: list[str] = Query([]),
           trigger: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    return ews_service.alerts(f, band or None, trigger, limit)


@router.get("/early-warnings/validation")
def validation():
    return ews_service.validation()


@router.get("/anomalies/summary")
def anomalies(f: PortfolioFilters = Depends(portfolio_filters)):
    return ews_service.anomaly_summary(f)


@router.get("/investigations/queue")
def queue(f: PortfolioFilters = Depends(portfolio_filters), limit: int = Query(50, ge=1, le=500)):
    return {"method": ews_service.priority_method(), "rows": ews_service.investigation_queue(f, limit)}
