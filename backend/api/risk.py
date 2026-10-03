from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.deps import portfolio_filters
from backend.schemas.filters import PortfolioFilters
from backend.services import risk_service

router = APIRouter(tags=["risk"])


@router.get("/customers/{customer_id}/risk")
def customer_risk(customer_id: str):
    return risk_service.customer_risk(customer_id)


@router.get("/customers/{customer_id}/explanation")
def customer_explanation(customer_id: str, top_n: int = Query(5, ge=1, le=15)):
    return risk_service.explanation(customer_id, top_n)


@router.get("/risk/assumptions")
def assumptions():
    return risk_service.assumptions()


@router.get("/risk/expected-loss/{dimension}")
def expected_loss(dimension: str, f: PortfolioFilters = Depends(portfolio_filters)):
    return risk_service.expected_loss_by(dimension, f)


@router.get("/risk/drivers")
def drivers(f: PortfolioFilters = Depends(portfolio_filters)):
    return risk_service.driver_analysis(f)
