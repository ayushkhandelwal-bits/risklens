from __future__ import annotations

from fastapi import APIRouter, Query

from backend.services import monitoring_service, risk_service

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics")
def metrics():
    """Champion / challenger metrics on the held-out test split, with ROC and calibration curves."""
    return risk_service.model_registry()


@router.get("/importance")
def importance(population: str = "portfolio", limit: int = Query(20, ge=1, le=60)):
    return risk_service.feature_importance(population, limit)


@router.get("/drift")
def drift(top_n: int = Query(15, ge=1, le=60), include_bins: bool = False):
    """Latest monitoring run: overall status, score PSI, feature PSI, segment mix drift, interpretation."""
    return monitoring_service.drift_report(top_n, include_bins)


@router.get("/drift/features")
def drift_features():
    return monitoring_service.feature_psi_all()


@router.get("/performance/segments")
def performance_segments():
    return monitoring_service.performance_history()


@router.post("/monitoring/run")
def run_monitoring():
    """Re-compute monitoring now (PSI, drift, performance stability)."""
    from ml.monitoring import main
    return {"run_id": main()}
