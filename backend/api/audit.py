from __future__ import annotations

from fastapi import APIRouter, Query

from backend.services import audit_service

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/log")
def audit_log(limit: int = Query(100, ge=1, le=1000), action: str | None = None, session_id: str | None = None):
    return audit_service.recent(limit, action, session_id)
