from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from common.config import LLM_PROVIDER

router = APIRouter(prefix="/ai", tags=["ai"])


class AIQuery(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    filters: dict[str, list[str]] = {}
    session_id: str | None = None
    history: list[dict] = []
    user: str = "analyst"


class AIInvestigate(BaseModel):
    customer_id: str
    filters: dict[str, list[str]] = {}
    session_id: str | None = None
    user: str = "analyst"


@router.get("/status")
def status():
    from ai import llm
    from ai.tools import REGISTRY
    return {"llm_configured": llm.llm_configured(), "provider": LLM_PROVIDER if llm.llm_configured() else None,
            "model": (llm.LLM_MODEL or llm.DEFAULT_MODELS.get(LLM_PROVIDER)) if llm.llm_configured() else None,
            "mode": "llm" if llm.llm_configured() else "planner", "tools": sorted(REGISTRY)}


@router.get("/tools")
def tools():
    from ai.tools import REGISTRY
    return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in REGISTRY.values()]


@router.post("/query")
def query(q: AIQuery):
    from ai.agent import investigate
    return investigate(q.question, q.filters, q.session_id, q.history, q.user).to_dict()


@router.post("/investigate")
def investigate_customer(q: AIInvestigate):
    from ai.agent import investigate
    from backend.services.customer_service import get_profile
    cid = get_profile(q.customer_id)["identity"]["customer_id"]       # 404 if unknown
    return investigate(f"Why is customer {cid} high risk? Investigate fully and recommend analyst actions.",
                       q.filters, q.session_id, [], q.user).to_dict()
