"""
RiskLens API — FastAPI backend.

    uvicorn backend.main:app --reload --port 8000

The Streamlit frontend talks only to this API, so the UI can later be replaced
(React / Next.js) without touching business logic.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas.filters import InvalidFilter
from backend.services.customer_service import CustomerNotFound
from common.config import ENVIRONMENT
from common.db import DatabaseUnavailable, ping

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="RiskLens API", version="1.0",
              description="AI-powered lending risk intelligence platform — portfolio, risk, early warning, "
                          "investigation, model monitoring and AI analyst services.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- error handling
@app.exception_handler(DatabaseUnavailable)
async def _db_down(_: Request, exc: DatabaseUnavailable):
    return JSONResponse(status_code=503, content={"error": "database_unavailable",
                                                  "message": "The risk database is unavailable. Check that PostgreSQL is running and DATABASE_URL is correct."})


@app.exception_handler(CustomerNotFound)
async def _not_found(_: Request, exc: CustomerNotFound):
    return JSONResponse(status_code=404, content={"error": "customer_not_found", "message": str(exc)})


@app.exception_handler(InvalidFilter)
async def _bad_filter(_: Request, exc: InvalidFilter):
    return JSONResponse(status_code=422, content={"error": "invalid_filter", "message": str(exc)})


@app.exception_handler(ValueError)
async def _bad_value(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": "bad_request", "message": str(exc)})


@app.exception_handler(FileNotFoundError)
async def _missing_artifact(_: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"error": "model_unavailable",
                                                  "message": f"Model artifact unavailable: {exc}. Run `python -m ml.train` first."})


# ---------------------------------------------------------------- routes
from backend.api import customers, portfolio  # noqa: E402

app.include_router(portfolio.router)
app.include_router(customers.router)

import importlib  # noqa: E402
import importlib.util  # noqa: E402

for _mod in ("backend.api.risk", "backend.api.ews", "backend.api.model", "backend.api.ai", "backend.api.audit"):
    if importlib.util.find_spec(_mod) is not None:   # routers added phase by phase; real import errors still raise
        app.include_router(importlib.import_module(_mod).router)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok" if ping() else "degraded", "database": ping(), "environment": ENVIRONMENT}
