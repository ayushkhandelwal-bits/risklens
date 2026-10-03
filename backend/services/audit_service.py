"""Audit trail — every AI query, tool call, investigation and scoring event."""
from __future__ import annotations

import json
import logging
from typing import Any

from common.db import execute, read_sql

log = logging.getLogger("risklens.audit")


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def log_event(action: str, *, tool: str | None = None, customer_id: str | None = None,
              parameters: dict | None = None, result_summary: str | None = None,
              user: str = "analyst", session_id: str | None = None, status: str = "ok") -> None:
    """Never raise: auditing must not break the user flow, but failures are logged."""
    try:
        execute(
            """INSERT INTO audit_log (user_name, session_id, action, tool, customer_id, parameters, result_summary, status)
               VALUES (:u, :s, :a, :t, :c, CAST(:p AS JSONB), :r, :st)""",
            {"u": user, "s": session_id, "a": action, "t": tool, "c": customer_id,
             "p": json.dumps(_jsonable(parameters or {})), "r": (result_summary or "")[:4000], "st": status},
        )
    except Exception as exc:  # pragma: no cover
        log.warning("audit write failed: %s", exc)


def recent(limit: int = 100, action: str | None = None, session_id: str | None = None) -> list[dict]:
    clauses, params = [], {"l": max(1, min(limit, 1000))}
    if action:
        clauses.append("action = :a")
        params["a"] = action
    if session_id:
        clauses.append("session_id = :s")
        params["s"] = session_id
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    df = read_sql(f"""SELECT id, ts, user_name, session_id, action, tool, customer_id, parameters, result_summary, status
                      FROM audit_log {where} ORDER BY ts DESC, id DESC LIMIT :l""", params)
    df["ts"] = df.ts.astype(str)
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")
