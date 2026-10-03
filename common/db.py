"""Database access helpers (SQLAlchemy engine + safe parameterised reads)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from common.config import DATABASE_URL


class DatabaseUnavailable(RuntimeError):
    """Raised when PostgreSQL cannot be reached."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)


def read_sql(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a parameterised SELECT and return a DataFrame."""
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except OperationalError as exc:  # connection refused, auth, etc.
        raise DatabaseUnavailable(f"Database unavailable: {exc.orig}") from exc


def execute(sql: str, params: dict[str, Any] | None = None) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(text(sql), params or {})
    except OperationalError as exc:
        raise DatabaseUnavailable(f"Database unavailable: {exc.orig}") from exc


def run_sql_file(path, conn=None) -> None:
    """Execute a .sql file (multiple statements) using the raw DBAPI connection."""
    sql = open(path, encoding="utf-8").read()
    raw = get_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(sql)
        raw.commit()
    finally:
        raw.close()


def ping() -> bool:
    try:
        read_sql("SELECT 1 AS ok")
        return True
    except Exception:
        return False
