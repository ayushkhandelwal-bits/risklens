"""Thin client for the RiskLens API.

Caching: responses are cached for a short TTL keyed on (path, full parameter
set). Because every filter value is part of the key, changing a filter can
never return a stale result."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


class APIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _params_key(params: dict | None) -> tuple:
    if not params:
        return ()
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, (list, tuple)):
            items.append((k, tuple(v)))
        elif v is not None:
            items.append((k, v))
    return tuple(items)


def _request(method: str, path: str, params: dict | None = None, json: dict | None = None, timeout: int = 60):
    try:
        r = requests.request(method, f"{API_URL}{path}", params=params, json=json, timeout=timeout)
    except requests.ConnectionError:
        raise APIError(f"The RiskLens API is not reachable at {API_URL}. Start it with "
                       "`uvicorn backend.main:app --port 8000`.")
    except requests.Timeout:
        raise APIError("The RiskLens API took too long to respond.")
    if r.status_code >= 400:
        try:
            msg = r.json().get("message") or r.json().get("detail") or r.text
        except Exception:
            msg = r.text
        raise APIError(str(msg), r.status_code)
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_get(path: str, key: tuple):
    params = {k: list(v) if isinstance(v, tuple) else v for k, v in key}
    return _request("GET", path, params=params)


def get(path: str, params: dict | None = None, cache: bool = True):
    if cache:
        return _cached_get(path, _params_key(params))
    return _request("GET", path, params=params)


def post(path: str, payload: dict, timeout: int = 180):
    return _request("POST", path, json=payload, timeout=timeout)
