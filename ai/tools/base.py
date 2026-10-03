"""Tool registry: every AI tool is a plain Python function with a JSON schema.
The same registry backs the LLM agent (function calling) and the MCP server."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON schema (object)
    func: Callable[..., Any]
    progress: str              # short status line shown in the UI ("Retrieving portfolio metrics...")


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict, progress: str):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, parameters, fn, progress)
        return fn
    return deco


FILTER_SCHEMA = {
    "type": "object",
    "description": "Optional portfolio filters. Omit to use the analyst's current dashboard selection.",
    "properties": {k: {"type": "array", "items": {"type": "string"}} for k in
                   ("vintage", "product", "income_band", "region", "customer_segment", "risk_tier",
                    "credit_score_band", "ews_band", "anomaly_status")},
}


def compact(obj: Any, digits: int = 4) -> Any:
    """Round floats and drop NaN so tool results are small and JSON-safe."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: compact(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [compact(v, digits) for v in obj]
    return obj
