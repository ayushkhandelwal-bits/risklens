"""analyze_model_drift — model performance and population/feature drift (PSI)."""
from __future__ import annotations

from ai.tools.base import compact, tool


@tool("analyze_model_drift",
      "Model monitoring for the champion PD model: current performance (AUC, Gini, KS, Brier, calibration) on the "
      "held-out test split, feature-level Population Stability Index (PSI) between the development sample and the "
      "recent application intake, score PSI, segment mix drift, and Stable/Warning/Critical status with thresholds. "
      "Use for questions about model health, drift, or whether the model can be trusted on new applicants.",
      {"type": "object", "properties": {"top_n": {"type": "integer", "minimum": 1, "maximum": 30}}},
      "Analysing model drift...")
def analyze_model_drift(top_n: int = 10) -> dict:
    from backend.services import monitoring_service
    return compact(monitoring_service.drift_report(int(top_n or 10)))
