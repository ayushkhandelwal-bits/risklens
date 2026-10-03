"""explain_prediction — SHAP explanation of the champion model's PD for a customer."""
from __future__ import annotations

from ai.tools.base import compact, tool
from ai.tools.customer_tool import CID
from backend.services import risk_service


@tool("explain_prediction",
      "Explain why the model assigned a customer their PD, using SHAP values from the champion XGBoost model: "
      "top risk-increasing drivers, top mitigating factors, contribution by data source, and the base (portfolio "
      "average) PD. Contributions are in log-odds and sum exactly to the model output.",
      {"type": "object", "properties": {"customer_id": CID,
                                        "top_n": {"type": "integer", "minimum": 1, "maximum": 10}},
       "required": ["customer_id"]},
      "Computing SHAP explanation...")
def explain_prediction(customer_id: str, top_n: int = 5) -> dict:
    e = risk_service.explanation(customer_id, int(top_n or 5))
    slim = lambda d: {k: d[k] for k in ("label", "display_value", "shap_value", "group")}
    return compact({"pd": e["pd"], "base_pd": e["base_pd"], "model": f"{e['model_name']} v{e['model_version']}",
                    "top_risk_increasing": [slim(d) for d in e["top_risk_increasing"]],
                    "top_risk_decreasing": [slim(d) for d in e["top_risk_decreasing"]],
                    "contribution_by_source": e["group_contributions"],
                    "reconstruction_error": e["reconstruction_error"]})
