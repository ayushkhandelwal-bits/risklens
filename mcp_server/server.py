"""
RiskLens MCP Server — exposes the AI analyst's controlled tools over the Model
Context Protocol, so any MCP-capable agent (Claude Desktop, Claude Code, an IDE
agent, or RiskLens's own analyst with RISKLENS_TOOL_TRANSPORT=mcp) can
investigate RiskLens data through the SAME governed interface:

  * identical tool implementations as the in-app agent (ai/tools)
  * read-only SQL guard, least-privilege DB role
  * every call written to the audit log (session 'mcp')

Run (stdio):            python -m mcp_server.server
Run (streamable HTTP):  python -m mcp_server.server --http --port 8765

Claude Desktop config (claude_desktop_config.json):
  {"mcpServers": {"risklens": {"command": "python", "args": ["-m", "mcp_server.server"],
                               "cwd": "D:/PROJECTS_GIT/credit-default-risk"}}}
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from mcp.server.mcpserver import MCPServer  # noqa: E402

from ai.tools import REGISTRY  # noqa: E402
from backend.services.audit_service import log_event  # noqa: E402

INSTRUCTIONS = (
    "RiskLens lending-risk intelligence tools. All data is a simulated digital-lending book built on the "
    "Home Credit dataset. Default rate is observed; PD/Expected Loss are model outputs; LGD/EAD are documented "
    "assumptions; anomaly signals are NOT fraud determinations. Use the tools for every number you report. "
    "Outputs are decision support — never approve or reject credit."
)


def _wrap(tool):
    """Audit every MCP invocation, return JSON-safe results, never raise into the protocol layer."""
    @functools.wraps(tool.func)
    def handler(*args, **kwargs):
        try:
            result, status = tool.func(*args, **kwargs), "ok"
        except Exception as exc:
            result, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
        log_event("tool_call", tool=f"mcp:{tool.name}", customer_id=kwargs.get("customer_id"), parameters=kwargs,
                  result_summary=json.dumps(result, default=str)[:300], session_id="mcp", user="mcp-client",
                  status=status)
        return json.loads(json.dumps(result, default=str))
    return handler


def build_server() -> MCPServer:
    server = MCPServer(name="risklens", title="RiskLens Risk Intelligence", version="1.0", instructions=INSTRUCTIONS)
    for tool in REGISTRY.values():
        server.add_tool(_wrap(tool), name=tool.name, description=tool.description)

    @server.resource("risklens://methodology/assumptions", name="assumptions",
                     description="Expected-loss assumptions, risk tiers and score definitions",
                     mime_type="application/json")
    def assumptions() -> str:
        from backend.services.risk_service import assumptions as a
        return json.dumps(a())

    @server.resource("risklens://data/dictionary", name="data_dictionary",
                     description="Model features with business labels and source systems",
                     mime_type="application/json")
    def dictionary() -> str:
        from ml.feature_engineering import FEATURES
        return json.dumps({k: {"label": v[0], "source": v[1], "unit": v[2]} for k, v in FEATURES.items()})

    @server.prompt(name="investigate_customer", description="Full credit-risk investigation of one customer")
    def investigate_customer(customer_id: str) -> str:
        return (f"Investigate customer {customer_id}: use get_customer_profile, calculate_customer_risk, "
                f"explain_prediction, get_early_warning_signals and get_anomaly_signals. Summarise the risk, the "
                f"SHAP drivers, the evidence and recommend analyst actions (decision support only).")
    return server


mcp = build_server()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.http:
        mcp.run("streamable-http", port=a.port)
    else:
        mcp.run("stdio")
