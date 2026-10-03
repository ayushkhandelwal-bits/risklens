"""MCP server tests — protocol-level (in-memory) and a real stdio subprocess."""
import asyncio
import json
import sys

from tests.conftest import requires_db

pytestmark = requires_db


def _run(coro):
    return asyncio.run(coro)


def test_mcp_lists_all_tools_and_resources():
    from mcp.client.client import Client
    from mcp_server.server import mcp
    from ai.tools import REGISTRY

    async def go():
        async with Client(mcp) as c:
            return await c.list_tools(), await c.list_resources(), await c.list_prompts()
    tools, res, prompts = _run(go())
    assert {t.name for t in tools.tools} == set(REGISTRY)
    assert {str(r.uri) for r in res.resources} == {"risklens://methodology/assumptions", "risklens://data/dictionary"}
    assert [p.name for p in prompts.prompts] == ["investigate_customer"]


def test_mcp_tool_call_matches_direct_call():
    from mcp.client.client import Client
    from mcp_server.server import mcp
    from ai.tools.risk_tool import calculate_customer_risk

    async def go():
        async with Client(mcp) as c:
            r = await c.call_tool("calculate_customer_risk", {"customer_id": "C100002"})
            bad = await c.call_tool("run_safe_sql_query", {"sql": "DELETE FROM risk_scores"})
            return json.loads(r.content[0].text), json.loads(bad.content[0].text)
    via_mcp, unsafe = _run(go())
    assert abs(via_mcp["pd"] - calculate_customer_risk("C100002")["pd"]) < 1e-9
    assert "error" in unsafe and "SELECT" in unsafe["error"]


def test_mcp_stdio_subprocess():
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters
    from common.config import ROOT

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT))
        async with Client(params) as c:
            tools = await c.list_tools()
            r = await c.call_tool("get_portfolio_metrics", {})
            return len(tools.tools), json.loads(r.content[0].text)
    n, metrics = _run(go())
    assert n == 10 and metrics["summary"]["customers"] == 50000


def test_agent_can_route_tools_through_mcp():
    from ai.agent import ToolExecutor
    ex = ToolExecutor("pytest-mcp", "pytest", transport="mcp")
    r = ex("get_early_warning_signals", customer_id="C100002")
    assert r["customer_id"] == "C100002" and ex.trace[0].status == "ok"
