"""The MCP server publishes exactly the allowlisted routes, described.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from autostat.mcp import TOOLS, mcp


def test_the_tool_inventory_is_the_allowlist():
    """Health checks and binary downloads stay out of the agent's tools."""
    assert sorted(tool.name for tool in mcp.tools) == sorted(TOOLS)
    assert {"health", "get_report_file"}.isdisjoint(tool.name for tool in mcp.tools)


def test_every_tool_says_what_it_does():
    """Each tool carries a description an agent can choose by."""
    for tool in mcp.tools:
        assert tool.description and len(tool.description) > 40, tool.name


def test_the_analysis_tool_states_its_bounds():
    """analyse_series exposes the limits the API enforces."""
    (tool,) = [t for t in mcp.tools if t.name == "analyse_series"]
    schema = str(tool.inputSchema)
    assert "max_depth" in schema and "values" in schema


def test_a_tool_answers_through_the_protocol():
    """Handshake, tool list and call over the real MCP endpoint."""
    from fastapi.testclient import TestClient

    from autostat.mcp import app

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    hello = {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    }
    with TestClient(app) as client:

        def call(method, params, ident=1):
            """Send one JSON-RPC request.

            Parameters
            ----------
            method : str
                The method.
            params : dict
                Its parameters.
            ident : int, optional
                The request id.

            Returns
            -------
            httpx.Response
                The answer.
            """
            payload = {
                "jsonrpc": "2.0",
                "id": ident,
                "method": method,
                "params": params,
            }
            return client.post("/mcp", headers=headers, json=payload)

        started = call("initialize", hello)
        headers["mcp-session-id"] = started.headers["mcp-session-id"]
        client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        tools = call("tools/list", {}, 2).json()["result"]["tools"]
        done = call("tools/call", {"name": "list_series", "arguments": {}}, 3).json()
        missing = call(
            "tools/call", {"name": "get_report", "arguments": {"report_id": "x"}}, 4
        )
    assert sorted(t["name"] for t in tools) == sorted(TOOLS)
    assert "01-airline" in done["result"]["content"][0]["text"]
    assert missing.json()["result"]["isError"] is True
