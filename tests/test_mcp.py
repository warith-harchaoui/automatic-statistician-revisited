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
