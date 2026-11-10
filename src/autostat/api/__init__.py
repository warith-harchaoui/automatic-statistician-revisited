"""HTTP API for autostat, built with FastAPI.

What this package does
    It serves the shared application services (:mod:`autostat.service`) over
    HTTP: list the benchmark series, describe a published 2014 model,
    analyse a series sent as JSON, then read back a report. The MCP server
    (:mod:`autostat.mcp`) wraps the same application.

How to run it
    ``uvicorn autostat.api:app`` serves the HTTP routes alone;
    ``autostat-mcp`` serves them with the MCP endpoint mounted beside them.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from .app import create_app

app = create_app()

__all__ = ["app", "create_app"]
