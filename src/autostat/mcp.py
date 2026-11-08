"""MCP (Model Context Protocol) server for autostat, through FastAPI-MCP.

What this module does
    It mounts an MCP endpoint at ``/mcp`` on the HTTP application of
    :mod:`autostat.api`, so an agent host calls the same routes as tools,
    each named by its ``operation_id``. ``autostat-mcp`` serves both.

Which routes become tools
    An allowlist (:data:`TOOLS`), not every route: ``list_series``,
    ``describe_published``, ``analyse_series`` and ``get_report``. Left out:
    ``health``, which is for monitors; ``get_report_file``, which
    returns binary files an agent reads better as ``get_report``'s text.

Why the long timeout
    FastAPI-MCP calls the application through an HTTP client whose default
    timeout is seconds; a search takes minutes. The client here waits up to
    an hour, the bound the API's schemas already impose on one request.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import click
import httpx
from fastapi_mcp import FastApiMCP

from . import __version__
from .api import app

#: The routes published as MCP tools, by operation id.
TOOLS = ["list_series", "describe_published", "analyse_series", "get_report"]

mcp = FastApiMCP(
    app,
    name="autostat",
    description=(
        "The Automatic Statistician, revisited. list_series names the 2014 "
        "benchmark series; describe_published writes the English report of a "
        "model the 2014 system published; analyse_series searches kernel "
        "structures for values you send and reports on the best one; "
        "get_report reads a report's Markdown or LaTeX source."
    ),
    include_operations=TOOLS,
    http_client=httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://autostat",
        timeout=3600.0,
    ),
)
mcp.mount_http()


@click.command(
    name="autostat-mcp",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Serve the autostat HTTP API with its MCP endpoint at /mcp. The host "
        "defaults to 127.0.0.1: pass --host 0.0.0.0 to accept connections "
        "from other machines."
    ),
)
@click.version_option(__version__, "--version", prog_name="autostat-mcp")
@click.option(
    "--host",
    envvar="AUTOSTAT_HOST",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind.",
)
@click.option(
    "--port",
    envvar="AUTOSTAT_PORT",
    type=int,
    default=8000,
    show_default=True,
    help="Port to bind.",
)
def command(host: str, port: int) -> None:
    """Serve the application with uvicorn, in one worker.

    Parameters
    ----------
    host : str
        Interface to bind.
    port : int
        Port to bind.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port, workers=1)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``autostat-mcp`` console script.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 once the server stops, 2 on invalid arguments.
    """
    try:
        command.main(
            args=list(sys.argv[1:] if argv is None else argv),
            prog_name="autostat-mcp",
            standalone_mode=False,
        )
    except click.UsageError as error:
        error.show()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
