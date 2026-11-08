"""Local GUI for autostat: the command line as a web form that runs.

What this module does
    It generates the form page from the Click command with
    ``sprezzature-cli-gui`` (one field per option, defaults filled in, a
    "Build command" button), adds a run panel (``static/run.js``) then
    serves both from a small FastAPI host on 127.0.0.1. "Run on this
    machine" sends the form's fields as JSON to ``POST /run``, which checks
    them against the command's own parameter types and calls
    :func:`autostat.service.analyse_file`, as the command line does.

Why a local host
    The form names a file on this machine and a report directory to write;
    that is safe only for the person sitting at it. The host binds to the
    loopback interface, accepts parameter names from an allowlist (the
    command's options), never a shell string. It serves back only the
    files of the reports it wrote.

What it produces
    ``autostat-gui`` opens the page in a browser. Everything it loads comes
    from the host: sprezzature-cli-gui compiles Tailwind into the page. Strings of the run panel
    come from ``locales/i18n.yaml``; the form's labels are the command's
    help texts.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import re
import sys
import threading
import uuid
import webbrowser
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import click
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import __version__
from .cli import command as cli_command
from .service import analyse_file

#: Click's "no default" marker (8.2 and later), or a value nothing equals.
_UNSET = getattr(click.core, "UNSET", object())

#: Options of the command line the run panel never sends.
_NOT_RUN = {"version", "as_json", "help"}

#: Fields removed from the generated form: a form has no use for printing
#: the version or for switching standard output to JSON.
HIDDEN_FIELDS = ("version", "as_json")

_FOOTER = re.compile(r"<footer\b.*?</footer>", re.S)


def strings(lang: str) -> dict[str, str]:
    """Read the run panel's strings in one language, English as fallback.

    Parameters
    ----------
    lang : str
        Two-letter language code, e.g. ``"fr"``.

    Returns
    -------
    dict of str to str
        Text by key.
    """
    source = resources.files("autostat").joinpath("locales/i18n.yaml").read_text()
    table = yaml.safe_load(source)
    return {key: by_lang.get(lang, by_lang["en"]) for key, by_lang in table.items()}


def _replace_once(pattern: re.Pattern[str], new: str, html: str) -> str:
    """Replace the one match of a pattern, or fail loudly.

    The page comes from sprezzature-cli-gui; when its markup changes, this
    says so here instead of leaving the CDN or a dead field in the page.

    Parameters
    ----------
    pattern : re.Pattern
        What to find, exactly once.
    new : str
        Its replacement.
    html : str
        The document.

    Returns
    -------
    str
        The document, changed.

    Raises
    ------
    RuntimeError
        When the pattern matches zero or several times.
    """
    out, count = pattern.subn(lambda _: new, html)
    if count != 1:
        raise RuntimeError(f"expected one match of {pattern.pattern!r}, found {count}")
    return out


def page() -> str:
    """Generate the form page from the command line and adapt it.

    sprezzature-cli-gui compiles the current Tailwind release into the page,
    cut to the utilities it uses, so the page loads nothing from another
    host. The run panel's script is handed to that compilation too: the
    classes it adds at run time are styled. The version and JSON fields go,
    the generator's footer gives way to one about this host; the run
    panel's script is added.

    Returns
    -------
    str
        A complete HTML document.
    """
    from sprezzature_cli_gui import render_html, walk

    script = resources.files("autostat").joinpath("static/run.js").read_text()
    html = render_html(
        walk(cli_command, prog="autostat"), title="autostat", extra_content=[script]
    )
    for name in HIDDEN_FIELDS:
        field = re.compile(
            rf'<div class="mb-4"><label for="{name}"(?:(?!</div>).)*</div>\n?', re.S
        )
        html = _replace_once(field, "", html)
    html = _replace_once(
        _FOOTER,
        '<footer class="mt-8 text-[12px] text-label-secondary '
        'dark:text-label-secondary-dark" data-i18n="gui.footer"></footer>',
        html,
    )
    return html.replace(
        "</body>", '<script type="module" src="/run.js"></script>\n</body>', 1
    )


def run_options(fields: dict[str, Any]) -> dict[str, Any]:
    """Check the form's fields against the command and map them to the service.

    Each value goes through its Click parameter's own type, so the GUI
    accepts exactly what the command line accepts.

    Parameters
    ----------
    fields : dict
        Values by parameter name, as ``run.js`` sends them.

    Returns
    -------
    dict
        Keyword arguments for :func:`autostat.service.analyse_file`.

    Raises
    ------
    ValueError
        On an unknown field, a value of the wrong type or a missing path.
    """
    params = {p.name: p for p in cli_command.params if p.name not in _NOT_RUN}
    unknown = set(fields) - params.keys() - _NOT_RUN
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}")
    context = click.Context(cli_command)
    values: dict[str, Any] = {}
    for name, param in params.items():
        raw = fields.get(name)
        if raw is None or raw == "":
            default = param.get_default(context)
            # Click 8.2 says "no default" with a sentinel rather than None.
            values[name] = None if default is _UNSET else default
            continue
        try:
            values[name] = param.type.convert(raw, param, context)
        except click.BadParameter as error:
            raise ValueError(f"{name}: {error.message}") from None
    if values.get("path") is None:
        raise ValueError("path: a CSV or Parquet file is required")
    return {
        "path": Path(values["path"]),
        "time": values["time"],
        "value": values["value"],
        "name": values["name"],
        "out_dir": values["out"],
        "preset": values["preset"],
        "max_depth": values["max_depth"],
        "max_points": values["max_points"],
        "n_jobs": values["n_jobs"],
        "seed": values["seed"],
        "patience": values["patience"],
        "symbolic": values["symbolic"],
        "check": not values["no_check"],
        "report_format": values["report_format"],
        "pdf": values["pdf"],
    }


def create_gui_app() -> FastAPI:
    """Build the local host: the page, its script, its strings and the runs.

    Returns
    -------
    fastapi.FastAPI
        The application, meant for 127.0.0.1 only.
    """
    host = FastAPI(title="autostat GUI", version=__version__, docs_url=None)
    runs: dict[str, tuple[Path, list[str]]] = {}
    # Generating the page compiles its stylesheet: once per host, not per visit.
    built: list[str] = []

    @host.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the form page.

        Returns
        -------
        str
            The HTML document.
        """
        if not built:
            built.append(page())
        return built[0]

    @host.get("/run.js")
    def script() -> Response:
        """Serve the run panel's script.

        Returns
        -------
        fastapi.responses.Response
            The JavaScript module.
        """
        code = resources.files("autostat").joinpath("static/run.js").read_text()
        return Response(code, media_type="text/javascript")

    @host.get("/i18n.json")
    def i18n(lang: str = "en") -> dict[str, str]:
        """Serve the run panel's strings in the browser's language.

        Parameters
        ----------
        lang : str, optional
            Two-letter language code.

        Returns
        -------
        dict of str to str
            Text by key.
        """
        return strings(lang)

    @host.post("/run")
    async def run(request: Request) -> JSONResponse:
        """Run one analysis from the form's fields.

        Parameters
        ----------
        request : fastapi.Request
            JSON object of field values.

        Returns
        -------
        fastapi.responses.JSONResponse
            The outcome with a run id for its files, or a 422 error envelope.
        """
        fields = await request.json()
        try:
            options = run_options(fields if isinstance(fields, dict) else {})
            outcome = await _in_thread(analyse_file, **options)
        except (OSError, ValueError) as error:
            return JSONResponse(
                {"error": {"code": "invalid_input", "message": str(error)}}, 422
            )
        run_id = uuid.uuid4().hex
        folder = Path(outcome.report).parent if outcome.report else Path()
        runs[run_id] = (folder, outcome.files)
        return JSONResponse({"run_id": run_id, **outcome.to_dict()})

    @host.get("/runs/{run_id}/{name:path}")
    def run_file(run_id: str, name: str) -> FileResponse:
        """Serve one file of a report this host wrote.

        Parameters
        ----------
        run_id : str
            From ``POST /run``.
        name : str
            A file the run listed.

        Returns
        -------
        fastapi.responses.FileResponse
            The file.

        Raises
        ------
        HTTPException
            404 for anything the run did not list.
        """
        if run_id not in runs or name not in runs[run_id][1]:
            raise HTTPException(404, "no such file")
        return FileResponse(runs[run_id][0] / name)

    return host


async def _in_thread(function: Any, **kwargs: Any) -> Any:
    """Run a CPU-bound call off the event loop.

    Parameters
    ----------
    function : callable
        The call.
    **kwargs
        Its keyword arguments.

    Returns
    -------
    Any
        What it returned.
    """
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(function, **kwargs)


@click.command(
    name="autostat-gui",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Open the autostat form in a browser, served from this machine only.",
)
@click.version_option(__version__, "--version", prog_name="autostat-gui")
@click.option("--port", type=int, default=8765, show_default=True, help="Port to bind.")
@click.option(
    "--browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the page in the default browser.",
)
def command(port: int, browser: bool) -> None:
    """Serve the GUI on 127.0.0.1 and open it.

    Parameters
    ----------
    port : int
        Port to bind.
    browser : bool
        Open the page in the default browser.
    """
    import uvicorn

    url = f"http://127.0.0.1:{port}/"
    if browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    click.echo(f"autostat GUI: {url}", err=True)
    uvicorn.run(create_gui_app(), host="127.0.0.1", port=port, workers=1)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``autostat-gui`` console script.

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
            prog_name="autostat-gui",
            standalone_mode=False,
        )
    except click.UsageError as error:
        error.show()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
