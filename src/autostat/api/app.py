"""The FastAPI application factory, its routes and its error envelope.

What this module does
    :func:`create_app` builds the application: five routes over
    :mod:`autostat.service`, one error envelope for every failure, plus a
    report store on disk. Routes are plain ``def``: the work is CPU-bound,
    so FastAPI runs it in its thread pool, off the event loop.

Where reports go
    Under ``AUTOSTAT_REPORTS_DIR`` (default ``./autostat_reports``), one
    directory per analysis, named by a random identifier. Responses carry
    the identifier and file names relative to it, never a server path.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .. import __version__, service
from .schemas import (
    Analysis,
    AnalysisRequest,
    ErrorEnvelope,
    Health,
    PublishedRequest,
    ReportSource,
)

#: Error responses every report route may give, for the OpenAPI document.
ERRORS = {
    404: {"model": ErrorEnvelope, "description": "No such report or file."},
    422: {"model": ErrorEnvelope, "description": "Invalid input."},
}

_REPORT_ID = re.compile(r"^[0-9a-f]{32}$")


def reports_root() -> Path:
    """Return the directory reports are written under, creating it.

    Returns
    -------
    pathlib.Path
        ``AUTOSTAT_REPORTS_DIR``, or ``./autostat_reports``.
    """
    root = Path(os.environ.get("AUTOSTAT_REPORTS_DIR", "autostat_reports"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _report_dir(report_id: str) -> Path:
    """Resolve a report identifier to its directory, or answer 404.

    Parameters
    ----------
    report_id : str
        32 hexadecimal digits, as :func:`_new_report` makes them.

    Returns
    -------
    pathlib.Path
        The report directory.

    Raises
    ------
    HTTPException
        404 when the identifier is malformed or unknown.
    """
    folder = reports_root() / report_id
    if not _REPORT_ID.match(report_id) or not folder.is_dir():
        raise HTTPException(404, f"no report {report_id!r}")
    return folder


def _new_report() -> tuple[str, Path]:
    """Reserve a fresh report directory.

    Returns
    -------
    str, pathlib.Path
        The identifier and its directory.
    """
    report_id = uuid.uuid4().hex
    return report_id, reports_root() / report_id


def _analysis(report_id: str, outcome: service.Outcome, report_format: str) -> Analysis:
    """Turn a service outcome into a response without server paths.

    Parameters
    ----------
    report_id : str
        The report's identifier.
    outcome : service.Outcome
        What the service returned.
    report_format : str
        The format written.

    Returns
    -------
    Analysis
        The response body.
    """
    return Analysis(
        report_id=report_id,
        structure=outcome.structure,
        bic=outcome.bic,
        components=outcome.components,
        report_format=report_format,
        files=outcome.files,
        formula=outcome.formula,
        search_path=outcome.search_path,
    )


router = APIRouter()


@router.get("/health", operation_id="health", tags=["meta"], response_model=Health)
def health() -> Health:
    """Say the server is up with the version it runs.

    Returns
    -------
    Health
        Status and version.
    """
    return Health(status="ok", version=__version__)


@router.get(
    "/v1/series",
    operation_id="list_series",
    tags=["series"],
    summary="List the 2014 benchmark series",
    response_model=list[str],
)
def list_series() -> list[str]:
    """Name the thirteen benchmark series of the AAAI 2014 paper.

    Use a name with POST /v1/published.

    Returns
    -------
    list of str
        Series names, e.g. ``01-airline``.
    """
    return service.list_series()


@router.post(
    "/v1/published",
    operation_id="describe_published",
    tags=["reports"],
    summary="Describe a model the 2014 system published",
    response_model=Analysis,
    responses=ERRORS,
)
def describe_published(request: PublishedRequest) -> Analysis:
    """Write the English report of the final 2014 model for one benchmark series.

    No search runs: the published model is re-scored and described, in
    about a minute with the model checks.

    Parameters
    ----------
    request : PublishedRequest
        The series and the report options.

    Returns
    -------
    Analysis
        The structure, its score, the component summaries and the report's id.
    """
    report_id, folder = _new_report()
    outcome = service.describe_published(
        request.series,
        folder,
        check=request.check,
        report_format=request.report_format,
        pdf=request.pdf,
        symbolic=request.symbolic,
    )
    return _analysis(report_id, outcome, request.report_format)


@router.post(
    "/v1/analyses",
    operation_id="analyse_series",
    tags=["reports"],
    summary="Search a series and describe the best structure",
    response_model=Analysis,
    responses=ERRORS,
)
def analyse_series(request: AnalysisRequest) -> Analysis:
    """Run the structure search on values sent in the request and write a report.

    The search is bounded to 1,000 points and six levels, or twenty with a
    patience, which stops it once BIC has not fallen for that many levels.
    Two levels take seconds to minutes, each further level several times the
    last. The response holds the formula found and the search path.

    Parameters
    ----------
    request : AnalysisRequest
        The series, the search settings and the report options.

    Returns
    -------
    Analysis
        The structure, its score, the component summaries and the report's id.
    """
    series = service.series_from_values(request.values, request.times)
    report_id, folder = _new_report()
    outcome = service.analyse_series(
        series,
        request.name,
        folder,
        preset=request.preset,
        max_depth=request.max_depth,
        patience=request.patience,
        symbolic=request.symbolic,
        n_jobs=1,
        check=request.check,
        report_format=request.report_format,
        pdf=request.pdf,
    )
    return _analysis(report_id, outcome, request.report_format)


@router.get(
    "/v1/reports/{report_id}",
    operation_id="get_report",
    tags=["reports"],
    summary="Read a report's source",
    response_model=ReportSource,
    responses=ERRORS,
)
def get_report(report_id: str) -> ReportSource:
    """Return the Markdown or LaTeX source of a report written earlier.

    Parameters
    ----------
    report_id : str
        From an earlier analysis.

    Returns
    -------
    ReportSource
        The source text and the list of the report's files.
    """
    folder = _report_dir(report_id)
    latex = (folder / "report.tex").exists()
    source = folder / ("report.tex" if latex else "report.md")
    return ReportSource(
        report_id=report_id,
        report_format="latex" if latex else "markdown",
        source=source.read_text(),
        files=sorted(p.name for p in folder.iterdir() if p.is_file()),
    )


@router.get(
    "/v1/reports/{report_id}/files/{name}",
    operation_id="get_report_file",
    tags=["reports"],
    summary="Download one file of a report",
    responses=ERRORS,
)
def get_report_file(report_id: str, name: str) -> FileResponse:
    """Send one file of a report: the PDF, a figure or the source.

    Parameters
    ----------
    report_id : str
        From an earlier analysis.
    name : str
        A file name listed by GET /v1/reports/{report_id}.

    Returns
    -------
    fastapi.responses.FileResponse
        The file.

    Raises
    ------
    HTTPException
        404 for a name the report does not list.
    """
    folder = _report_dir(report_id)
    listed = {p.name for p in folder.iterdir() if p.is_file()}
    if name not in listed:
        raise HTTPException(404, f"report {report_id} has no file {name!r}")
    return FileResponse(folder / name, filename=name)


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    """Build the one error body every route uses.

    Parameters
    ----------
    status : int
        HTTP status.
    code : str
        Stable machine-readable code.
    message : str
        What went wrong.

    Returns
    -------
    fastapi.responses.JSONResponse
        The response.
    """
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


def create_app() -> FastAPI:
    """Build the application: routes, error envelope, OpenAPI metadata.

    Returns
    -------
    fastapi.FastAPI
        The application, ready for ``uvicorn``.
    """
    api = FastAPI(
        title="autostat",
        version=__version__,
        description=(
            "The Automatic Statistician, revisited: search Gaussian process "
            "kernel structures for a time series and write an English report "
            "of the best one, in Markdown or LaTeX."
        ),
    )
    api.include_router(router)

    @api.exception_handler(HTTPException)
    def _http_error(_: Request, error: HTTPException) -> JSONResponse:
        """Wrap an HTTP error in the envelope.

        Parameters
        ----------
        _ : fastapi.Request
            The request, unused.
        error : HTTPException
            The error raised by a route.

        Returns
        -------
        fastapi.responses.JSONResponse
            The enveloped error.
        """
        code = "not_found" if error.status_code == 404 else "http_error"
        return _envelope(error.status_code, code, str(error.detail))

    @api.exception_handler(RequestValidationError)
    def _invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        """Report a request the schemas reject.

        Parameters
        ----------
        _ : fastapi.Request
            The request, unused.
        error : RequestValidationError
            Pydantic's account of what failed.

        Returns
        -------
        fastapi.responses.JSONResponse
            422 with the first failure in plain words.
        """
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"][1:])
        return _envelope(422, "invalid_input", f"{where}: {first['msg']}")

    @api.exception_handler(ValueError)
    def _invalid_value(_: Request, error: ValueError) -> JSONResponse:
        """Report input the service refused, such as an unknown series.

        Parameters
        ----------
        _ : fastapi.Request
            The request, unused.
        error : ValueError
            The service's reason.

        Returns
        -------
        fastapi.responses.JSONResponse
            422 with that reason.
        """
        return _envelope(422, "invalid_input", str(error))

    return api
