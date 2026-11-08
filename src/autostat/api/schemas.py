"""Request and response models of the HTTP API.

What this module does
    It declares, with pydantic, what each route accepts and returns. The
    bounds here (series length, search depth) keep one request from
    occupying the server for hours.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReportFormat = Literal["markdown", "latex"]
Preset = Literal["abcd-2014", "cks-2013"]


class ReportOptions(BaseModel):
    """How the report is written.

    Parameters
    ----------
    report_format : {"markdown", "latex"}
        Markdown compiled by md2star, or LaTeX compiled by latexmk.
    check : bool
        Compute the model-checking section.
    pdf : bool
        Compile ``report.pdf``.
    """

    report_format: ReportFormat = Field(
        "markdown",
        description="Markdown compiled by md2star, or LaTeX compiled by latexmk.",
    )
    check: bool = Field(True, description="Compute the model-checking section.")
    pdf: bool = Field(True, description="Compile report.pdf.")


class PublishedRequest(ReportOptions):
    """Describe the model the 2014 system published for one benchmark series.

    Parameters
    ----------
    series : str
        A benchmark series name, e.g. ``"01-airline"``.
    """

    series: str = Field(
        ...,
        description="A name from GET /v1/series, e.g. 01-airline.",
        examples=["01-airline"],
    )


class AnalysisRequest(ReportOptions):
    """Search a series sent as values and describe the best structure.

    Parameters
    ----------
    values : list of float
        The observations, 3 to 1,000 of them.
    times : list of float or str, optional
        Numbers or ISO 8601 dates, one per value.
    name : str
        Dataset name for the report title.
    preset : {"abcd-2014", "cks-2013"}
        Search configuration.
    max_depth : int
        Search levels, 1 to 6.
    """

    values: list[float] = Field(..., min_length=3, max_length=1000)
    times: list[float | str] | None = Field(
        None, description="Numbers or ISO 8601 dates, one per value."
    )
    name: str = Field("data", max_length=200)
    preset: Preset = "abcd-2014"
    max_depth: int = Field(
        2, ge=1, le=6, description="Search levels; each costs several times the last."
    )


class Analysis(BaseModel):
    """What an analysis found, with the report written for it.

    Parameters
    ----------
    report_id : str
        Identifier for GET /v1/reports/{report_id}.
    structure : str
        The selected kernel structure.
    bic : float
        Its Bayesian information criterion.
    components : list of str
        One summary per additive component.
    report_format : {"markdown", "latex"}
        The report's format.
    files : list of str
        The report directory's files.
    """

    report_id: str
    structure: str
    bic: float
    components: list[str]
    report_format: ReportFormat
    files: list[str]


class ReportSource(BaseModel):
    """The source text of a written report.

    Parameters
    ----------
    report_id : str
        The report's identifier.
    report_format : {"markdown", "latex"}
        The source's format.
    source : str
        ``report.md`` or ``report.tex``, verbatim.
    files : list of str
        The report directory's files, readable one by one.
    """

    report_id: str
    report_format: ReportFormat
    source: str
    files: list[str]


class Health(BaseModel):
    """Liveness and version.

    Parameters
    ----------
    status : str
        Always ``"ok"``.
    version : str
        The package version.
    """

    status: str
    version: str


class ErrorBody(BaseModel):
    """One error, as every failing route reports it.

    Parameters
    ----------
    code : str
        Stable machine-readable code, e.g. ``"invalid_input"``.
    message : str
        What went wrong, for a person.
    """

    code: str
    message: str


class ErrorEnvelope(BaseModel):
    """The body of every error response.

    Parameters
    ----------
    error : ErrorBody
        The error.
    """

    error: ErrorBody
