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

from pydantic import BaseModel, Field, model_validator

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
    symbolic : bool
        Look for a formula for each smooth component with PySR.
    """

    report_format: ReportFormat = Field(
        "markdown",
        description="Markdown compiled by md2star, or LaTeX compiled by latexmk.",
    )
    check: bool = Field(True, description="Compute the model-checking section.")
    pdf: bool = Field(True, description="Compile report.pdf.")
    symbolic: bool = Field(
        False,
        description="Look for a formula for each smooth component with PySR; slower.",
    )


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
        Search levels: 1 to 6, or up to 20 with a patience.
    patience : int
        Levels searched past the last BIC fall before stopping; 0 searches
        all ``max_depth`` levels.
    """

    values: list[float] = Field(..., min_length=3, max_length=1000)
    times: list[float | str] | None = Field(
        None, description="Numbers or ISO 8601 dates, one per value."
    )
    name: str = Field("data", max_length=200)
    preset: Preset = "abcd-2014"
    max_depth: int = Field(
        2,
        ge=1,
        le=20,
        description="Search levels, each costing several times the last; over 6 needs a patience.",
    )
    patience: int = Field(
        0,
        ge=0,
        le=3,
        description="Levels searched past the last BIC fall before stopping; 0 for none.",
    )

    @model_validator(mode="after")
    def bounded(self) -> AnalysisRequest:
        """Refuse a deep search with nothing to stop it.

        Returns
        -------
        AnalysisRequest
            The request, unchanged.

        Raises
        ------
        ValueError
            When ``max_depth`` is over 6 with no patience.
        """
        if self.max_depth > 6 and self.patience == 0:
            raise ValueError("max_depth over 6 needs a patience of at least 1")
        return self


class FormulaBody(BaseModel):
    """The expression a search kept, as LaTeX strings.

    Parameters
    ----------
    model : str
        ``y(t) = ...``, one term per component.
    laws : list of str
        Each component's Gaussian process with its covariance written out.
    explicit : list of str or None
        Each component's closed form, or None when it has none.
    noise : str or None
        The likelihood's noise, when the model has one outside its kernel.
    switches : list of str
        Definitions of the change-of-regime weights the laws use.
    unit : str
        What ``t`` is measured in.
    notes : list of str or None
        How each closed form was obtained and how close it is.
    """

    model: str
    laws: list[str]
    explicit: list[str | None]
    noise: str | None
    switches: list[str]
    unit: str
    notes: list[str | None]


class StepBody(BaseModel):
    """One level of a search.

    Parameters
    ----------
    level : int
        Level number, from 1.
    bic, nll : float
        The winner's BIC and negative log likelihood.
    parameters : int
        Parameters BIC charges for.
    structure : str
        The winner's structure.
    components, closed_forms : int
        Additive components; how many have a closed form.
    """

    level: int
    bic: float
    nll: float
    parameters: int
    structure: str
    components: int
    closed_forms: int


class SearchPathBody(BaseModel):
    """A search level by level, with what it shows about stopping.

    Parameters
    ----------
    rule : str
        How the search was told to stop.
    tolerance : float
        The BIC fall that counts as an improvement.
    kept_level, settled_level : int
        The level kept; the level where BIC stopped falling.
    steps : list of StepBody
        One per level.
    findings : list of str
        The readings, as sentences.
    """

    rule: str
    tolerance: float
    kept_level: int
    settled_level: int
    steps: list[StepBody]
    findings: list[str]


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
    formula : FormulaBody
        The expression found, written as mathematics.
    search_path : SearchPathBody, optional
        The search level by level with what it shows about stopping.
    """

    report_id: str
    structure: str
    bic: float
    components: list[str]
    report_format: ReportFormat
    files: list[str]
    formula: FormulaBody
    search_path: SearchPathBody | None = None


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
