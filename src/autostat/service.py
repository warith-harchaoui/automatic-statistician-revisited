"""Application services shared by every surface: CLI, GUI, HTTP API and MCP.

What this module does
    It turns a request into a finished analysis. It loads or receives a
    series, searches, describes, writes the report in the format asked and
    returns a plain summary. The command line, the GUI host, the HTTP API and the MCP
    tools all call these functions, so the four agree on validation and
    behaviour.

Why it exists
    Surfaces are adapters (CODING.md, section 19). Parsing a flag, reading
    JSON from a request or a form belongs to them; what happens after
    belongs here, once.

What it consumes
    A file path, a list of values with optional times, or the name of a
    benchmark series, plus the search and report options.

What it produces
    :class:`Outcome`: the structure, its BIC, one summary per component and
    the paths of the report files written.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import analyse, benchmarks, describe
from .describe import REPORT_FORMATS
from .search import Experiment
from .stopping import findings, search_path

#: How a published 2014 search was told to stop, in words.
PUBLISHED_RULE = "the published 2014 search: ten levels, no stopping rule"

#: Search configurations, by the name every surface uses.
PRESETS = ("abcd-2014", "cks-2013")


@dataclass
class Outcome:
    """What one analysis found and where its report went.

    Parameters
    ----------
    structure : str
        The selected kernel structure.
    bic : float
        Its Bayesian information criterion.
    components : list of str
        One summary sentence per additive component, in report order.
    report : str or None
        The report's source file (``report.md`` or ``report.tex``).
    pdf : str or None
        The compiled PDF, when one was written.
    files : list of str
        Every file of the report directory, relative to it.
    formula : dict or None
        The model as mathematics: see :class:`autostat.formula.Formula`.
    search_path : dict or None
        The search level by level with what it shows about stopping.
    """

    structure: str
    bic: float
    components: list[str]
    report: str | None = None
    pdf: str | None = None
    files: list[str] = field(default_factory=list)
    formula: dict[str, Any] | None = None
    search_path: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the outcome as JSON-ready data.

        Returns
        -------
        dict
            The fields, by name.
        """
        return asdict(self)


def experiment_for(
    preset: str, max_depth: int, n_jobs: int = -1, seed: int = 1, patience: int = 0
) -> Experiment:
    """Build the search settings a surface asked for.

    Parameters
    ----------
    preset : {"abcd-2014", "cks-2013"}
        Search configuration.
    max_depth : int
        Search levels.
    n_jobs : int, optional
        Worker processes; -1 uses every core.
    seed : int, optional
        Random seed.
    patience : int, optional
        Stop after this many levels without a BIC fall (0: search all
        ``max_depth`` levels, as published).

    Returns
    -------
    Experiment
        The settings.

    Raises
    ------
    ValueError
        On an unknown preset or a depth below 1.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, not {preset!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1, not {max_depth}")
    if patience < 0:
        raise ValueError(f"patience must be 0 or more, not {patience}")
    factory = Experiment.cks_2013 if preset == "cks-2013" else Experiment.abcd_2014
    settings: dict[str, Any] = {
        "max_depth": max_depth,
        "n_jobs": n_jobs,
        "random_seed": seed,
    }
    if patience:
        settings.update(stopping_criteria=["no_improvement"], patience=patience)
    return factory(**settings)


def check_format(report_format: str) -> None:
    """Refuse an unknown report format before any work starts.

    Parameters
    ----------
    report_format : str
        ``"markdown"`` or ``"latex"``.

    Raises
    ------
    ValueError
        On any other value.
    """
    if report_format not in REPORT_FORMATS:
        raise ValueError(
            f"report_format must be one of {REPORT_FORMATS}, not {report_format!r}"
        )


def load_series(path: Path, time: str | None, value: str | None) -> pd.Series:
    """Read one series from a CSV or Parquet file.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    time : str or None
        Time column; the file's index when ``None``.
    value : str or None
        Value column; the first numeric column when ``None``.

    Returns
    -------
    pandas.Series
        Values indexed by time, named after the value column.

    Raises
    ------
    ValueError
        When a named column is missing or no numeric column exists.
    """
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if time is not None:
        if time not in frame.columns:
            raise ValueError(f"no column {time!r} in {path}")
        index = frame[time]
        if not pd.api.types.is_numeric_dtype(index):
            index = pd.to_datetime(index, utc=True)
        frame = frame.drop(columns=[time]).set_index(index)
    if value is None:
        numeric = frame.select_dtypes("number").columns
        if numeric.empty:
            raise ValueError(f"no numeric column in {path}")
        value = str(numeric[0])
    if value not in frame.columns:
        raise ValueError(f"no column {value!r} in {path}")
    return frame[value].rename(value)


def series_from_values(
    values: Sequence[float], times: Sequence[float | str] | None = None
) -> pd.Series:
    """Build a series from values and optional times, as a request carries them.

    Parameters
    ----------
    values : sequence of float
        The observations.
    times : sequence of float or str, optional
        Numbers, or dates in ISO 8601; ``0, 1, 2, ...`` when omitted.

    Returns
    -------
    pandas.Series
        Values indexed by time.

    Raises
    ------
    ValueError
        When the lengths differ or fewer than three values are given.
    """
    if len(values) < 3:
        raise ValueError("a series needs at least three values")
    if times is None:
        index: Any = pd.RangeIndex(len(values))
    elif len(times) != len(values):
        raise ValueError(f"{len(times)} times for {len(values)} values")
    elif all(isinstance(t, int | float) for t in times):
        index = pd.Index([float(t) for t in times])
    else:
        index = pd.to_datetime(list(times), utc=True)
    return pd.Series([float(v) for v in values], index=index, name="value")


def _path_summary(report: Any) -> dict[str, Any] | None:
    """Turn a report's search path into JSON-ready data.

    Parameters
    ----------
    report : Report
        A description, with or without a search path.

    Returns
    -------
    dict or None
        The levels, the rule, the kept and settled levels, the findings.
    """
    path = report.path
    if path is None:
        return None
    return {
        "rule": path.rule,
        "tolerance": path.tolerance,
        "kept_level": path.kept.level,
        "settled_level": path.settled.level,
        "steps": [asdict(step) for step in path.steps],
        "findings": findings(path, report.model.ndata),
    }


def _outcome(result_best: Any, report: Any, out_dir: Path | None) -> Outcome:
    """Summarise a described model and list its report files.

    Parameters
    ----------
    result_best : GPModel
        The model described.
    report : Report
        Its description.
    out_dir : pathlib.Path or None
        Where the report was written, if anywhere.

    Returns
    -------
    Outcome
        The summary.
    """
    outcome = Outcome(
        structure=result_best.kernel.structure(),
        bic=float(result_best.bic),
        components=[summary for summary, _, _ in report.texts],
        formula=asdict(report.formula()),
        search_path=_path_summary(report),
    )
    if out_dir is not None:
        source = next(
            (
                out_dir / n
                for n in ("report.md", "report.tex")
                if (out_dir / n).exists()
            ),
            None,
        )
        outcome.report = None if source is None else str(source)
        pdf = out_dir / "report.pdf"
        outcome.pdf = str(pdf) if pdf.exists() else None
        outcome.files = sorted(
            str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()
        )
    return outcome


def analyse_series(
    series: pd.Series,
    name: str,
    out_dir: Path | None,
    preset: str = "abcd-2014",
    max_depth: int = 10,
    max_points: int = 1000,
    n_jobs: int = -1,
    seed: int = 1,
    check: bool = True,
    report_format: str = "markdown",
    pdf: bool = True,
    patience: int = 0,
    symbolic: bool = False,
) -> Outcome:
    """Search one series, describe the winner and write its report.

    Parameters
    ----------
    series : pandas.Series
        Values indexed by time.
    name : str
        Dataset name for the report title.
    out_dir : pathlib.Path or None
        Report directory; nothing is written when ``None``.
    preset : {"abcd-2014", "cks-2013"}, optional
        Search configuration.
    max_depth : int, optional
        Search levels.
    max_points : int, optional
        Longer series are averaged into this many bins.
    n_jobs : int, optional
        Worker processes; -1 uses every core.
    seed : int, optional
        Random seed.
    check : bool, optional
        Compute the model-checking section.
    report_format : {"markdown", "latex"}, optional
        ``report.md`` compiled by md2star, or ``report.tex`` by ``latexmk``.
    pdf : bool, optional
        Compile ``report.pdf``.
    patience : int, optional
        Stop after this many levels without a BIC fall; 0 searches all
        ``max_depth`` levels.
    symbolic : bool, optional
        Look for a formula for each smooth component with PySR.

    Returns
    -------
    Outcome
        The structure, its score, the component summaries and the files.
    """
    check_format(report_format)
    experiment = experiment_for(preset, max_depth, n_jobs, seed, patience)
    result, report = analyse(
        series,
        name,
        out_dir,
        experiment,
        max_points,
        check=check,
        report_format=report_format,
        pdf=pdf,
        symbolic=symbolic,
    )
    return _outcome(result.best, report, None if out_dir is None else Path(out_dir))


def analyse_file(
    path: Path,
    time: str | None = None,
    value: str | None = None,
    name: str | None = None,
    out_dir: Path | None = None,
    **options: Any,
) -> Outcome:
    """Read a CSV or Parquet file and analyse its series.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    time, value : str, optional
        Column names; see :func:`load_series`.
    name : str, optional
        Dataset name; ``"<file stem>: <column>"`` when omitted.
    out_dir : pathlib.Path, optional
        Report directory; ``autostat_<stem>_<column>`` when omitted.
    **options
        Passed to :func:`analyse_series`.

    Returns
    -------
    Outcome
        See :func:`analyse_series`.
    """
    series = load_series(path, time, value)
    name = name or f"{path.stem}: {series.name}"
    out_dir = out_dir or Path(f"autostat_{path.stem}_{series.name}")
    return analyse_series(series, name, out_dir, **options)


def describe_published(
    series: str,
    out_dir: Path | None,
    check: bool = True,
    report_format: str = "markdown",
    pdf: bool = True,
    symbolic: bool = False,
) -> Outcome:
    """Describe the model the 2014 system published for a benchmark series.

    Parameters
    ----------
    series : str
        A name from :func:`autostat.benchmarks.names`, e.g. ``"01-airline"``.
    out_dir : pathlib.Path or None
        Report directory; nothing is written when ``None``.
    check : bool, optional
        Compute the model-checking section.
    report_format : {"markdown", "latex"}, optional
        See :func:`analyse_series`.
    pdf : bool, optional
        Compile ``report.pdf``.
    symbolic : bool, optional
        Look for a formula for each smooth component with PySR.

    Returns
    -------
    Outcome
        The published structure, its BIC and the component summaries.

    Raises
    ------
    ValueError
        On an unknown series name.
    """
    check_format(report_format)
    if series not in benchmarks.names():
        raise ValueError(f"unknown series {series!r}; see list_series()")
    x, y = benchmarks.load(series)
    published = benchmarks.published(series)
    model = published[-1]
    report = describe(
        model,
        x,
        y,
        name=series,
        out_dir=out_dir,
        check=check,
        report_format=report_format,
        pdf=pdf,
        path=search_path(published, x, y, PUBLISHED_RULE),
        symbolic=symbolic,
    )
    return _outcome(model, report, None if out_dir is None else Path(out_dir))


def list_series() -> list[str]:
    """Name the thirteen benchmark series of the 2014 paper.

    Returns
    -------
    list of str
        Series names, in the paper's order.
    """
    return list(benchmarks.names())
