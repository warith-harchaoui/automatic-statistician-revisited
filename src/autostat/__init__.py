"""autostat: the Automatic Statistician's structure search, in modern Python.

What this package does
    Given a time series, it searches a grammar of Gaussian process kernels
    (smooth, periodic, linear, constant, noise; combined by sums, products,
    changepoints and change windows), scores each structure by the Bayesian
    information criterion (BIC), splits the winner into additive components,
    describes each component in English, checks each against its prior and
    writes a report.

    >>> from autostat import analyse
    >>> result, report = analyse(series, name="my series", out_dir="report/")

    From a shell: ``autostat data.csv --time date --value y``.

Why it exists
    The Automatic Statistician (Duvenaud et al., ICML 2013; Lloyd et al.,
    AAAI 2014) shipped as Python 2.7 driving MATLAB and the GPML toolbox,
    last changed in 2015 and 2017. This package ports both published
    configurations to Python 3 and ``torch``. It reproduces every published
    2014 model's likelihood.

What it exposes
    :func:`analyse` for one series end to end. The building blocks:
    :func:`prepare`, :func:`search`, :class:`Experiment`, :func:`describe`,
    :func:`fit`, :func:`posterior`, :class:`GPModel`. A report's formula is
    :meth:`Report.formula`; :func:`search_path` and :func:`findings` study
    where a search should stop.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>. The ported design is (c) 2013-2015
James Robert Lloyd, David Duvenaud and Roger Grosse, MIT licence, reproduced
in ``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .describe import SECONDS_PER_YEAR, Report, describe
from .formula import Formula
from .model import GPModel, fit, posterior
from .search import Experiment, SearchResult, search
from .stopping import SearchPath, findings, rule_of, search_path

__version__ = "0.1.0"

__all__ = [
    "Experiment",
    "Formula",
    "GPModel",
    "Report",
    "SearchPath",
    "SearchResult",
    "__version__",
    "analyse",
    "describe",
    "findings",
    "fit",
    "posterior",
    "prepare",
    "rule_of",
    "search",
    "search_path",
]


def prepare(
    series: pd.Series, max_points: int = 1000
) -> tuple[np.ndarray, np.ndarray, str]:
    """Turn a time series into the inputs the search expects.

    A datetime index becomes decimal years, the unit the report's sentences
    speak, with the original's 365-day year. A series longer than
    ``max_points`` is averaged into ``max_points`` equal-width time bins:
    exact Gaussian process inference costs ``O(n^3)``. The original kept
    its inputs to 1,000 points for the same reason. A structure finer than
    one bin is a structure this package does not report.

    Parameters
    ----------
    series : pandas.Series
        Values indexed by time (datetime or numeric).
    max_points : int, optional
        Upper bound on the number of points searched.

    Returns
    -------
    x : numpy.ndarray
        Inputs, shape ``(n, 1)``.
    y : numpy.ndarray
        Values, shape ``(n,)``.
    unit : str
        ``"year"`` for a datetime index, ``""`` otherwise.

    Raises
    ------
    ValueError
        When fewer than three finite values remain, or ``max_points < 3``.
    """
    if max_points < 3:
        raise ValueError("max_points must be at least 3")
    series = series[np.isfinite(pd.to_numeric(series, errors="coerce"))].sort_index()
    if len(series) < 3:
        raise ValueError("the series needs at least three finite values")
    if isinstance(series.index, pd.DatetimeIndex):
        index = series.index
        if index.tz is not None:
            index = index.tz_convert(None)
        seconds = (index - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
        x = 1970.0 + seconds / SECONDS_PER_YEAR
        unit = "year"
    else:
        x = series.index.to_numpy(dtype=float)
        unit = ""
    y = series.to_numpy(dtype=float)
    if len(y) > max_points:
        edges = np.linspace(x.min(), x.max(), max_points + 1)
        bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max_points - 1)
        frame = pd.DataFrame({"bin": bins, "x": x, "y": y}).groupby("bin").mean()
        x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
    return x[:, None], y, unit


def analyse(
    series: pd.Series,
    name: str = "data",
    out_dir: str | Path | None = None,
    experiment: Experiment | None = None,
    max_points: int = 1000,
    check: bool = True,
    report_format: str = "markdown",
    pdf: bool = True,
    symbolic: bool = False,
) -> tuple[SearchResult, Report]:
    """Search, describe, check and optionally write a report for one series.

    Parameters
    ----------
    series : pandas.Series
        Values indexed by time.
    name : str, optional
        Name for the report.
    out_dir : str or pathlib.Path, optional
        Where to write the report, its figures and ``search.json``.
    experiment : Experiment, optional
        Search settings; the 2014 preset when omitted.
    max_points : int, optional
        See :func:`prepare`.
    check : bool, optional
        Compute the model-checking section.
    report_format : {"markdown", "latex"}, optional
        ``report.md`` compiled by md2star, or ``report.tex`` compiled by
        ``latexmk``; see :meth:`Report.write`.
    pdf : bool, optional
        Compile ``report.pdf``.
    symbolic : bool, optional
        Look for a formula for each smooth component with PySR.

    Returns
    -------
    SearchResult, Report
        The search and the description of its winner.
    """
    x, y, unit = prepare(series, max_points)
    result = search(x, y, experiment)
    path = search_path(
        result.best_per_level(),
        x,
        y,
        rule_of(result.experiment),
        unit,
        result.experiment.improvement_tolerance,
    )
    report = describe(
        result.best,
        x,
        y,
        name=name,
        unit=unit,
        out_dir=out_dir,
        check=check,
        report_format=report_format,
        pdf=pdf,
        path=path,
        symbolic=symbolic,
    )
    if out_dir is not None:
        result.save(Path(out_dir) / "search.json")
    return result, report
