"""Command-line interface: ``autostat data.csv --time date --value y``.

What this module does
    It reads one series from a CSV (comma-separated values) or Parquet file,
    runs the structure search, writes a report directory and prints the
    selected structure with one line per component.

Why it exists as a separate module
    The command line is an adapter: it parses arguments, loads a file and
    calls :func:`autostat.analyse`. Keeping the parser in its own module lets
    tests build it and call :func:`main` without a subprocess.

What it consumes
    A file path and options. Diagnostics go to standard error through
    :mod:`logging`.

What it produces
    On standard output: the structure, its BIC (Bayesian information
    criterion), one summary per component and the report path; or one JSON
    object with ``--json``. Exit status 0 on success, 2 on invalid input
    (the ``argparse`` convention).

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import Experiment, __version__, analyse

LOGGER = logging.getLogger("autostat")

#: Exit status for invalid input, as ``argparse`` uses for usage errors.
EXIT_INVALID = 2


def make_parser() -> argparse.ArgumentParser:
    """Build the argument parser without running anything.

    Returns
    -------
    argparse.ArgumentParser
        The parser for :func:`main`.
    """
    parser = argparse.ArgumentParser(
        prog="autostat",
        description=(
            "Search Gaussian process kernel structures for one time series "
            "and write a report describing the best one."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("path", type=Path, help="CSV or Parquet file")
    parser.add_argument("--time", help="time column (default: the file's index)")
    parser.add_argument("--value", help="value column (default: the first numeric one)")
    parser.add_argument("--out", type=Path, help="report directory")
    parser.add_argument("--name", help="dataset name for the report title")
    parser.add_argument(
        "--preset",
        choices=("abcd-2014", "cks-2013"),
        default="abcd-2014",
        help="search configuration (default: abcd-2014)",
    )
    parser.add_argument("--max-depth", type=int, default=10, help="search levels")
    parser.add_argument(
        "--max-points", type=int, default=1000, help="bin longer series to this size"
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="worker processes")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--no-check", action="store_true", help="skip the model-checking section"
    )
    parser.add_argument(
        "--json", action="store_true", help="print one JSON object instead of text"
    )
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 on success, :data:`EXIT_INVALID` on invalid input.
    """
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        series = load_series(args.path, args.time, args.value)
    except (OSError, ValueError) as error:
        LOGGER.error("autostat: %s", error)
        return EXIT_INVALID
    preset = Experiment.cks_2013 if args.preset == "cks-2013" else Experiment.abcd_2014
    experiment = preset(
        max_depth=args.max_depth, n_jobs=args.n_jobs, random_seed=args.seed
    )
    name = args.name or f"{args.path.stem}: {series.name}"
    out = args.out or Path(f"autostat_{args.path.stem}_{series.name}")
    try:
        result, report = analyse(
            series, name, out, experiment, args.max_points, check=not args.no_check
        )
    except ValueError as error:
        LOGGER.error("autostat: %s", error)
        return EXIT_INVALID
    summaries = [summary for summary, _, _ in report.texts]
    if args.json:
        payload = {
            "structure": result.best.kernel.structure(),
            "bic": result.best.bic,
            "components": summaries,
            "report": str(out / "report.md"),
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    sys.stdout.write(f"{result.best.kernel.structure()}  (BIC {result.best.bic:.1f})\n")
    for i, summary in enumerate(summaries, 1):
        sys.stdout.write(f"  {i}. {summary}\n")
    sys.stdout.write(f"report: {out / 'report.md'}\n")
    return 0
