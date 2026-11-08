"""Command-line interface: ``autostat data.csv --time date --value y``.

What this module does
    It reads one series from a CSV (comma-separated values) or Parquet file,
    runs the structure search, writes a report directory and prints the
    selected structure with one line per component.

Why it exists as a separate module
    The command line is an adapter: Click parses the arguments and
    :func:`autostat.service.analyse_file` does the work, as it does for the
    GUI, the HTTP API and the MCP tools. :data:`command` is importable
    without side effects, so the GUI generator can walk it and tests can
    call :func:`main` without a subprocess.

What it consumes
    A file path and options. Diagnostics go to standard error through
    :mod:`logging`.

What it produces
    On standard output: the structure, its BIC (Bayesian information
    criterion), one summary per component and the report path; or one JSON
    object with ``--json``. Exit status 0 on success, 2 on invalid input
    (Click's usage-error status).

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import click

from . import __version__
from .describe import REPORT_FORMATS
from .service import PRESETS, analyse_file

LOGGER = logging.getLogger("autostat")

#: Exit status for invalid input, as Click uses for usage errors.
EXIT_INVALID = 2


@click.command(
    name="autostat",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Search Gaussian process kernel structures for one time series and "
        "write a report describing the best one."
    ),
)
@click.version_option(__version__, "--version", prog_name="autostat")
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--time", help="Time column (default: the file's index).")
@click.option("--value", help="Value column (default: the first numeric one).")
@click.option("--out", type=click.Path(path_type=Path), help="Report directory.")
@click.option("--name", help="Dataset name for the report title.")
@click.option(
    "--preset",
    type=click.Choice(PRESETS),
    default="abcd-2014",
    show_default=True,
    help="Search configuration.",
)
@click.option(
    "--max-depth", type=int, default=10, show_default=True, help="Search levels."
)
@click.option(
    "--max-points",
    type=int,
    default=1000,
    show_default=True,
    help="Bin longer series to this size.",
)
@click.option(
    "--n-jobs", type=int, default=-1, show_default=True, help="Worker processes."
)
@click.option("--seed", type=int, default=1, show_default=True, help="Random seed.")
@click.option(
    "--patience",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help=(
        "Stop after this many levels without a BIC fall, to study what the levels "
        "past the best one find; 0 searches all --max-depth levels."
    ),
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(REPORT_FORMATS),
    default="markdown",
    show_default=True,
    help="Report source: Markdown compiled by md2star, or LaTeX compiled by latexmk.",
)
@click.option(
    "--pdf/--no-pdf",
    default=True,
    show_default=True,
    help="Compile the report to report.pdf.",
)
@click.option(
    "--symbolic",
    is_flag=True,
    help="Look for a formula for each smooth component with PySR (optional extra).",
)
@click.option("--no-check", is_flag=True, help="Skip the model-checking section.")
@click.option(
    "--json", "as_json", is_flag=True, help="Print one JSON object instead of text."
)
def command(
    path: Path,
    time: str | None,
    value: str | None,
    out: Path | None,
    name: str | None,
    preset: str,
    max_depth: int,
    max_points: int,
    n_jobs: int,
    seed: int,
    patience: int,
    report_format: str,
    pdf: bool,
    symbolic: bool,
    no_check: bool,
    as_json: bool,
) -> int:
    """Run one analysis and print what it found.

    Parameters
    ----------
    path : pathlib.Path
        CSV or Parquet file.
    time, value : str or None
        Column names.
    out : pathlib.Path or None
        Report directory.
    name : str or None
        Dataset name.
    preset : str
        Search configuration.
    max_depth : int
        Search levels.
    max_points : int
        Bin longer series to this size.
    n_jobs : int
        Worker processes.
    seed : int
        Random seed.
    patience : int
        Levels without a BIC fall before stopping; 0 for none.
    report_format : str
        ``"markdown"`` or ``"latex"``.
    pdf : bool
        Compile ``report.pdf``.
    symbolic : bool
        Look for formulas for smooth components with PySR.
    no_check : bool
        Skip the model checks.
    as_json : bool
        Print one JSON object.

    Returns
    -------
    int
        0 on success, :data:`EXIT_INVALID` on invalid input.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        outcome = analyse_file(
            path,
            time=time,
            value=value,
            name=name,
            out_dir=out,
            preset=preset,
            max_depth=max_depth,
            max_points=max_points,
            n_jobs=n_jobs,
            seed=seed,
            patience=patience,
            symbolic=symbolic,
            check=not no_check,
            report_format=report_format,
            pdf=pdf,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("autostat: %s", error)
        return EXIT_INVALID
    if as_json:
        payload = {
            "structure": outcome.structure,
            "bic": outcome.bic,
            "components": outcome.components,
            "report": outcome.report,
            "pdf": outcome.pdf,
            "formula": outcome.formula,
            "search_path": outcome.search_path,
        }
        click.echo(json.dumps(payload))
        return 0
    click.echo(f"{outcome.structure}  (BIC {outcome.bic:.1f})")
    for i, summary in enumerate(outcome.components, 1):
        click.echo(f"  {i}. {summary}")
    click.echo(f"report: {outcome.report}")
    if outcome.pdf:
        click.echo(f"pdf: {outcome.pdf}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status.

    ``--help`` and ``--version`` print and return 0.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 on success, :data:`EXIT_INVALID` on invalid input.
    """
    try:
        status = command.main(
            args=list(sys.argv[1:] if argv is None else argv),
            prog_name="autostat",
            standalone_mode=False,
        )
    except click.UsageError as error:
        error.show()
        return EXIT_INVALID
    return 0 if status is None else int(status)
