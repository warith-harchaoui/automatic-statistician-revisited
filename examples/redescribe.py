"""Rewrite the reports of the search examples from their saved searches.

What this script does
    Examples 3 to 6 save their search (``search.json``). This script reads
    each one back and writes its reports again, in both formats, without
    searching: the way to apply a change in how reports are written to a
    search that took hours.

Usage
    ``python examples/redescribe.py`` (all four) or
    ``python examples/redescribe.py 4 6`` (the ones named)

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from autostat import benchmarks, describe, prepare
from autostat.search import SearchResult
from autostat.stopping import rule_of, search_path

HERE = Path(__file__).parent


def redescribe(
    folder: Path, name: str, x, y, unit: str = "year", check: bool = True
) -> None:
    """Rewrite one example's two reports from its saved search.

    Parameters
    ----------
    folder : pathlib.Path
        The example's report folder, holding ``search.json`` directly or in
        ``markdown/``.
    name : str
        Dataset name for the title.
    x, y : array_like
        The data the search ran on.
    unit : str, optional
        ``"year"`` or ``""``.
    check : bool, optional
        Compute the model checks.
    """
    saved = folder / "search.json"
    if not saved.exists():
        saved = folder / "markdown" / "search.json"
    result = SearchResult.load(saved)
    path = search_path(result.best_per_level(), x, y, rule_of(result.experiment), unit)
    report = describe(
        result.best,
        x,
        y,
        name=name,
        unit=unit,
        out_dir=folder / "markdown",
        check=check,
        path=path,
        symbolic=True,
    )
    report.write(folder / "latex", "latex")
    print(folder.name, result.best.kernel.structure(), round(result.best.bic, 1))


def main(only: list[str]) -> None:
    """Rewrite the reports of examples 3, 4, 5 and 6, or of those named.

    Parameters
    ----------
    only : list of str
        Example numbers to rewrite, e.g. ``["4", "6"]``; all four when empty.
    """
    wanted = set(only) or {"3", "4", "5", "6"}
    x, y = benchmarks.load("01-airline")
    if "3" in wanted:
        redescribe(HERE / "03_search_report", "airline passengers", x, y)
    if "6" in wanted:
        redescribe(HERE / "06_cks_2013_report", "airline passengers", x, y)
    sales = pd.read_csv(HERE / "sales.csv", parse_dates=["month"])
    xs, ys, unit = prepare(sales.set_index("month")["units"])
    if "4" in wanted:
        redescribe(HERE / "04_sales_report", "sales: units", xs, ys, unit, check=False)
    if "5" in wanted:
        redescribe(HERE / "05_sales_report", "monthly sales", xs, ys, unit)


if __name__ == "__main__":
    main(sys.argv[1:])
