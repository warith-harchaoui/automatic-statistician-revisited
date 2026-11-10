"""Draw the figures of the repository report, in SVG (Markdown) and PDF (LaTeX).

What this script does
    It draws the airline passengers series alone, then the BIC of each
    level of two published 2014 searches: airline passengers, which
    converged; internet traffic, which stalled for two levels before
    finding a much better structure. The figures go to ``doc/figures/``.

Usage
    ``python doc/make_figures.py`` (``doc/build.sh`` runs it).

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autostat import benchmarks
from autostat.describe import plot_bic, plot_data
from autostat.stopping import search_path

OUT = Path(__file__).resolve().parent / "figures"
RULE = "the published 2014 search: ten levels, no stopping rule"


def main() -> None:
    """Write every figure in both formats."""
    OUT.mkdir(exist_ok=True)
    x, y = benchmarks.load("01-airline")
    xs = np.asarray(x, float).reshape(len(np.ravel(y)), -1)
    for suffix in (".svg", ".pdf"):
        name = plot_data(xs, np.ravel(y), "airline passengers", OUT, "year", suffix)
        (OUT / name).rename(OUT / f"data_airline{suffix}")
    for series, label in (("01-airline", "airline"), ("06-internet", "internet")):
        xs, ys = benchmarks.load(series)
        path = search_path(benchmarks.published(series), xs, ys, RULE)
        for suffix in (".svg", ".pdf"):
            name = plot_bic(path, OUT, suffix)
            (OUT / name).rename(OUT / f"bic_{label}{suffix}")


if __name__ == "__main__":
    main()
