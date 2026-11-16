"""Summarise the recovery study: how often the search finds what drew a series.

What this script does
    It reads the rows ``run.py`` wrote and prints, as Markdown, how often
    each kind of component (cycle, changepoint or window, linear trend,
    smooth term) is found when present and invented when absent; how often
    a true period is found within 5 %; how the misses split between
    the search and the data. When the true structure refitted
    scores a lower BIC than the search's model by more than 2, the search
    missed a better model; otherwise the data do not single out the truth.

Usage
    ``python doc/research/recovery/summarise.py results.json``

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: The kinds of component compared, with their names in the table.
KINDS = {
    "periodic": "cycle",
    "switch": "changepoint or window",
    "linear": "linear trend",
    "smooth": "smooth term",
}

#: BIC difference above which the truth is called better than the finding.
MARGIN = 2.0


def main() -> int:
    """Print the summary of the rows in the file given.

    Returns
    -------
    int
        0.
    """
    path = Path(sys.argv[1])
    rows = json.loads(path.read_text(encoding="utf-8"))
    n = len(rows)
    lines = [
        f"{n} series drawn from the grammar.",
        "",
        "| component | present | found when present | absent | invented when absent |",
        "|---|---|---|---|---|",
    ]
    for key, label in KINDS.items():
        present = [r for r in rows if r[f"truth_{key}"]]
        absent = [r for r in rows if not r[f"truth_{key}"]]
        found = sum(r[f"found_{key}"] for r in present)
        invented = sum(r[f"found_{key}"] for r in absent)
        lines.append(
            f"| {label} | {len(present)} | {found} | {len(absent)} | {invented} |"
        )
    # A cycle multiplied by white noise is white noise once simplified: only
    # the cycles that survive simplification can be found.
    cycles = [r for r in rows if r["truth_periodic"]]
    # ``period_recovered`` asked for every drawn period, absorbed ones
    # included; count the series where one true period is found instead.
    periods = sum(
        any(
            abs(f / p - 1) <= 0.05
            for p in r["periods_truth"]
            for f in r["periods_found"]
        )
        for r in cycles
    )
    exact = sum(r["found"] == r["truth"] for r in rows)
    worse = [r for r in rows if r["bic_truth"] < r["bic_found"] - MARGIN]
    misfit = [r for r in worse if r["found"] == r["truth"]]
    search_failed = [r for r in worse if r["found"] != r["truth"]]
    as_good = n - len(worse)
    lines += [
        "",
        f"- A true period found within 5 %: {periods} of {len(cycles)} "
        "series with a cycle.",
        f"- The exact structure: {exact} of {n}.",
        f"- The search's model scores within {MARGIN:g} of the refitted truth, "
        f"or better: {as_good} of {n}.",
        f"- The truth scores better by more than {MARGIN:g} with another "
        f"structure found (the search missed it): {len(search_failed)}; with "
        f"the right structure found but fitted worse (the optimiser stopped "
        f"short): {len(misfit)}.",
    ]
    for r in worse:
        lines.append(
            f"  - seed {r['seed']}: truth `{r['truth']}` at {r['bic_truth']:.1f}, "
            f"found `{r['found']}` at {r['bic_found']:.1f}"
        )
    hours = sum(r["seconds"] for r in rows) / 3600
    lines += ["", f"Search time: {hours:.1f} hours in all."]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
