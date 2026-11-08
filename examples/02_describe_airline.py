"""Example 2: describe the published 2014 airline model in English.

Writes the report twice from one description: 02_airline_report/markdown/
(report.md, SVG figures, report.pdf by md2star) and
02_airline_report/latex/ (report.tex, PDF figures, report.pdf by latexmk).
The search-path section follows the ten levels the 2014 system published.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from pathlib import Path

from autostat import benchmarks, describe
from autostat.stopping import search_path

OUT = Path(__file__).parent / "02_airline_report"

x, y = benchmarks.load("01-airline")
published = benchmarks.published("01-airline")  # the winner of each 2014 level
path = search_path(
    published, x, y, "the published 2014 search: ten levels, no stopping rule"
)
report = describe(
    published[-1],
    x,
    y,
    name="airline passengers",
    out_dir=OUT / "markdown",
    path=path,
    symbolic=True,
)
report.write(OUT / "latex", "latex")
for summary, _, _ in report.texts:
    print(summary)
