"""Example 8: an English report for each of the thirteen published 2014 models.

Each report's search-path section follows the ten levels the 2014 system
published for that series. Writes 08_published_2014/<series>/markdown/ and
.../latex/.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from pathlib import Path

from autostat import benchmarks, describe
from autostat.stopping import search_path

OUT = Path(__file__).parent / "08_published_2014"
RULE = "the published 2014 search: ten levels, no stopping rule"

for name in benchmarks.names():
    x, y = benchmarks.load(name)
    published = benchmarks.published(name)
    path = search_path(published, x, y, RULE)
    report = describe(
        published[-1],
        x,
        y,
        name=name,
        out_dir=OUT / name / "markdown",
        path=path,
        symbolic=True,
    )
    report.write(OUT / name / "latex", "latex")
    print(name, published[-1].kernel.structure())
