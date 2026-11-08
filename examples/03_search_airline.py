"""Example 3: search from scratch until BIC stops falling, then two levels more.

The search stops after two levels without a BIC fall above 0.1, at most
twenty: the two extra levels show what going on would have found. Writes
03_search_report/markdown/ and 03_search_report/latex/, plus the search itself
as search.json. About four and a half hours on four worker processes.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from pathlib import Path

from autostat import Experiment, benchmarks, describe, search
from autostat.stopping import rule_of, search_path

OUT = Path(__file__).parent / "03_search_report"

x, y = benchmarks.load("01-airline")
experiment = Experiment.abcd_2014(
    max_depth=20, n_jobs=4, stopping_criteria=["no_improvement"], patience=2
)
result = search(x, y, experiment)
print(result.best.kernel.structure(), round(result.best.bic, 1))
path = search_path(result.best_per_level(), x, y, rule_of(result.experiment))
report = describe(
    result.best,
    x,
    y,
    name="airline passengers",
    out_dir=OUT / "markdown",
    path=path,
    symbolic=True,
)
report.write(OUT / "latex", "latex")
result.save(OUT / "search.json")  # the search, to re-describe it without rerunning
