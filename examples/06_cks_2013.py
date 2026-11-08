"""Example 6: the 2013 configuration on the airline series, two levels past its best.

The search stops after two levels without a BIC fall above 0.1, at most
twenty. Writes 06_cks_2013_report/markdown/ and 06_cks_2013_report/latex/,
and the search itself as search.json. The yearly cycle appears at the fourth
level.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from pathlib import Path

from autostat import Experiment, benchmarks, describe, search
from autostat.stopping import rule_of, search_path

OUT = Path(__file__).parent / "06_cks_2013_report"

x, y = benchmarks.load("01-airline")
experiment = Experiment.cks_2013(
    max_depth=20, n_jobs=4, stopping_criteria=["no_improvement"], patience=2
)
result = search(x, y, experiment)
print(result.best.kernel.structure(), round(result.best.bic, 1))
path = search_path(result.best_per_level(), x, y, rule_of(result.experiment))
report = describe(
    result.best, x, y, name="airline passengers", out_dir=OUT / "markdown", path=path
)
report.write(OUT / "latex", "latex")
result.save(OUT / "search.json")  # the search, to re-describe it without rerunning
