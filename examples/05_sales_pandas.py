"""Example 5: the same sales as a pandas series, from Python, model checks included.

The search recovers Lin + WN + Per, the formula the data were generated with,
and runs two levels past it to show that going on finds nothing better.

Writes 05_sales_report/markdown/ and 05_sales_report/latex/.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from pathlib import Path

import pandas as pd

from autostat import Experiment, analyse

OUT = Path(__file__).parent / "05_sales_report"

sales = pd.read_csv(Path(__file__).parent / "sales.csv", parse_dates=["month"])
series = sales.set_index("month")["units"]
result, report = analyse(
    series,
    name="monthly sales",
    out_dir=OUT / "markdown",
    symbolic=True,
    experiment=Experiment.abcd_2014(
        max_depth=20, stopping_criteria=["no_improvement"], patience=2
    ),
)
report.write(OUT / "latex", "latex")
print(result.best.kernel.structure(), round(result.best.bic, 1))
for summary, _, _ in report.texts:
    print(summary)
