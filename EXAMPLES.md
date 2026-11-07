# Examples

[En français](EXEMPLES.md)

Every output below was produced by the code in this repository on
7 November 2026. Times were measured on a laptop shared with other jobs.

## 1. Re-score a model the 2014 system published

The package ships the thirteen benchmark series of the AAAI 2014 paper and
the original search's results.

```python
import torch
from autostat import benchmarks
from autostat.model import negative_log_likelihood

print(benchmarks.names()[:3])
# ['01-airline', '02-solar', '03-mauna']

x, y = benchmarks.load("01-airline")
model = benchmarks.published("01-airline")[-1]   # the final 2014 winner
print(model.kernel.structure())
# SE + Lin + WN x Lin + SE x Per x Lin
print(round(model.nll, 3))                        # as published in 2014
# 534.518

theta = torch.as_tensor(model.theta_vector())
with torch.no_grad():
    print(round(float(negative_log_likelihood(model, torch.as_tensor(x), torch.as_tensor(y), theta)), 3))
# 534.518
```

## 2. Describe that model in English

```python
from autostat import describe

report = describe(model, x, y, name="airline passengers", out_dir="airline_report")
for summary, _, _ in report.texts:
    print(summary)
# A linearly increasing function
# An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude
# A smooth function
# Uncorrelated noise with linearly increasing standard deviation
```

`airline_report/report.md` holds the full report with five figures and the
model-checking table. It took 16 seconds, model checking included.

## 3. Search from scratch, briefly

```python
from autostat import Experiment, search

result = search(x, y, Experiment.abcd_2014(max_depth=2, n_jobs=4))
print(result.best.kernel.structure(), round(result.best.bic, 1))
# WN + Per x SE + SE 1236.4
```

Two levels already find the yearly cycle. The published ten levels reach
BIC 1123.7; this port's ten levels reach 1108.9, in 4.1 hours with six
worker processes.

## 4. Your own CSV, from the command line

```python
# make_sales.py: eight years of monthly sales, trend + yearly cycle + noise
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
months = pd.date_range("2016-01-01", periods=96, freq="MS")
k = np.arange(96)
units = 100 + 0.5 * k + 8 * np.sin(2 * np.pi * k / 12) + rng.normal(0, 2, 96)
pd.DataFrame({"month": months.strftime("%Y-%m-%d"), "units": units.round(2)}).to_csv("sales.csv", index=False)
```

```bash
autostat sales.csv --time month --value units --max-depth 3 --no-check --out sales_report
# WN + Lin x SE + Per x SE  (BIC 458.4)
#   1. A very smooth monotonically increasing function with linearly increasing marginal standard deviation
#   2. An approximately periodic function with a period of 1.0 years
#   3. Uncorrelated noise
# report: sales_report/report.md
```

Seven minutes. Progress goes to standard error; the lines above go to
standard output. Add `--json` for one machine-readable line instead.

## 5. Your own pandas series, from Python

```python
import pandas as pd
from autostat import Experiment, analyse

series = pd.read_csv("sales.csv", parse_dates=["month"]).set_index("month")["units"]
result, report = analyse(
    series,
    name="monthly sales",
    out_dir="sales_report",
    experiment=Experiment.abcd_2014(max_depth=3),
)
```

A datetime index becomes decimal years, so the report speaks of months and
years. A series longer than 1,000 points is averaged into 1,000 bins first
(`max_points=`), because the cost grows with the cube of the length.

## 6. The 2013 configuration

```python
result = search(x, y, Experiment.cks_2013(max_depth=3, n_jobs=4))
```

Add, multiply and replace only; rational quadratic among the base kernels;
noise in the likelihood. From the command line: `--preset cks-2013`.
