# Examples

[En français](EXEMPLES.md)

Every output below was produced by the code in this repository on
7 November 2026. Times were measured on a laptop shared with other jobs.

Each example that writes a report has a script in [examples/](examples/),
and the report it wrote sits beside it twice: `markdown/` holds `report.md`
with SVG figures and `report.pdf` compiled by md2star; `latex/` holds
`report.tex` with PDF figures and `report.pdf` compiled by `latexmk`. A
report is described once and written in both formats with `report.write`.

| # | Script | Report |
|---|---|---|
| 1 | [01_rescore_published.py](examples/01_rescore_published.py) | none |
| 2 | [02_describe_airline.py](examples/02_describe_airline.py) | [markdown](examples/02_airline_report/markdown/report.md), [LaTeX](examples/02_airline_report/latex/report.tex) |
| 3 | [03_search_airline.py](examples/03_search_airline.py) | [markdown](examples/03_search_report/markdown/report.md), [LaTeX](examples/03_search_report/latex/report.tex) |
| 4 | [04_make_sales.py](examples/04_make_sales.py), [04_sales_cli.sh](examples/04_sales_cli.sh) | [markdown](examples/04_sales_report/markdown/report.md), [LaTeX](examples/04_sales_report/latex/report.tex) |
| 5 | [05_sales_pandas.py](examples/05_sales_pandas.py) | [markdown](examples/05_sales_report/markdown/report.md), [LaTeX](examples/05_sales_report/latex/report.tex) |
| 6 | [06_cks_2013.py](examples/06_cks_2013.py) | [markdown](examples/06_cks_2013_report/markdown/report.md), [LaTeX](examples/06_cks_2013_report/latex/report.tex) |
| 7 | [07_autogp_airline.py](examples/07_autogp_airline.py) | [markdown](examples/07_autogp_report/markdown/report.md), [LaTeX](examples/07_autogp_report/latex/report.tex) |
| 8 | [08_published_2014.py](examples/08_published_2014.py) | [one folder per series](examples/08_published_2014/) |

Run any script from the repository root, e.g. `python examples/02_describe_airline.py`.

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

## 2. Describe that model in English, in Markdown and in LaTeX

```python
from autostat import describe

report = describe(model, x, y, name="airline passengers", out_dir="airline_report/markdown")
report.write("airline_report/latex", "latex")
for summary, _, _ in report.texts:
    print(summary)
# A linearly increasing function
# An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude
# A smooth function
# Uncorrelated noise with linearly increasing standard deviation
```

Each folder holds the full report with its PDF: the data first, the
executive summary, the formula the search found, the search path with its
BIC curve (here the ten levels the 2014 system published), one section per
component with its figure and the model-checking table. Both together took
about a minute, model checking and the two compilations included. `describe(..., report_format="latex")` writes
the LaTeX one alone; `pdf=False` skips the compilation.

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
# examples/04_make_sales.py: eight years of monthly sales, trend + yearly cycle + noise
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
months = pd.date_range("2016-01-01", periods=96, freq="MS")
k = np.arange(96)
units = 100 + 0.5 * k + 8 * np.sin(2 * np.pi * k / 12) + rng.normal(0, 2, 96)
pd.DataFrame({"month": months.strftime("%Y-%m-%d"), "units": units.round(2)}).to_csv("sales.csv", index=False)
```

```bash
autostat sales.csv --time month --value units --max-depth 20 --patience 2 --no-check --out sales_report
# Lin + WN + Per  (BIC 449.2)
#   1. A linearly increasing function
#   2. A periodic function with a period of 1.0 years
#   3. Uncorrelated noise
# report: sales_report/report.md
# pdf: sales_report/report.pdf
```

The search stops two levels after BIC stopped falling: it finds
`Lin + WN + Per`, the formula the data were drawn from, at level 5; levels
6 and 7 find nothing better. With `--max-depth 3` it stopped at an
approximation, `WN + Lin x SE + Per x SE` (BIC 458.4). About fifteen
minutes. Progress goes to standard error; the lines above go to
standard output. `--format latex` writes `report.tex` instead, `--no-pdf`
skips the compilation and `--json` prints one machine-readable line.

## 5. Your own pandas series, from Python

```python
import pandas as pd
from autostat import Experiment, analyse

series = pd.read_csv("sales.csv", parse_dates=["month"]).set_index("month")["units"]
result, report = analyse(
    series,
    name="monthly sales",
    out_dir="sales_report",
    experiment=Experiment.abcd_2014(
        max_depth=20, stopping_criteria=["no_improvement"], patience=2
    ),
    report_format="latex",
)
```

A datetime index becomes decimal years, so the report speaks of months and
years. A series longer than 1,000 points is averaged into 1,000 bins first
(`max_points=`), because the cost grows with the cube of the length.

## 6. The 2013 configuration

```python
exp = Experiment.cks_2013(max_depth=20, n_jobs=4, stopping_criteria=["no_improvement"], patience=2)
result = search(x, y, exp)
print(result.best.kernel.structure(), round(result.best.bic, 1))
# Lin x Per + (Lin + RQ) x (Lin + Lin x Per) 1112.4
```

Add, multiply and replace only; rational quadratic among the base kernels;
noise in the likelihood. From the command line: `--preset cks-2013`. BIC
falls at every level up to the tenth, then two levels find nothing better:
twelve levels, 23 minutes. At three levels the search kept `Lin x RQ + RQ`
(BIC 1370.4), which has no yearly cycle; the cycle appears at level 4. The
2014 sentence writer has no rule for the rational quadratic, so two
components read "This simple AI is not capable of describing the component
Lin x RQ", as in the original; the formula section still writes them out.

## 7. A posterior over structures with AutoGP

```python
from autostat.smc import AutoGP, linear_schedule
from autostat.smc.report import describe_posterior

model = AutoGP(x.ravel(), y.ravel(), n_particles=8, seed=0)
model.fit_smc(linear_schedule(len(x), 0.10), n_mcmc=75, n_hmc=10)
for weight, formula in model.structures()[:3]:
    print(f"{weight:.3f}  {formula}")
# 0.543  (Lin * (((Lin * Per) + ((Lin * Lin) * (Lin + (Per + GE)))) + Lin))
# 0.276  ((Lin * (((Per + (GE * Lin)) + Lin) + Lin)) * Lin)
# 0.089  ((Lin * ((Per + (((Lin + GE) * Lin) * GE)) + Lin)) * Lin)

posterior = describe_posterior(model, name="airline passengers", out_dir="autogp_report")
```

The summary page lists the structures with their weights; each of the
heaviest has its own report, with the 2014 decomposition and checks. The
whole script took 21 minutes on eight particles, both report formats
included.

## 8. The thirteen published 2014 models

`examples/08_published_2014.py` describes the final model of each benchmark
series, from airline passengers to wages, in both formats. The solar series
shows the change windows: its components apply "until 1643 and from 1716
onwards", the years around the Maunder minimum.

## 9. The GUI

```bash
autostat-gui
# autostat GUI: http://127.0.0.1:8765/
```

The browser opens a form generated from the command line by
sprezzature-cli-gui: one field per option, defaults filled in. "Build
command" prints the equivalent command line; "Run on this machine" runs
the analysis through the local host and lists the report's files as links.
The host listens on 127.0.0.1 only.

## 10. The HTTP API

```bash
uvicorn autostat.api:app
```

```python
import httpx

api = httpx.Client(base_url="http://127.0.0.1:8000", timeout=3600)
print(api.get("/v1/series").json()[:3])
# ['01-airline', '02-solar', '03-mauna']
body = api.post("/v1/published", json={"series": "02-solar", "report_format": "latex", "check": False}).json()
print(body["structure"])
# C + CW(SE + SE + WN x Lin + WN x Lin + SE x Per, WN + C)
print(body["components"][3])
# An approximately periodic function with a period of 10.8 years. This function applies until 1643 and from 1716 onwards
pdf = api.get(f"/v1/reports/{body['report_id']}/files/report.pdf").content
```

38 seconds without checks or PDF. `POST /v1/analyses` takes `values` (3 to
1,000), optional `times`, `max_depth` (1 to 6) and the report options.
Errors share one body, e.g. `{"error": {"code": "invalid_input", "message":
"values: List should have at least 3 items after validation, not 2"}}`.
Reports go under `AUTOSTAT_REPORTS_DIR` (default `./autostat_reports`).

## 11. The MCP server

```bash
autostat-mcp            # HTTP routes and /mcp on 127.0.0.1:8000
```

Register `http://127.0.0.1:8000/mcp` in the agent host. It publishes four
tools: `list_series`, `describe_published`, `analyse_series` and
`get_report`, with the API's argument names and bounds.

## 12. The Agent Skill

Copy or link [skills/autostat](skills/autostat) into the runtime's skills
folder (`~/.claude/skills/` for Claude Code). The agent then routes "what
is in this series", « décompose cette série » or "write a LaTeX report of
this CSV" to autostat, following [TRIGGERS.md](TRIGGERS.md).

## 13. Reading the formula and the search path

Every report states the formula the search found and studies where the
search should stop. For the sales of examples 4 and 5:

```python
print(report.formula().explicit)
# ['f_{1}(t) = 99.63 + 6.095\\,(t - 2016)', None,
#  'f_{3}(t) \\approx 7.95\\cos\\!\\left(\\frac{2\\pi \\cdot (t - 2016)}{1} - 1.807\\right) + ...']
for sentence in findings(report.path, report.model.ndata):
    print(sentence)
# The search ran 7 levels (stopping after 2 level(s) without a BIC fall above 0.1, at most 20) ...
# BIC stopped falling at level 5. The 2 level(s) searched past it kept its structure ...
```

`findings` comes from `autostat.stopping`. The data were drawn from a slope
of 6 per year from 100, a sine of amplitude 8 and period one year plus a
noise of variance 4; the formula reads 6.095, 7.95, one year and 3.578.
