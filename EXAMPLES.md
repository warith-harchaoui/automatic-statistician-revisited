# Examples

[En français](EXEMPLES.md)

Every output below comes from the code in this repository, run between 7 and
9 November 2026 on a laptop shared with other jobs. The times are those of
that laptop.

The same analysis opens six ways:

| surface | start it with | examples |
|---|---|---|
| Python library | `from autostat import analyse` | sections 1 to 8 and 13 |
| Command line (Click) | `autostat data.csv ...` | section 4 |
| GUI (sprezzature-cli-gui) | `autostat-gui` | section 9 |
| HTTP API (FastAPI) | `uvicorn autostat.api:app` | section 10 |
| MCP server (FastAPI-MCP) | `autostat-mcp` | section 11 |
| Agent skill | `skills/autostat` with `TRIGGERS.md` | section 12 |

Each example that writes a report has a script in [examples/](examples/). The
report sits beside it twice. `markdown/` holds `report.md` with SVG figures
and `report.pdf` compiled by md2star. `latex/` holds `report.tex` with PDF
figures and `report.pdf` compiled by `latexmk`. A report is described once,
then written in both formats with `report.write`.

| # | script | report |
|---|---|---|
| 1 | [01_rescore_published.py](examples/01_rescore_published.py) | none |
| 2 | [02_describe_airline.py](examples/02_describe_airline.py) | [markdown](examples/02_airline_report/markdown/report.md), [LaTeX](examples/02_airline_report/latex/report.tex) |
| 3 | [03_search_airline.py](examples/03_search_airline.py) | [markdown](examples/03_search_report/markdown/report.md), [LaTeX](examples/03_search_report/latex/report.tex) |
| 4 | [04_make_sales.py](examples/04_make_sales.py), [04_sales_cli.sh](examples/04_sales_cli.sh) | [markdown](examples/04_sales_report/markdown/report.md), [LaTeX](examples/04_sales_report/latex/report.tex) |
| 5 | [05_sales_pandas.py](examples/05_sales_pandas.py) | [markdown](examples/05_sales_report/markdown/report.md), [LaTeX](examples/05_sales_report/latex/report.tex) |
| 6 | [06_cks_2013.py](examples/06_cks_2013.py) | [markdown](examples/06_cks_2013_report/markdown/report.md), [LaTeX](examples/06_cks_2013_report/latex/report.tex) |
| 7 | [07_autogp_airline.py](examples/07_autogp_airline.py) | [markdown](examples/07_autogp_report/markdown/report.md), [LaTeX](examples/07_autogp_report/latex/report.tex) |
| 8 | [08_published_2014.py](examples/08_published_2014.py) | [one folder per series](examples/08_published_2014/) |

Run any script from the repository root, for instance
`python examples/02_describe_airline.py`.

## 1. Re-score a model the 2014 system published

The package ships the thirteen benchmark series of the AAAI 2014 paper with
the results of the original search. Here is its final model for the airline
passengers, scored again:

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

The two numbers agree to the last printed digit, twelve years apart and in
another language.

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

Each folder holds the full report with its PDF. It opens on the data, then
gives the executive summary, the formula the search found, the search path
with its BIC curve (here the ten levels the 2014 system published), one
section per component with its figure and the model checks. Both formats took
about a minute, model checks and the two compilations included.
`describe(..., report_format="latex")` writes the LaTeX one alone.
`pdf=False` skips the compilation.

## 3. Search from scratch

```python
from autostat import Experiment, search

result = search(x, y, Experiment.abcd_2014(max_depth=10, n_jobs=6))
print(result.best.kernel.structure(), round(result.best.bic, 1))
# Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE 1108.9
```

Ten levels on 144 points took 4.1 hours with six worker processes. The
published search had reached a BIC of 1123.7, so the port does 14.8 better.
Stopped after two levels, the search keeps `WN + Per x SE + SE` (BIC 1236.4):
the yearly cycle without the growth.
`Experiment.abcd_2014(max_depth=20, stopping_criteria=["no_improvement"], patience=2)`
searches until BIC has not fallen for two levels, as
[03_search_airline.py](examples/03_search_airline.py) does.

## 4. Your own CSV, from the command line

Eight years of monthly sales, drawn from a known formula: a trend, a yearly
cycle and noise.

```python
# examples/04_make_sales.py
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

At level 5 the search finds `Lin + WN + Per`, the very formula the data were
drawn from. Levels 6 and 7 find nothing better, so the search stops there.
With `--max-depth 3` it had stopped on an approximation,
`WN + Lin x SE + Per x SE` (BIC 458.4). The run takes about fifteen minutes.
Progress goes to standard error, the lines above to standard output.
`--format latex` writes `report.tex` instead. `--no-pdf` skips the
compilation. `--json` prints one line for another program to read.

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
years. A series longer than 1,000 points is first averaged into 1,000 bins
(`max_points=`), since the cost grows with the cube of the length.

## 6. The 2013 configuration

```python
exp = Experiment.cks_2013(max_depth=20, n_jobs=4, stopping_criteria=["no_improvement"], patience=2)
result = search(x, y, exp)
print(result.best.kernel.structure(), round(result.best.bic, 1))
# Lin x Per + (Lin + RQ) x (Lin + Lin x Per) 1112.4
```

The 2013 grammar adds, multiplies and replaces. It has the rational quadratic
among its base kernels and keeps the noise in the likelihood. From the
command line, the same preset is `--preset cks-2013`. BIC falls at every
level up to the tenth, then two levels find nothing better: twelve levels in
23 minutes. Stopped at three levels, the search kept `Lin x RQ + RQ`
(BIC 1370.4), which has no yearly cycle. The cycle appears at level 4.

The sentence writer of 2014 has no rule for the rational quadratic. Two
components therefore read "This simple AI is not capable of describing the
component Lin x RQ", as in the original. The formula section still writes
them out.

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

The summary page lists the structures with their weights. AutoGP has no
search levels, hence no BIC curve. The page draws instead each structure's
BIC above the lowest and says whether BIC ranks the structures as the
posterior does. Each of the heaviest structures gets its own report, with the
decomposition and the checks of 2014. The whole script took 21 minutes on
eight particles, both report formats included.

## 8. The thirteen published 2014 models

`examples/08_published_2014.py` describes the final model of each benchmark
series, from airline passengers to wages, in both formats. The solar series
shows what a change window does. Its components apply "until 1643 and from
1716 onwards": the years around the Maunder minimum, when the Sun's spots
nearly vanished.

## 9. The GUI

```bash
autostat-gui
# autostat GUI: http://127.0.0.1:8765/
```

The browser opens a form that sprezzature-cli-gui generated from the command
line: one field per option, defaults filled in. "Build command" prints the
matching command line. "Run on this machine" runs the analysis through the
local host and lists the report's files as links. The host listens on
127.0.0.1 only. The page loads nothing from another host: its first build
downloads the Tailwind command line (about 45 MB), then compiles only the
classes the page uses.

## 10. The HTTP API

```bash
uvicorn autostat.api:app
```

```python
import httpx

api = httpx.Client(base_url="http://127.0.0.1:8000", timeout=3600)
print(api.get("/v1/series").json()[:3])
# ['01-airline', '02-solar', '03-mauna']
body = api.post("/v1/published", json={"series": "12-births", "check": False, "pdf": False}).json()
print(sorted(body))
# ['bic', 'components', 'files', 'formula', 'report_format', 'report_id', 'search_path', 'structure']
print(body["structure"], round(body["bic"], 1))
# WN + SE + Per + Per 9398.3
print(body["search_path"]["settled_level"])
# 5
print(body["formula"]["explicit"][1][:70])
# f_{2}(t) \approx 38.53\cos\!\left(\frac{2\pi \cdot (t - 1977)}{0.01917}
```

The response holds the formula as LaTeX strings and the search level by
level. The births series has a weekly cycle: its period, 0.01917 years, is
seven days. `POST /v1/analyses` takes `values` (3 to 1,000), optional
`times`, `max_depth` (1 to 6, up to 20 with a `patience`), `patience`,
`symbolic` and the report options. Errors share one body, for instance
`{"error": {"code": "invalid_input", "message": "values: List should have at
least 3 items after validation, not 2"}}`. Reports go under
`AUTOSTAT_REPORTS_DIR` (by default `./autostat_reports`).

## 11. The MCP server

```bash
autostat-mcp            # HTTP routes and /mcp on 127.0.0.1:8000
claude mcp add --transport http autostat http://127.0.0.1:8000/mcp
```

A session over the protocol, taken from `tests/test_mcp.py`:

```text
tools/list   list_series, describe_published, analyse_series, get_report
tools/call   list_series -> ["01-airline", "02-solar", "03-mauna", ...]
tools/call   get_report {"report_id": "x"} -> isError: true,
             {"error":{"code":"not_found","message":"no report 'x'"}}
```

Four tools are published, with the argument names and bounds of the API.
`health` and the binary download stay out of the agent's reach.

## 12. The agent skill

```bash
python scripts/build_skill.py
# wrote dist/autostat-skill.zip
cp -r skills/autostat ~/.claude/skills/      # Claude Code
```

In Claude, the zip goes under Settings > Capabilities > Skills. The agent
then sends requests such as "what is in this series", "fit a formula to this
noisy curve" or « décompose cette série » to autostat, as
[TRIGGERS.md](TRIGGERS.md) lays out. The skill checks its own work:

```bash
python skills/autostat/scripts/check_report.py examples/02_airline_report/markdown
# report OK
```

[evals/triggers.yaml](skills/autostat/evals/triggers.yaml) lists 14 requests
that load the skill and 7 that leave it alone, for the triggering test of
Anthropic's guide to skills. `tests/test_skill.py` holds the skill to the
guide's rules.

## 13. Reading the formula and the search path

Every report states the formula the search found and studies where the search
should stop. On the sales of examples 4 and 5:

```python
import pandas as pd
from autostat import Experiment, analyse, findings

series = pd.read_csv("sales.csv", parse_dates=["month"]).set_index("month")["units"]
experiment = Experiment.abcd_2014(max_depth=20, stopping_criteria=["no_improvement"], patience=2)
result, report = analyse(series, name="monthly sales", out_dir="sales_report", experiment=experiment)

for closed_form, note in zip(report.formula().explicit, report.formula().notes):
    print(closed_form, "|", note)
# f_{1}(t) = 99.63 + 6.095\,(t - 2016) | exact polynomial of degree 1
# f_{2}(t) \approx 7.95\cos\!\left(\frac{2\pi \cdot (t - 2016)}{1} - 1.807\right) + 0.3961\cos\!\left(...\right) | Fourier series, 2 harmonics, largest gap 0.027 % of the component
# None | None

for sentence in findings(report.path, report.model.ndata):
    print(sentence)
# The search ran 7 levels (stopping after 2 level(s) without a BIC fall above 0.1, at most 20) and keeps the model of level 6, Lin + WN + Per, BIC 449.2.
# That structure was first reached at level 5; the levels after it only refitted its parameters, lowering BIC by 0.0 in all.
# The best different structure, Lin + WN + Per x SE (level 4), is 4.6 above it: positive evidence for the kept one on the Kass and Raftery scale.
# Weighed by exp(-BIC/2) under a flat prior, as the Bayesian machine scientist weighs expressions, the structures met give the kept one a posterior weight of 0.899 and the next, Lin + WN + Per x SE, 0.092: the data favour the kept one without ruling the next out.
# BIC stopped falling at level 5. The 2 level(s) searched past it kept its structure and lowered BIC by 0.00 in all, less than the tolerance of 0.1: going on buys nothing the data pay for.
# A rule stopping after 1 level(s) without a fall above 0.1 would have ended the search at level 6 and kept the same model.
# A rule stopping after 2 level(s) without a fall above 0.1 would have ended the search at level 7 and kept the same model.
# The kept expression has 6 parameters for 96 points: its BIC is a fit term 2 NLL = 421.9 plus a complexity term p ln n = 27.4, each extra parameter costing 4.56; it splits into 3 components, 2 of them with a closed form.
```

The data were drawn from a slope of 6 a year starting at 100, a sine of
amplitude 8 and period one year, plus a noise of variance 4. The formula reads
6.095 for the slope, 7.95 for the amplitude, one year for the period and 3.578
for the noise variance. The second harmonic, 5 % of the first, is
what the search adds of its own: a part of it comes from months that decimal
years do not space evenly.
