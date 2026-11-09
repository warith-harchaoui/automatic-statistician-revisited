<p align="center"><img src="assets/logo-288.png" width="200" alt="A statistician at a desk, finger raised"></p>

# Automatic Statistician, revisited

[Lire en français](LISEZMOI.md)

Give it a noisy curve. It finds the formula behind it: a trend, cycles,
smooth variations and noise, each with its fitted values. It describes each
part in an English sentence, checks each one against the data and writes the
whole analysis up as a report, in Markdown or in LaTeX.

Here is what it says about twelve years of monthly airline passengers, from
1949 to 1960:

1. A linearly increasing function, $f_1(t) = 92.68 + 31.58\,(t - 1949)$, in
   thousands of passengers a month.
2. An approximately periodic function with a period of 1.0 years and with
   linearly increasing amplitude.
3. A smooth function.
4. Uncorrelated noise with linearly increasing standard deviation.

These four sentences are word for word the ones the Automatic Statistician
published for the same series in 2014. This repository is that system, by
Duvenaud, Lloyd, Grosse, Tenenbaum and Ghahramani (ICML 2013, AAAI 2014),
rewritten from its Python 2.7 and MATLAB original into Python 3 and `torch`.
The original code received its last commits in 2015 and 2017. It needs
MATLAB, the GPML toolbox (Gaussian Processes for Machine Learning) and a
Python that reached its end of life in 2020. This one installs with `pip` and
reproduces the likelihood of every model the 2014 system published.

```python
from autostat import analyse

result, report = analyse(series, name="monthly sales", out_dir="report/")
print(result.best.kernel.structure())
# e.g. Lin + WN + Per
```

```bash
autostat sales.csv --time month --value units --out report/
```

The report opens on the data, then gives:

- an executive summary, with one sentence per component;
- the formula found, with every fitted parameter and a closed form for each
  component that has one;
- the search path: the BIC curve, the structure kept at every level and what
  stopping earlier would have kept;
- one section per component, with its figure;
- the model checks.

`report/report.pdf` is the same document, compiled by
[md2star](https://github.com/warith-harchaoui/md2star). With `--format latex`
(in Python, `report_format="latex"`) the report becomes `report.tex`,
compiled by `latexmk`, as the 2014 system wrote it. The figures come from
[sprezzature-figures](https://github.com/warith-harchaoui/sprezzature-figures).
[EXAMPLES.md](EXAMPLES.md) walks through every example.
[examples/](examples/) holds each script with the report it wrote, in both
formats.

## What the search does

A kernel is a formula for how strongly two points of a series move together.
Two neighbouring months look alike under a smooth kernel. Two Julys look alike
under a periodic one. Under noise, two measurements owe each other nothing.
The search builds its formulas from a few base kernels:

| base kernel | what it describes |
|---|---|
| `SE`, squared exponential | a smooth function |
| `Per`, periodic | a pattern that repeats |
| `Lin`, linear | a trend, or an amplitude that grows |
| `C`, constant | an offset |
| `WN`, white noise | the measurement noise |
| `RQ`, rational quadratic | smooth variation at several scales (2013 preset only) |

It combines them with `+` (two effects added together), `x` (one effect
modulating another), `CP` (a changepoint: one structure, then another) and
`CW` (a change window: a structure that holds for a while). The search starts
from noise alone and climbs level by level. At each level it rewrites the best
formula of the level below in every way the grammar allows, fits each
candidate from nine random starts and keeps the one with the lowest Bayesian
information criterion (BIC). BIC adds the misfit to a price for every
parameter tuned, so a formula that hugs the data through many parameters
loses to a simpler one that fits almost as well.

Machine learning calls this symbolic regression: the search returns a formula
where other methods return weights. Eureqa and PySR assemble arithmetic. This
search assembles kernels, so the noise and the uncertainty belong to the
formula instead of being what it leaves over. The
[technical report](doc/markdown/report.md) sets the two side by side.

## Where the search should stop

The 2014 system always searched ten levels. Each report now studies, for its
own series, whether that was too many or too few. It plots the BIC of the best
model at every level and marks the level where BIC stopped falling. Run with
`--patience 2`, the search stops once BIC has not fallen for two levels in a
row, so the report shows what those two extra levels found.

The report also weighs the structures against one another, after the Bayesian
machine scientist of Guimerà and colleagues (Science Advances, 2020): each
structure met gets a posterior weight proportional to `exp(-BIC/2)`. On the
published call-centre series the structure kept weighs 0.713 and the next one
0.281. The data barely tell them apart: the report says so.

## How we know the port is faithful

**Every published model reproduces.** The original's result files for its
thirteen benchmark series ship with this package. For each series and each of
the ten search levels, `tests/test_published.py` takes the published winning
model, evaluates its negative log likelihood (NLL) at the printed
hyperparameters and compares it with the published number. All 130 agree (thirteen series times ten levels):
twelve series to within 5e-5 nats, the solar series to within 5e-3 nats. Its
covariance matrices reach a condition number of 1e14, which explains the
looser bound.

That test caught one real defect during the port. A small diagonal jitter,
relative to the mean diagonal, moved the solar likelihood by 61 nats, because
a constant kernel dominated that diagonal. The first factorisation now adds no
jitter, as GPML did.

**The text matches.** For the published airline model the four component
sentences are word for word those of the 2014 system.

**The gradients match.** Autograd replaces GPML's hand-written derivatives,
and a finite-difference test holds them to 1e-4.

**The search finds structures at least as good.** The search draws random
starts and uses another optimiser (L-BFGS instead of GPML's conjugate
gradients), so it does not retrace the 2014 search digit for digit.
`scripts/replicate.py` reruns the full ten-level search under the published
settings and compares the final BIC (lower is better):

| series | published BIC | this port | difference |
|---|---|---|---|
| 01-airline | 1123.7 | 1108.9 | -14.8 |
| 02-solar | -870.3 | -872.3 | -2.0 |

The other eleven series have not been rerun yet. At about four hours a series
on a laptop, they wait for a free machine.

**What has no published yardstick.** The 2013 preset and the model-checking
p-values: the original repositories hold no numbers to compare them with.

## AutoGP in Python: a posterior over structures

`autostat.smc` ports AutoGP.jl (Saad, Patton, Hoffman, Saurous and
Mansinghka, ICML 2023) to Python and `torch`. The 2013 and 2014 searches keep
the one structure with the best BIC. AutoGP keeps a weighted sample of
structures from their posterior, found by sequential Monte Carlo (SMC). The
data enter a tenth at a time. Particles are reweighted by how well they
predicted each batch, resampled, then moved by structure moves (subtree
replace, detach-attach) and by Hamiltonian Monte Carlo (HMC) on the
parameters. Forecasts average the particles, so the intervals widen where the
data cannot tell structures apart. New observations join later without a
refit.

```python
from autostat.smc import AutoGP, linear_schedule
from autostat.smc.report import describe_posterior

model = AutoGP(t, y, n_particles=8, seed=0)
model.fit_smc(linear_schedule(len(t), 0.10), n_mcmc=75, n_hmc=10)
model.structures()[:3]          # [(weight, formula), ...]
mean, bounds = model.predict(t_future, quantiles=(0.05, 0.95))
describe_posterior(model, name="airline", out_dir="report/")
```

`describe_posterior` translates each heavy structure into the 2014 kernels
and writes one report per structure, plus a summary page with the weights and
a figure of each structure's BIC.

**Checked against the original, number for number.** `tests/test_smc_julia.py`
loads AutoGP.jl v0.1.19 through `juliacall` and compares every deterministic
quantity on the airline series:

| quantity | cases | largest relative difference |
|---|---|---|
| joint log density of a trace (prior, noise, likelihood) | 30 structures drawn by Julia | 1e-14 |
| covariance matrix | the same 30 | 1e-15 |
| forward and reverse densities of a structure move | 400 moves drawn by Julia (218 replace, 165 attach, 17 detach) | 1e-15 |
| model density ratio of the move | the same 400 | 4e-13 |
| predictive mean and covariance | 20 structures, 15 future points | 3e-12 |

**What the comparison found in AutoGP.jl.** Gen's gradient for every kernel
parameter equals `-z`, the gradient of its standard normal prior, on all 359
parameters tested. `covariance_prior` is not declared differentiable
(`Gen.accepts_output_grad` is false), so the likelihood never reaches the
leapfrog steps. The acceptance test uses the full density, so the sampler
stays correct, but its HMC moves kernel parameters as a random walk driven by
the prior. The port reproduces this by default (`HMCConfig(gradient="prior")`)
and offers the intended behaviour as `gradient="full"`. The noise gets the
full gradient in both, as in Julia.

**Compared in distribution.** Random draws differ between the two languages,
so whole runs agree only statistically. `scripts/compare_autogp.py` fits
AutoGP.jl, the port and the port with full gradients on the same benchmark
series, several seeds each, then scores all three with the same code. Over 87
runs no difference was significant. The port ran about six times faster.

## Forecasting against the methods of 2026

The 2014 paper also judged its system on extrapolation: train on the first
90 % of each of the thirteen series, forecast the last 10 %. The
[benchmark](benchmark/README.md) folder reruns that test. It first recomputes
the paper's table from the published 2014 forecasts, to within 2e-10, then
adds the methods of 2026 family by family:

- classical models: ETS, ARIMA, Theta;
- gradient boosting: LightGBM;
- deep networks: N-HiTS, N-BEATS, PatchTST;
- foundation models: Chronos-2, Chronos-Bolt, TimesFM 2.5, TabPFN-TS, NVIDIA
  KumoTabular;
- symbolic regression: Eureqa (the published 2014 forecasts), PySR, gplearn;
- this repository's own ports.

Methods with free hyperparameters are tuned by rolling-origin
cross-validation on the training part only, with skrub and skore. The 2014
forecasts finish fifth of twenty-four methods and never collapse. Eureqa
misses one series 590 times worse than the best method. PySR, on the same
split, stays bounded on that series (4.56). The table is in
[benchmark/RESULTS.md](benchmark/RESULTS.md).

## The Automatic Statistician family

Every project we found, from the original papers to their successors of 2026,
with what this repository takes from each. Each repository was checked on
GitHub on 7 November 2026.

| project | year | what it is | here |
|---|---|---|---|
| [gp-structure-search](https://github.com/jamesrobertlloyd/gp-structure-search) | 2013 | Compositional kernel search (CKS), ICML 2013. MATLAB and Python 2.7, MIT licence. | the `Experiment.cks_2013()` preset |
| [gpss-research](https://github.com/jamesrobertlloyd/gpss-research) | 2014 | Automatic Bayesian Covariance Discovery (ABCD), AAAI 2014: changepoints, reports in natural language. MIT licence. | ported in full |
| [kmc-research](https://github.com/jamesrobertlloyd/kmc-research) | 2015 | Model criticism with the maximum mean discrepancy (MMD), NIPS 2015. MIT licence. | the model-checking section |
| [comp-kern](https://github.com/jamesrobertlloyd/comp-kern) | 2014 | "Simplified interface to compositional kernel learning". No licence. | read, not copied |
| [auto-stat-basic](https://github.com/jamesrobertlloyd/auto-stat-basic) | 2015 | "First attempts at automatic statistician architecture". No licence. | read, not copied |
| [autogpc](https://github.com/charles92/autogpc) | 2016 | Kernel search for GP classification (Qiurui He's thesis). No licence. | cited |
| [Relational Automatic Statistician](https://github.com/OpenXAIProject/Relational-Automatic-Statistician) | 2019 | Structure shared across several series (Hwang, Tong and Choi, ICML 2016). Apache 2.0. | cited |
| [autostat (sutoiku)](https://github.com/sutoiku/autostat) | 2022 | An ABCD reimplementation in Mathematica. MIT licence. | cited |
| [autostat_gpflow](https://github.com/DavidJanz/autostat_gpflow) | 2018 | A basic GPflow implementation. MIT licence. | cited |
| [GPy-ABCD](https://github.com/T-Flet/GPy-ABCD) | 2021 | Configurable ABCD on GPy (Fletcher, Bundy and Nuamah, ICML 2021 AutoML workshop). BSD 3-clause. | cited |
| [AutoGP.jl](https://github.com/probsys/AutoGP.jl) | 2023 | Sequential Monte Carlo (SMC) over kernel structures (Saad et al., ICML 2023, arXiv 2307.09607). Julia, Apache 2.0, maintained. | ported as `autostat.smc`, checked against it through `juliacall` |
| [gallifrey](https://github.com/ChrisBoettner/gallifrey) | 2025 | A JAX re-implementation of AutoGP.jl for astronomy (Boettner, A&A 2025, arXiv 2505.20394). Apache 2.0. | cited |
| [AutoBNN](https://github.com/tensorflow/probability/tree/main/spinoffs/autobnn) | 2024 | Compositional Bayesian neural networks in JAX (Google). Apache 2.0. | cited |

Three related lines of work have no public code we could find:

- BoxLM, where a large language model (LLM) proposes probabilistic programs
  (Li, Fox and Goodman, ICML 2024, arXiv 2402.17879);
- a pipeline in which a vision-language model judges kernel fits by their
  plots (Lee et al., arXiv 2509.25946, 2025);
- scalable kernel composition (Kim and Teh, AISTATS 2018, arXiv 1706.02524).

The address automaticstatistician.com now serves a site unrelated to the
project. The original, online from July 2014, survives in web archives.

## Install

Python 3.11 or later. The search, the report sources and their figures need
Python wheels only. Compiling the PDF needs, for a Markdown report,
[Pandoc](https://pandoc.org/installing.html) and
[LibreOffice](https://www.libreoffice.org/download/) (`md2star doctor` checks
both). For a LaTeX report it needs a TeX distribution with `latexmk` (MacTeX,
TeX Live or MiKTeX). Without them the source is written and a warning names
the missing tool. `--no-pdf` skips the step.

```bash
pip install git+https://github.com/warith-harchaoui/automatic-statistician-revisited
# or, from a clone, for development:
pip install -r requirements-dev.txt
make check
```

Reading Parquet files needs `pip install "automatic-statistician-revisited[parquet]"`.
Closed forms for smooth components need
`pip install "automatic-statistician-revisited[symbolic]"`, which installs PySR
and, through it, Julia. On Windows without `make`, run the four commands of
the `Makefile` by hand.

## Usage

The same analysis opens six ways. Each one calls `autostat.service`, so all
six check their inputs and behave alike.

| surface | start it with | for |
|---|---|---|
| Python library | `from autostat import analyse` | notebooks and other code |
| Command line (Click) | `autostat data.csv ...` | a shell or a script |
| GUI (sprezzature-cli-gui) | `autostat-gui` | a form in the browser, on this machine only |
| HTTP API (FastAPI) | `uvicorn autostat.api:app` | another program, with OpenAPI at `/docs` |
| MCP server (FastAPI-MCP) | `autostat-mcp` | an agent host, at the endpoint `/mcp` |
| Agent skill | [skills/autostat/SKILL.md](skills/autostat/SKILL.md) | Claude Code or OpenCode, routed by [TRIGGERS.md](TRIGGERS.md) |

The command line reads a CSV or Parquet file:

```bash
autostat data.csv --time date --value y --out report/
autostat data.csv --time date --value y --format latex --out report/
autostat data.csv --time date --value y --preset cks-2013 --max-depth 4
autostat data.csv --time date --value y --max-depth 20 --patience 2
autostat data.csv --time date --value y --symbolic --out report/
autostat data.csv --time date --value y --no-check --no-pdf --json
```

The published configuration searches `--max-depth` levels and no fewer.
`--patience 2` stops once BIC has not fallen for two levels in a row.
`--symbolic` asks PySR for a formula for each smooth component, about a
minute each.

The Python interface exposes each stage: `prepare` (dates to decimal years,
long series binned), `search` with an `Experiment`, `describe` for the report,
`fit` and `posterior` for a single model.

The API and the MCP server share one application: `autostat-mcp` serves the
HTTP routes and `/mcp` on port 8000. The MCP tools are `list_series`,
`describe_published`, `analyse_series` and `get_report`. The API bounds a
search to 1,000 points and to six levels, twenty with a patience. Both bind
to 127.0.0.1 by default and have no authentication: put one in front of them
before exposing them.

## Limits

- **Size.** Exact GP inference costs time cubic in the number of points.
  `prepare` averages any series longer than 1,000 points into 1,000 bins, the
  bound the original used, so a structure finer than one bin goes unreported.
  The full ten-level search on the 144 points of the airline series took 4.1
  hours with six worker processes on a shared laptop. `--max-depth 3` gives a
  first look in minutes.
- **One input dimension in the report.** Kernels and search accept several
  input dimensions, but the English description assumes one.
- **Regression only.** The classification variant (autogpc) is not ported.
- **The 2013 preset approximates.** It uses the 2013 grammar and settings with
  the parameter initialisation of 2014. Its two random starts with a spread of
  4 often miss a period: on the trend-plus-cycle series of
  `tests/test_search.py` it stops at `SE` where the 2014 preset finds the
  cycle.
- **AutoGP's grammar is its own.** `autostat.smc` keeps the base kernels of
  AutoGP.jl (linear, gamma-exponential, periodic) and its changepoint, which
  differ from the 2014 grammar. Its reports translate each structure into the
  2014 kernels.
- **Model checking is a screen.** A small p-value says the data disagree with
  a component's prior on one statistic. It does not say which model is right.

## Citing

Cite the original work, then this port if you use it:

```bibtex
@inproceedings{duvenaud2013structure,
  title     = {Structure Discovery in Nonparametric Regression through Compositional Kernel Search},
  author    = {Duvenaud, David and Lloyd, James Robert and Grosse, Roger and Tenenbaum, Joshua B. and Ghahramani, Zoubin},
  booktitle = {Proceedings of the 30th International Conference on Machine Learning},
  year      = {2013},
  eprint    = {1302.4922},
  archivePrefix = {arXiv}
}

@inproceedings{lloyd2014automatic,
  title     = {Automatic Construction and Natural-Language Description of Nonparametric Regression Models},
  author    = {Lloyd, James Robert and Duvenaud, David and Grosse, Roger and Tenenbaum, Joshua B. and Ghahramani, Zoubin},
  booktitle = {Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence},
  year      = {2014},
  eprint    = {1402.4304},
  archivePrefix = {arXiv}
}

@inproceedings{lloyd2015statistical,
  title     = {Statistical Model Criticism using Kernel Two Sample Tests},
  author    = {Lloyd, James Robert and Ghahramani, Zoubin},
  booktitle = {Advances in Neural Information Processing Systems 28},
  year      = {2015}
}
```

[CITATION.cff](CITATION.cff) describes this repository for citation tools.

## Licence and credit

The software is by Warith Harchaoui, under the MIT licence
([LICENSE](LICENSE)). The design ported here, the benchmark data and the
published results are by James Robert Lloyd, David Duvenaud and Roger Grosse,
under the MIT licence reproduced in [LICENSE-ORIGINAL](LICENSE-ORIGINAL).
`autostat.smc` ports AutoGP.jl, (c) 2023 Google LLC, under the Apache 2.0
licence reproduced in [LICENSE-AUTOGP](LICENSE-AUTOGP). Most benchmark series
come from Rob Hyndman's Time Series Data Library. [PORTING.md](PORTING.md)
lists what changed in the port and the defects found in the original along
the way. The logo and its variants are in [assets/](assets/).

The reports the software writes are signed "Some Math empowered by some AI
guided by some researchers".
