# Automatic Statistician, revisited

[Lire en français](LISEZMOI.md)

Give it a time series. It searches a grammar of Gaussian process (GP) kernels
for the structure that explains the series best, splits that structure into
additive components, describes each component in English, checks each one
against its own assumptions and writes the whole thing up as a report.

This is the Automatic Statistician of Duvenaud, Lloyd, Grosse, Tenenbaum and
Ghahramani (ICML 2013, AAAI 2014), ported from its Python 2.7 and MATLAB
original to Python 3 and `torch`. The original code last changed in 2015 and
2017 and needs MATLAB, the GPML (Gaussian Processes for Machine Learning)
toolbox and a Python that reached end of life in 2020. This repository runs
with `pip install` and reproduces the likelihood of every model the 2014
system published.

```python
from autostat import analyse

result, report = analyse(series, name="monthly sales", out_dir="report/")
print(result.best.kernel.structure())
# e.g. WN + SE + Lin x Per x SE
```

```bash
autostat sales.csv --time month --value units --out report/
```

`report/report.md` then holds the executive summary, one section per
component with its figure, then the model-checking table. The airline example
in [EXAMPLES.md](EXAMPLES.md) shows the output.

## What the search does

A kernel is a formula for how strongly two points of a series move together.
The search builds kernels out of a few base kernels:

| base kernel | it describes |
|---|---|
| `SE`, squared exponential | a smooth function |
| `Per`, periodic | a repeating pattern |
| `Lin`, linear | a trend, or an amplitude that grows |
| `C`, constant | an offset |
| `WN`, white noise | measurement noise |
| `RQ`, rational quadratic | smooth variation at several scales (2013 preset) |

It combines them with `+` (two effects added), `x` (one effect modulating
another), `CP` (a changepoint: one structure, then another) and `CW` (a change
window: a structure that holds for a while). At each level it tries every
expression one rewrite away from the current best, fits each one from nine
random starts and keeps the one with the lowest Bayesian information
criterion (BIC). Ten levels later, the airline passenger series comes out as
`SE + Lin + WN x Lin + SE x Per x Lin`, which the report reads as:

1. A linearly increasing function.
2. An approximately periodic function with a period of 1.0 years and with
   linearly increasing amplitude.
3. A smooth function.
4. Uncorrelated noise with linearly increasing standard deviation.

## How we know the port is faithful

**Every published model reproduces.** The original's result files for its
thirteen benchmark series ship with this package. For each series and each of
the ten search levels, `tests/test_published.py` takes the published winning
model, evaluates its negative log likelihood (NLL) at the printed
hyperparameters and compares with the published number. All 130 agree: twelve
series to within 5e-5 nats, the solar series (whose covariance matrices reach
a condition number of 1e14) to within 5e-3 nats.

This test found one real defect during the port. A small diagonal jitter,
relative to the mean diagonal, moved the solar likelihood by 61 nats, because
a constant kernel dominated that diagonal. The first factorisation now adds
no jitter, as GPML did.

**The text matches.** For the published airline model the four component
summaries are word for word those of the 2014 system.

**The gradients match.** Autograd replaces GPML's hand-written derivatives;
a finite-difference test holds them to 1e-4.

**The search finds structures at least as good.** The search uses random
starts and a different optimiser (L-BFGS instead of GPML's conjugate
gradients), so it does not retrace the 2014 search digit for digit.
`scripts/replicate.py` reruns the full ten-level search under the published
settings and compares the final BIC (lower is better):

| series | published BIC | this port | difference |
|---|---|---|---|
| 01-airline | 1123.7 | 1108.9 | -14.8 |
| 02-solar | -870.3 | -872.3 | -2.0 |

The other eleven series were still running on 7 November 2026; this table
grows as they finish.

**What has no published yardstick.** The 2013 preset and the model-checking
p-values: the original repositories contain no numbers to compare them with.

## The Automatic Statistician family

Every project we found, from the original papers to the 2026 successors,
with what this repository takes from each. Each repository was checked on
GitHub on 7 November 2026.

| project | year | what it is | here |
|---|---|---|---|
| [gp-structure-search](https://github.com/jamesrobertlloyd/gp-structure-search) | 2013 | Compositional kernel search (CKS), ICML 2013. MATLAB and Python 2.7, MIT licence. | `Experiment.cks_2013()` preset |
| [gpss-research](https://github.com/jamesrobertlloyd/gpss-research) | 2014 | Automatic Bayesian Covariance Discovery (ABCD), AAAI 2014: changepoints, natural-language reports. MIT licence. | ported in full |
| [kmc-research](https://github.com/jamesrobertlloyd/kmc-research) | 2015 | Model criticism with the maximum mean discrepancy (MMD), NIPS 2015. MIT licence. | the model-checking section |
| [comp-kern](https://github.com/jamesrobertlloyd/comp-kern) | 2014 | "Simplified interface to compositional kernel learning". No licence. | read, not copied |
| [auto-stat-basic](https://github.com/jamesrobertlloyd/auto-stat-basic) | 2015 | "First attempts at automatic statistician architecture". No licence. | read, not copied |
| [autogpc](https://github.com/charles92/autogpc) | 2016 | Kernel search for GP classification (Qiurui He's thesis). No licence. | cited |
| [Relational Automatic Statistician](https://github.com/OpenXAIProject/Relational-Automatic-Statistician) | 2019 | Shared structure across several series (Hwang, Tong and Choi, arXiv 1511.08343). Apache 2.0. | cited |
| [autostat (sutoiku)](https://github.com/sutoiku/autostat) | 2022 | ABCD reimplementation in Mathematica. MIT licence. | cited |
| [autostat_gpflow](https://github.com/DavidJanz/autostat_gpflow) | 2018 | Basic GPflow implementation. MIT licence. | cited |
| [GPy-ABCD](https://github.com/T-Flet/GPy-ABCD) | 2021 | Configurable ABCD on GPy (Fletcher, Bundy and Nuamah, ICML 2021 AutoML workshop). BSD 3-clause. | cited |
| [AutoGP.jl](https://github.com/probsys/AutoGP.jl) | 2023 | Sequential Monte Carlo (SMC) over kernel structures (Saad et al., ICML 2023, arXiv 2307.09607). Julia, Apache 2.0, maintained. | cited |
| [AutoBNN](https://github.com/tensorflow/probability/tree/main/spinoffs/autobnn) | 2024 | Compositional Bayesian neural networks in JAX (Google). Apache 2.0. | cited |

Three related lines of work have no public code we could find: BoxLM, where a
large language model (LLM) proposes probabilistic programs (Li, Fox and
Goodman, ICML 2024, arXiv 2402.17879); a vision-language model pipeline that
judges kernel fits by their plots (Lee et al., arXiv 2509.25946, 2025); and
scalable kernel composition (Kim and Teh, AISTATS 2018, arXiv 1706.02524).
The site automaticstatistician.com is maintained independently of the
original authors and offers no code.

## Install

Python 3.11 or later. The dependencies are pure Python wheels on every
platform, so no system package is needed on macOS, Ubuntu or Windows.

```bash
pip install git+https://github.com/warith-harchaoui/automatic-statistician-revisited
# or, from a clone, for development:
pip install -r requirements-dev.txt
make check
```

Reading Parquet files needs `pip install "automatic-statistician-revisited[parquet]"`.
On Windows without `make`, run the four commands of the `Makefile` by hand.

## Usage

The command line reads a CSV or Parquet file:

```bash
autostat data.csv --time date --value y --out report/
autostat data.csv --time date --value y --preset cks-2013 --max-depth 4
autostat data.csv --time date --value y --no-check --json
```

The Python interface exposes each stage: `prepare` (dates to decimal years,
long series binned), `search` with an `Experiment`, `describe` for the
report, `fit` and `posterior` for a single model. [EXAMPLES.md](EXAMPLES.md)
has runnable recipes.

## Limits

- **Size.** Exact GP inference costs time cubic in the number of points.
  `prepare` averages any series longer than 1,000 points into 1,000 bins, the
  bound the original used. A structure finer than one bin goes unreported.
  The full ten-level search on the 144-point airline series took 4.1 hours
  with six worker processes on a shared laptop. Pass `--max-depth 3` for a
  first look in minutes.
- **One input dimension in the report.** Kernels and search accept several
  input dimensions; the English description assumes one.
- **Regression only.** The classification variant (autogpc) is not ported.
- **The 2013 preset approximates.** It uses the 2013 grammar and settings
  with the 2014 parameter initialisation. Its two random starts with a spread
  of 4 often miss a period: on the trend-plus-cycle series of
  `tests/test_search.py` it stops at `SE` where the 2014 preset finds the
  cycle.
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

MIT licence ([LICENSE](LICENSE)). The design ported here, the benchmark data
and the published results are by James Robert Lloyd, David Duvenaud and Roger
Grosse, under the MIT licence reproduced in
[LICENSE-ORIGINAL](LICENSE-ORIGINAL). Most benchmark series come from Rob
Hyndman's Time Series Data Library. [PORTING.md](PORTING.md) lists what
changed in the port and the defects found in the original along the way.
