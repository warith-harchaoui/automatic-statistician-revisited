# Extrapolation benchmark: 2014 methods against 2026 methods

The AAAI 2014 paper judged its system on a forecasting task: train on the
first 90 % of each of thirteen series, forecast the last 10 %, then compare
the root mean squared error (RMSE) of the forecast with that of eight other
methods. This folder reruns that comparison, first with the published 2014
predictions, then with the methods available in 2026, family by family.

## What is reproduced; how we know

`published/predictions_2014.parquet` holds the forecasts the 2014 systems
made, read from the `results/<folder>/<name>_predictions.mat` files of
`github.com/jamesrobertlloyd/gpss-research` (MIT licence, commit `2a64958`).
Each published folder was matched to a column of the paper's table by
recomputing its standardised RMSE: the match is exact, so the mapping is
proved rather than guessed.

| method | published folder | what it is |
|---|---|---|
| ABCD-2014 | `2013-12-10-extrap-GPSS` | the AAAI 2014 system ("ABCD accuracy" in the paper) |
| CKS-2013 | `2013-11-08-extrap-GPSS` | the ICML 2013 compositional kernel search |
| SP | `2013-12-17-extrap-SP-bic` | spectral mixture kernels |
| TCI | `2013-11-11-extrap-TCI` | trend, cyclical and irregular components |
| MKL | `2013-11-11-extrap-MKL` | Bayesian multiple kernel learning |
| Eureqa | `eureqa-extrap` | symbolic regression (the 2026 family compares it with PySR and gplearn) |
| CP | `2013-11-11-extrap-CP` | changepoint kernels |
| SE | `2014-01-28-extrap-SE` | one squared exponential kernel |
| LN | `2014-01-28-extrap-LN` | linear regression |

`score.py` recomputes the paper's table
(`published/paper_table_standardised_rmse.parquet`, from the upstream
`source/R/postprocessing/extrap_sop_only.csv` behind the paper's box plot)
from these predictions and agrees with it to within 2e-10.
`tests/test_benchmark.py` keeps it that way.

## Files and provenance

Everything is stored as Parquet, rebuilt from a checkout of the original by
`make_parquet.py`:

| file | content | upstream source |
|---|---|---|
| `data/series.parquet` | the thirteen 90/10 splits, columns `dataset, split, step, x, y` | `data/tsdlr_9010/<name>-{train,test}.csv` |
| `published/predictions_2014.parquet` | the nine 2014 methods' forecasts | `results/<folder>/<name>_predictions.mat` |
| `published/paper_table_standardised_rmse.parquet` | the paper's table, with a `dataset` column | `source/R/postprocessing/extrap_sop_only.csv` |

`published/provenance.json` records the commit and the SHA-256 of all 144
upstream files read; `make_parquet.py` checks them before writing, so a
changed upstream file stops the rebuild. The Parquet values are the
upstream values bit for bit. An earlier version of this folder kept CSV
copies; their SHA-256 equal the upstream ones listed in `provenance.json`,
and reading them back equals the Parquet tables exactly (pandas' default CSV
reader is one unit in the last place off on sixteen mauna2003 values, which
`float_precision="round_trip"` avoids; Parquet removes the question).

## The 2026 methods

| family | methods | how |
|---|---|---|
| classical | naive, seasonal naive, AutoETS, AutoARIMA, AutoTheta | `statsforecast` 2.1.1 |
| machine learning | LightGBM on lagged values, recursive, tuned by cross-validation | `lightgbm` 4.6.0 |
| deep learning | N-HiTS, N-BEATS, PatchTST, trained on the one series, tuned by cross-validation | `neuralforecast` 3.3.0, CPU |
| foundation, zero-shot | Chronos-2, Chronos-Bolt base, TimesFM 2.5, TabPFN-TS | `chronos-forecasting` 2.3.2, `timesfm` 3.0.2, `tabpfn-time-series` 1.4.0 |
| foundation, in context | NVIDIA KumoTabular on lagged differences, recursive | `structured-data-models` 0.2.1 |
| structure search | this repository's 2014 and 2013 ports, ten levels | `autostat` |

## Hyper-parameters

AutoETS, AutoARIMA and AutoTheta choose their own orders by information
criterion, as their authors intend. The foundation models are used
zero-shot, with their published defaults. The structure-search ports use
the 2014 and 2013 published configurations.

LightGBM and the three deep models have hyper-parameters that nothing
inside them chooses. `tuning.py` chooses them per series by rolling-origin
cross-validation on the training part only: three origins
(`TimeSeriesSplit(n_splits=3, test_size=h)`, `h` the length of the
held-out tail), each fold forecasting the whole next block of `h` points
from its origin and scoring its RMSE. The grid is a set of skrub 0.10.0
`choose_from` choices searched with `make_grid_search`; the chosen learner
is then evaluated by `skore.evaluate` (skore 0.23.1) with the same splitter. The
choices and fold RMSEs go into `results/<method>.json` under `tuning`.

| method | grid |
|---|---|
| LightGBM | lags of 1, 2 or 4 seasons; 100, 300 or 1000 trees; learning rate 0.02, 0.05 or 0.1; 7, 15 or 31 leaves |
| N-HiTS, N-BEATS, PatchTST | input of 2 or 4 horizons; 250 or 1000 steps; learning rate 1e-3 or 1e-4 |

The held-out tail never enters the choice.

Three structure-search systems of the 2023 to 2026 period do not run here.
AutoGP.jl (Saad et al., ICML 2023) needs a Julia installation, which this
machine lacks. GPy-ABCD (2021) needs GPy, which does not build on Python
3.13. AutoBNN (Google, 2024) imports `jax.interpreters.xla.pytype_aval_mappings`,
which current JAX removed. Moirai (`uni2ts`) also fails to build on Python
3.13.

## Run it

The 2026 libraries pin versions that conflict with each other, so they live
in their own virtual environment, which reuses the base installation's
`torch`:

```bash
python -m venv --system-site-packages .venv-bench
.venv-bench/bin/pip install statsforecast neuralforecast timesfm tabpfn-time-series skrub skore
.venv-bench/bin/pip uninstall -y torch   # keep the base torch and torchvision in step
.venv-bench/bin/python benchmark/run_2026.py --methods auto_ets chronos2
python benchmark/score.py
```

`run_2026.py` writes `results/<method>.parquet` and a JSON record of time,
failures and chosen hyper-parameters; `score.py` writes
`results/scores.parquet`, `results/summary.parquet` and
[RESULTS.md](RESULTS.md).

The symbolic-regression family needs `.venv-bench/bin/pip install pysr gplearn
pyoperon`; PySR installs Julia on first use. It fits `y = f(t)` on centred,
scaled time, continues the formula and records the formulas found in the
JSON. Operon's macOS wheel does not load (`libxxhash`), so its 13 failures are
recorded and it is left out of the table.

TabPFN-TS needs a one-time licence acceptance on the Prior Labs site before
it downloads its weights; without it every series fails and the method is
left out of the table.

## Data

`data/series.parquet` holds the thirteen train and test splits of the
original's `data/tsdlr_9010`, unchanged. Most series come from Rob Hyndman's Time
Series Data Library.
