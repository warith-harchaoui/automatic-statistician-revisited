# Extrapolation benchmark: 2014 methods against 2026 methods

The AAAI 2014 paper judged its system on a forecasting task: train on the
first 90 % of each of thirteen series, forecast the last 10 %, then compare
the root mean squared error (RMSE) of the forecast with that of eight other
methods. This folder reruns that comparison, first with the published 2014
predictions, then with the methods available in 2026, family by family.

## What is reproduced; how we know

`published/predictions_2014.csv` holds the forecasts the 2014 systems made,
read from the result files of `github.com/jamesrobertlloyd/gpss-research`
(MIT licence). Each published folder was matched to a column of the paper's
table by recomputing its standardised RMSE: the match is exact, so the
mapping is proved rather than guessed.

| method | published folder | what it is |
|---|---|---|
| ABCD-2014 | `2013-12-10-extrap-GPSS` | the AAAI 2014 system ("ABCD accuracy" in the paper) |
| CKS-2013 | `2013-11-08-extrap-GPSS` | the ICML 2013 compositional kernel search |
| SP | `2013-12-17-extrap-SP-bic` | spectral mixture kernels |
| TCI | `2013-11-11-extrap-TCI` | trend, cyclical and irregular components |
| MKL | `2013-11-11-extrap-MKL` | Bayesian multiple kernel learning |
| Eureqa | `eureqa-extrap` | symbolic regression |
| CP | `2013-11-11-extrap-CP` | changepoint kernels |
| SE | `2014-01-28-extrap-SE` | one squared exponential kernel |
| LN | `2014-01-28-extrap-LN` | linear regression |

`score.py` recomputes the paper's table (`published/paper_table_standardised_rmse.csv`,
the file behind the paper's box plot) from these predictions and agrees with
it to within 2e-10. `tests/test_benchmark.py` keeps it that way.

## The 2026 methods

| family | methods | how |
|---|---|---|
| classical | naive, seasonal naive, AutoETS, AutoARIMA, AutoTheta | `statsforecast` 2.1.1 |
| machine learning | LightGBM on lagged differences, recursive | `lightgbm` 4.6.0 |
| deep learning | N-HiTS, N-BEATS, PatchTST, trained on the one series | `neuralforecast` 3.3.0 |
| foundation, zero-shot | Chronos-2, Chronos-Bolt base, TimesFM 2.5, TabPFN-TS | `chronos-forecasting` 2.3.2, `timesfm` 3.0.2, `tabpfn-time-series` 1.4.0 |
| foundation, in context | NVIDIA KumoTabular on lagged differences, recursive | `structured-data-models` 0.2.1 |
| structure search | this repository's 2014 and 2013 ports, ten levels | `autostat` |

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
.venv-bench/bin/pip install statsforecast neuralforecast timesfm tabpfn-time-series
.venv-bench/bin/pip uninstall -y torch   # keep the base torch and torchvision in step
.venv-bench/bin/python benchmark/run_2026.py --methods auto_ets chronos2
python benchmark/score.py
```

`run_2026.py` writes `results/<method>.csv` and a JSON record of time and
failures; `score.py` writes `results/scores.csv`, `results/summary.csv` and
[RESULTS.md](RESULTS.md).

## Data

`data/` holds the thirteen train and test files of the original's
`data/tsdlr_9010`, unchanged. Most series come from Rob Hyndman's Time
Series Data Library.
