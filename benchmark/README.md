# Extrapolation benchmark: 2014 methods against 2026 methods

The AAAI 2014 paper judged its system on a forecasting task. Each of the thirteen
series was cut in two: the system trained on the first 90 % and forecast the
last 10 %. The root mean squared error (RMSE) of its forecast was then set
against that of eight other methods. This folder runs that comparison again,
first with the predictions published in 2014, then with the methods
available in 2026, family by family.

## What is reproduced and how we know

`published/predictions_2014.parquet` holds the forecasts the 2014 systems
made. They come from the `results/<folder>/<name>_predictions.mat` files of
`github.com/jamesrobertlloyd/gpss-research` (MIT licence, commit `2a64958`).
Each published folder was matched to a column of the paper's table by
computing its standardised RMSE again. Every match is exact, so the mapping
below is proved, never guessed.

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

`score.py` rebuilds the paper's table from these predictions. The table
itself is `published/paper_table_standardised_rmse.parquet`, taken from the
upstream `source/R/postprocessing/extrap_sop_only.csv` behind the paper's box
plot. The two agree to within 2e-10. `tests/test_benchmark.py` keeps it
that way.

## Files and provenance

Everything is stored as Parquet. `make_parquet.py` rebuilds it from a
checkout of the original:

| file | content | upstream source |
|---|---|---|
| `data/series.parquet` | the thirteen 90/10 splits, columns `dataset, split, step, x, y` | `data/tsdlr_9010/<name>-{train,test}.csv` |
| `published/predictions_2014.parquet` | the forecasts of the nine 2014 methods | `results/<folder>/<name>_predictions.mat` |
| `published/paper_table_standardised_rmse.parquet` | the paper's table, with a `dataset` column | `source/R/postprocessing/extrap_sop_only.csv` |

`published/provenance.json` records the commit and the SHA-256 of all 144
upstream files read. `make_parquet.py` checks them before writing, so a
changed upstream file stops the rebuild. The Parquet values equal the
upstream values bit for bit.

An earlier version of this folder kept CSV copies. Their SHA-256 equal the
upstream ones listed in `provenance.json`. Reading them back gives the
Parquet tables exactly, with one catch: pandas' default CSV reader lands one
unit in the last place off on sixteen mauna2003 values. The option
`float_precision="round_trip"` avoids it. Parquet removes the question.

## The 2026 methods

| family | methods | how |
|---|---|---|
| classical | naive, seasonal naive, AutoETS, AutoARIMA, AutoTheta | `statsforecast` 2.1.1 |
| machine learning | LightGBM on lagged values, recursive, tuned by cross-validation | `lightgbm` 4.6.0 |
| deep learning | N-HiTS, N-BEATS, PatchTST, trained on the one series, tuned by cross-validation | `neuralforecast` 3.3.0, CPU |
| foundation, zero-shot | Chronos-2, Chronos-Bolt base, TimesFM 2.5, Mistral's Timestral (209 million parameters), TabPFN-TS | `chronos-forecasting` 2.3.2, `timesfm` 3.0.2, `timestral` 1.0.0, `tabpfn-time-series` 1.4.0 |
| foundation, in context | NVIDIA KumoTabular on lagged differences, recursive | `structured-data-models` 0.2.1 |
| structure search | this repository's 2014 and 2013 ports, ten levels | `autostat` |

## Hyper-parameters

AutoETS, AutoARIMA and AutoTheta choose their own orders by information
criterion, as their authors intend. The foundation models run zero-shot,
with their published defaults. The structure-search ports use the 2014 and
2013 published configurations.

LightGBM and the three deep models have hyper-parameters that nothing inside
them chooses. `tuning.py` picks them for each series by rolling-origin
cross-validation on the training part only. There are three origins
(`TimeSeriesSplit(n_splits=3, test_size=h)`, with `h` the length of the
held-out tail). Each fold forecasts the whole next block of `h` points from
its origin and scores its RMSE. The grid is a set of skrub 0.10.0
`choose_from` choices searched with `make_grid_search`. `skore.evaluate`
(skore 0.23.1) then evaluates the chosen learner with the same splitter. The
choices and the fold RMSEs go into `results/<method>.json` under `tuning`.

| method | grid |
|---|---|
| LightGBM | lags of 1, 2 or 4 seasons; 100, 300 or 1000 trees; learning rate 0.02, 0.05 or 0.1; 7, 15 or 31 leaves |
| N-HiTS, N-BEATS, PatchTST | input of 2 or 4 horizons; 250 or 1000 steps; learning rate 1e-3 or 1e-4 |

The held-out tail never enters the choice.

Four systems of the 2023 to 2026 period are missing from the table:

- AutoGP.jl (Saad and colleagues, ICML 2023) needed a Julia installation,
  which the machine lacked when the benchmark ran. The repository has since
  ported it to Python as `autostat.smc`. The benchmark does not run that
  port yet.
- GPy-ABCD (2021) needs GPy, which does not build on Python 3.13.
- AutoBNN (Google, 2024) imports `jax.interpreters.xla.pytype_aval_mappings`,
  which current JAX removed.
- Moirai (`uni2ts`) fails to build on Python 3.13.

## Run it

The 2026 libraries pin versions that conflict with each other. They live in
their own virtual environment, which reuses the `torch` of the base
installation:

```bash
python -m venv --system-site-packages .venv-bench
.venv-bench/bin/pip install statsforecast neuralforecast timesfm tabpfn-time-series skrub skore
.venv-bench/bin/pip uninstall -y torch   # keep the base torch and torchvision in step
.venv-bench/bin/python benchmark/run_2026.py --methods auto_ets chronos2
python benchmark/score.py
```

`run_2026.py` writes `results/<method>.parquet` and a JSON record of the
time taken, the failures and the chosen hyper-parameters. `score.py` writes
`results/scores.parquet`, `results/summary.parquet` and
[RESULTS.md](RESULTS.md).

The symbolic-regression family needs
`.venv-bench/bin/pip install pysr gplearn pyoperon`. PySR installs Julia on
first use. Each of these methods fits `y = f(t)` on centred, scaled time,
continues the formula over the tail and records the formulas it found in the
JSON. The macOS wheel of Operon does not load (`libxxhash`): its 13 failures
are recorded and it stays out of the table.

TabPFN-TS downloads its weights only after a one-time licence acceptance on
the Prior Labs site. Without it every series fails and the method stays out
of the table.

## Data

`data/series.parquet` holds the thirteen train and test splits of the
original's `data/tsdlr_9010`, unchanged. Most series come from Rob
Hyndman's Time Series Data Library.
