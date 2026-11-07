# Changelog

## Unreleased

- `autostat.smc`: AutoGP.jl (ICML 2023) ported to Python. Sequential Monte
  Carlo over kernel structures, subtree-replace and detach-attach moves,
  HMC, online `add_data`, mixture forecasts. Checked against AutoGP.jl
  v0.1.19 through `juliacall` (`tests/test_smc_julia.py`): densities,
  covariances, move ratios and predictions agree to 1e-12 or better.
- Found that AutoGP.jl's HMC moves kernel parameters by the prior gradient
  only; reproduced by default, `HMCConfig(gradient="full")` for the
  intended sampler.
- Extrapolation benchmark: the paper's table recomputed from the published
  2014 forecasts; the methods of 2026 are scored on the same thirteen
  series, family by family.
- Benchmark data stored as Parquet, rebuilt from the original repository
  by `benchmark/make_parquet.py`, which checks the SHA-256 of every
  upstream file first.
- LightGBM and the neuralforecast models tuned per series by rolling-origin
  cross-validation on the training part (skrub grid, skore evaluation).

## 0.1.0, 7 November 2026

First release.

- Port of the 2014 Automatic Bayesian Covariance Discovery (ABCD) system to
  Python 3 and `torch`: kernels, grammar, greedy search, decomposition,
  English report.
- The 2013 compositional kernel search configuration as a preset.
- Model checking (QQ, autocorrelation, periodogram, MMD) from the original's
  MATLAB code.
- The thirteen benchmark series and the 2014 results, with a test that
  re-scores all 130 published models.
- Command line `autostat`, with `--preset`, `--json` and `--no-check`.
