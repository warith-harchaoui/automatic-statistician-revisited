# Changelog

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
