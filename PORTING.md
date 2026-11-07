# What the port changed, kept and found

The original is `github.com/jamesrobertlloyd/gpss-research` (last commit
27 July 2015, MIT licence, reproduced in `LICENSE-ORIGINAL`), with the 2013
settings of `github.com/jamesrobertlloyd/gp-structure-search` (last commit
2 July 2017) and the model-checking statistics of
`github.com/jamesrobertlloyd/kmc-research` (2015).

## Where each part went

| original | language | here |
|---|---|---|
| `flexible_function.py`: kernels, structural algebra, 2,249 lines | Python 2.7 | `kernels.py`, `model.py` |
| GPML `cov*.m`, `infDelta.m`, `minimize.m` | MATLAB | `kernels.py` (`torch`), `model.py` (L-BFGS) |
| `grammar.py`, 188 lines; 2013 `ONE_D_RULES` | Python 2.7 | `grammar.py` |
| `experiment.py`, `job_controller.py`, `cblparallel` | Python 2.7, Sun Grid Engine | `search.py` (`joblib`) |
| `component_stats_and_plots.m`, `order_by_mae_reduction.m` | MATLAB | `describe.py` |
| `checking_stats.m`, `mmdTestBoot_jl.m` | MATLAB | `checking.py` |
| `translation.py`, 1,348 lines | Python 2.7 | `describe.py` |
| result files read back with `eval` | Python 2.7 | `legacy.py` (`ast`, never executes) |

## Kept exactly

The parameterisation (log scales, GPML parameter order), so a kernel printed
in 2014 means the same thing here. The grammar rules and the way n-ary sums
are expanded. The simplification, canonical-form and additive-form rules. The
initialisation heuristics, including their constants. The out-of-bounds
rules. The BIC (Bayesian information criterion) with the original's
parameter count. The greedy loop. The published 2014 configuration. The
ordering of components by cross-validated mean absolute error (MAE). The
English sentences, two typos aside.

## Changed

- **One language.** Covariances are `torch` expressions; autograd replaces
  GPML's hand-written derivatives.
- **The optimiser.** L-BFGS with a strong-Wolfe line search replaces
  `minimize.m` (conjugate gradients). The two-stage schedule stays: `iters`
  on a random 250-point subset, then `full_iters` on all the data.
- **Parallelism.** A local `joblib` process pool replaces the cluster
  scheduler, which shipped `.mat` files over SSH (Secure Shell) and polled
  for results.
- **Serialisation.** `to_dict` and `from_dict` (JSON) replace `repr` and
  `eval`; old result files are parsed with `ast`.
- **Reproducibility.** Duplicates are removed in first-seen order. The
  original used `list(set(...))`, whose order follows hash randomisation, so
  two runs with one seed searched in different orders.
- **Numerical rescue.** A Cholesky factorisation that fails is retried with
  escalating jitter. The first attempt adds none, as GPML's `infDelta` did; a
  GPML fit that failed was scored NaN (not a number) and discarded.
- **Model checking.** `checking_stats.m` built the MMD (maximum mean
  discrepancy) test's "posterior" sample from a covariance that mixes two
  index sets. The port writes the intent instead: remove the component's
  posterior mean at the resampled inputs, then add a draw from the
  component's prior there. QQ (quantile-quantile), autocorrelation and
  periodogram statistics are ported as written and reported as two-sided
  p-values.
- **Output.** Markdown and PNG (Portable Network Graphics) figures replace
  LaTeX on the NIPS 2013 style file.

## Not ported

- Classification (`LikErf`; the `autogpc` project).
- The experimental kernels absent from every published run: `SP`, `PerOLD`,
  `LinOLD`.
- Multi-dimensional sentences: kernels and search take several input
  dimensions, the English description assumes one.

## Defects found in the original

None of them changes a published number.

- `experiment.py` filters out-of-bounds models before collecting them, so
  its `.oob` log was always empty.
- `GPModel.break_into_summands` passes `ean=` for `mean=` and raises if
  called.
- `translation.interval_intersection` returns the undefined name `none`.
- `ConstKernel.initialise_params` draws two independent uniforms, so its
  three branches have probabilities 1/3, 4/9 and 2/9 instead of thirds. The
  port keeps this behaviour, because the published runs had it.
