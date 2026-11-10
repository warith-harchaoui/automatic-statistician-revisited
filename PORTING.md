# What the port changed, kept and found

The original is `github.com/jamesrobertlloyd/gpss-research`. Its last commit
dates from 27 July 2015 and its MIT licence is reproduced in
`LICENSE-ORIGINAL`. Two sibling repositories complete it: the 2013 settings
come from `github.com/jamesrobertlloyd/gp-structure-search` (last commit
2 July 2017), the model-checking statistics from
`github.com/jamesrobertlloyd/kmc-research` (2015).

## Where each part went

| original | language | here |
|---|---|---|
| `flexible_function.py`: kernels, structural algebra, 2,249 lines | Python 2.7 | `kernels.py`, `model.py` |
| GPML `cov*.m`, `infDelta.m`, `minimize.m` | MATLAB | `kernels.py` (`torch`), `model.py` (L-BFGS) |
| `grammar.py` (188 lines) and the 2013 `ONE_D_RULES` | Python 2.7 | `grammar.py` |
| `experiment.py`, `job_controller.py`, `cblparallel` | Python 2.7, Sun Grid Engine | `search.py` (`joblib`) |
| `component_stats_and_plots.m`, `order_by_mae_reduction.m` | MATLAB | `describe.py` |
| `checking_stats.m`, `mmdTestBoot_jl.m` | MATLAB | `checking.py` |
| `translation.py`, 1,348 lines | Python 2.7 | `describe.py` |
| result files read back with `eval` | Python 2.7 | `legacy.py` (`ast`, which never executes them) |

## Kept exactly

A kernel printed in 2014 means the same thing here, because the port keeps
the parameterisation: log scales, in GPML's parameter order. The port also
keeps, unchanged:

- the grammar rules and the way n-ary sums are expanded;
- the simplification, canonical-form and additive-form rules;
- the initialisation heuristics, constants included;
- the out-of-bounds rules;
- the BIC (Bayesian information criterion) with the original's parameter
  count;
- the greedy loop and the published 2014 configuration;
- the ordering of components by cross-validated mean absolute error (MAE);
- the English sentences, two typos aside.

## Changed

- **One language.** Covariances are `torch` expressions. Autograd replaces
  GPML's hand-written derivatives.
- **The optimiser.** L-BFGS with a strong-Wolfe line search replaces
  `minimize.m` (conjugate gradients). The two-stage schedule stays: `iters`
  on a random 250-point subset, then `full_iters` on all the data.
- **Parallelism.** A local `joblib` process pool replaces the cluster
  scheduler, which shipped `.mat` files over SSH (Secure Shell) and polled
  for results.
- **Serialisation.** `to_dict` and `from_dict` (JSON) replace `repr` and
  `eval`. Old result files are parsed with `ast`.
- **Reproducibility.** Duplicates are removed in first-seen order. The
  original used `list(set(...))`, whose order follows hash randomisation: two
  runs with one random seed searched in different orders.
- **Numerical rescue.** A Cholesky factorisation that fails is retried with
  growing jitter. The first attempt adds none, as GPML's `infDelta` did. In
  GPML, a fit that failed was scored NaN (not a number) and discarded.
- **Model checking.** `checking_stats.m` built the "posterior" sample of the
  MMD (maximum mean discrepancy) test from a covariance that mixes two index
  sets. The port writes what the code meant to do: remove the component's
  posterior mean at the resampled inputs, then add a draw from the
  component's prior there. The QQ (quantile-quantile), autocorrelation and
  periodogram statistics are ported as written and reported as two-sided
  p-values.
- **Output.** The 2014 system wrote LaTeX on the NIPS 2013 style file. The
  port writes Markdown with SVG figures drawn by `sprezzature-figures`, or,
  on request, a plain LaTeX article with PDF figures.

## Not ported

- Classification (`LikErf` and the `autogpc` project).
- The experimental kernels that no published run used: `SP`, `PerOLD`,
  `LinOLD`.
- English descriptions of inputs with several dimensions. Kernels and search
  accept several input dimensions, but the sentences assume one.

## Defects found in the original

The port found four defects. None changes a published number.

- `experiment.py` filters out-of-bounds models before collecting them, so
  its `.oob` log was always empty.
- `GPModel.break_into_summands` passes `ean=` for `mean=` and raises if
  called.
- `translation.interval_intersection` returns the undefined name `none`.
- `ConstKernel.initialise_params` draws two independent uniforms, so its
  three branches have probabilities 1/3, 4/9 and 2/9 instead of a third each.
  The port keeps this behaviour, because the published runs had it.
