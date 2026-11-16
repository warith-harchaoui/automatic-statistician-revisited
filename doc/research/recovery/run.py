"""Does the search find the structure that drew a series? A recovery study.

What this script does
    It draws series from the grammar itself (:mod:`autostat.synthetic`): a
    random structure one or two rewrites from a base kernel, parameters drawn
    as the search's starts are, ten years of monthly values and white noise
    at a fifth of the signal's spread. It then runs the 2014 search on each
    series and compares what it found with what drew it:

    * the properties both carry (a cycle, a changepoint or window, a linear
      trend, a smooth term) and, for a cycle, whether its period is found
      within 5 %;
    * the BIC of the search's model against the BIC of the true structure
      refitted from its true parameters. When the truth scores better, the
      search missed a better model (a search failure); when the search
      scores as well or better with another structure, the data do not
      single out the truth (an identifiability limit).

    One JSON line per series, appended as it ends, so an interrupted run
    resumes; then the whole list.

Usage
    ``python doc/research/recovery/run.py results.json [series]``
    ``series`` defaults to 40. Light settings: three levels, four random
    starts, 100 optimiser iterations, two worker processes.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from autostat import Experiment, search
from autostat import kernels as ff
from autostat import synthetic as sy
from autostat.features import leaves, periods
from autostat.model import GPModel, fit

#: Ten years of monthly inputs.
X = 2000 + np.arange(120) / 12

#: White noise, as a share of the signal's standard deviation.
NOISE = 0.2


def properties(kernel: ff.Kernel) -> dict[str, bool]:
    """Say which kinds of component an expression holds.

    Parameters
    ----------
    kernel : Kernel
        Any expression.

    Returns
    -------
    dict
        ``periodic``, ``switch``, ``linear`` and ``smooth``.
    """
    kinds = {type(k) for k in leaves(kernel.simplified())}
    return {
        "periodic": ff.PeriodicKernel in kinds,
        "switch": bool(kinds & {ff.ChangePointKernel, ff.ChangeWindowKernel}),
        "linear": ff.LinearKernel in kinds,
        "smooth": bool(kinds & {ff.SqExpKernel, ff.RQKernel, ff.GammaExpKernel}),
    }


def one(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    """Draw one series, search it and compare.

    Parameters
    ----------
    seed : int
        Seeds the structure, its parameters, the noise and the search.
    settings : dict
        Overrides of :meth:`Experiment.abcd_2014`.

    Returns
    -------
    dict
        The truth, the finding and their comparison.
    """
    rng = np.random.default_rng(seed)
    steps = int(rng.integers(1, 3))
    for _ in range(20):
        structure = sy.random_structure(rng, steps)
        try:
            signal, truth = sy.sample(structure, X, rng, noise=0.0)
        except ValueError:
            continue
        spread = float(np.std(signal[0]))
        if math.isfinite(spread) and spread > 1e-6:
            break
    y = signal[0] + NOISE * spread * rng.standard_normal(X.size)
    tic = time.perf_counter()
    result = search(X, y, Experiment.abcd_2014(random_seed=seed, **settings))
    seconds = time.perf_counter() - tic
    noise = ff.NoiseKernel(sf=math.log(NOISE * spread))
    refit = fit(
        GPModel(ff.SumKernel([truth.copy(), noise]), ndata=y.size),
        X,
        y,
        iters=settings.get("iters", 250),
    )
    found = result.best.kernel
    want, got = properties(truth), properties(found)
    true_periods = periods(truth)
    found_periods = periods(found)
    return {
        "seed": seed,
        "steps": steps,
        "truth": truth.simplified().structure_key(),
        "found": found.simplified().structure_key(),
        "bic_found": result.best.bic,
        "bic_truth": refit.bic,
        **{f"truth_{k}": v for k, v in want.items()},
        **{f"found_{k}": v for k, v in got.items()},
        "periods_truth": true_periods,
        "periods_found": found_periods,
        "period_recovered": (
            all(
                any(abs(f / p - 1) <= 0.05 for f in found_periods) for p in true_periods
            )
            if true_periods
            else None
        ),
        "seconds": seconds,
    }


def main() -> int:
    """Run every series; write the rows to the path given.

    Returns
    -------
    int
        0.
    """
    warnings.filterwarnings("ignore")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    settings = {"max_depth": 3, "n_rand": 4, "iters": 100, "n_jobs": 2}
    lines = Path(sys.argv[1]).with_suffix(".jsonl")
    rows = (
        [json.loads(line) for line in lines.read_text(encoding="utf-8").splitlines()]
        if lines.exists()
        else []
    )
    done = {r["seed"] for r in rows}
    for seed in range(count):
        if seed in done:
            continue
        record = one(seed, settings)
        rows.append(record)
        print(json.dumps(record), flush=True)
        with lines.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        json.dump(rows, out, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
