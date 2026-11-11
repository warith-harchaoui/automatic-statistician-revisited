"""Search with and without the business context on series whose truth is known.

What this script does
    It draws two synthetic families (a quarterly cycle, a level jump at a
    known date), three seeds each, and runs the same light structure search
    twice per series: blind, then with ``Context.search_hints()``. For each
    run it records the BIC, the structure kept and the features read against
    the context, one JSON line per run, then the whole list.

Usage
    ``python doc/research/context_search/run.py results.json``
    Two worker processes; set ``LOKY_MAX_CPU_COUNT=2`` and the BLAS thread
    variables to 1 to keep the machine usable.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from autostat import Experiment, describe, search
from autostat.context import Context
from autostat.features import features

Series = tuple[np.ndarray, np.ndarray, dict[str, Any]]


def quarterly(seed: int) -> Series:
    """Draw eight years of monthly data with a trend and a three-month cycle.

    Parameters
    ----------
    seed : int
        Random seed of the noise.

    Returns
    -------
    tuple
        Times in decimal years, values and the context as a mapping.
    """
    rng = np.random.default_rng(seed)
    t = 2010 + np.arange(96) / 12
    y = (
        50
        + 2 * (t - 2010)
        + 6 * np.sin(2 * np.pi * (t - 2010) / 0.25)
        + rng.normal(0, 1.5, t.size)
    )
    context = {"positive": True, "cycles": [{"name": "quarterly", "period": "3mo"}]}
    return t, y, context


def jump(seed: int) -> Series:
    """Draw ten years of monthly data with a yearly cycle and a jump in mid-2014.

    Parameters
    ----------
    seed : int
        Random seed of the noise.

    Returns
    -------
    tuple
        Times in decimal years, values and the context as a mapping.
    """
    rng = np.random.default_rng(seed)
    t = 2010 + np.arange(120) / 12
    y = (
        20
        + 1.0 * (t - 2010)
        + 8 * (t > 2014.5)
        + 3 * np.sin(2 * np.pi * (t - 2010))
        + rng.normal(0, 1.0, t.size)
    )
    context = {
        "positive": True,
        "cycles": [{"name": "yearly", "period": "1y"}],
        "events": [{"name": "reform", "at": 2014.5, "tolerance": "3mo"}],
    }
    return t, y, context


def main() -> int:
    """Run every family, seed and arm; write the rows to the path given.

    Returns
    -------
    int
        0.
    """
    warnings.filterwarnings("ignore")
    families: list[tuple[str, Callable[[int], Series]]] = [
        ("quarterly", quarterly),
        ("jump", jump),
    ]
    rows = []
    for name, make in families:
        for seed in range(3):
            t, y, mapping = make(seed)
            context = Context.from_dict(mapping)
            for guided in (False, True):
                hints = context.search_hints() if guided else {}
                experiment = Experiment.abcd_2014(
                    max_depth=3,
                    n_jobs=2,
                    n_rand=4,
                    iters=100,
                    random_seed=seed,
                    **hints,
                )
                tic = time.perf_counter()
                result = search(t, y, experiment)
                report = describe(result.best, t, y, name=name, check=False)
                row = features(report, context)
                record = {
                    "family": name,
                    "seed": seed,
                    "guided": guided,
                    "bic": result.best.bic,
                    "structure": result.best.kernel.structure_key(),
                    "seconds": time.perf_counter() - tic,
                    **{
                        k: v
                        for k, v in row.items()
                        if k.startswith(("cycle_", "event_", "unexpected"))
                    },
                }
                rows.append(record)
                print(json.dumps(record), flush=True)
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        json.dump(rows, out, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
