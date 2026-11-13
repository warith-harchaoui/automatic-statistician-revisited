"""Search with and without the business context on series whose truth is known.

What this script does
    It draws two synthetic families (a quarterly cycle, a level jump at a
    known date) and runs the same structure search three times per series:
    blind, with the true context's hints (``Context.search_hints()``) and
    with a misleading context (a wrong period, a wrong date). Features are
    always read against the true context, so each arm is judged on what it
    recovers. One JSON line per run, then the whole list.

Usage
    ``python doc/research/context_search/run.py results.json [seeds] [full]``
    ``seeds`` defaults to 3; ``full`` uses the published search settings
    (nine random starts, 250 iterations) instead of the light ones.
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
from pathlib import Path
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


#: The misleading context of each family: a wrong period, a wrong date.
WRONG = {
    "quarterly": {"cycles": [{"name": "quarterly", "period": "5mo"}]},
    "jump": {
        "cycles": [{"name": "yearly", "period": "1y"}],
        "events": [{"name": "reform", "at": 2012.0}],
    },
}


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
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    settings = {} if "full" in sys.argv[3:] else {"n_rand": 4, "iters": 100}
    # Each run is appended to a JSON Lines file beside the output, so an
    # interrupted measurement starts again where it stopped.
    lines = Path(sys.argv[1]).with_suffix(".jsonl")
    rows = (
        [json.loads(line) for line in lines.read_text(encoding="utf-8").splitlines()]
        if lines.exists()
        else []
    )
    done = {(r["family"], r["seed"], r["arm"]) for r in rows}
    for name, make in families:
        for seed in range(seeds):
            t, y, mapping = make(seed)
            context = Context.from_dict(mapping)
            arms = {
                "blind": {},
                "true": context.search_hints(),
                "wrong": Context.from_dict(WRONG[name]).search_hints(),
            }
            for arm, hints in arms.items():
                if (name, seed, arm) in done:
                    continue
                experiment = Experiment.abcd_2014(
                    max_depth=3, n_jobs=2, random_seed=seed, **settings, **hints
                )
                tic = time.perf_counter()
                result = search(t, y, experiment)
                report = describe(result.best, t, y, name=name, check=False)
                row = features(report, context)
                record = {
                    "family": name,
                    "seed": seed,
                    "arm": arm,
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
                with lines.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(record) + "\n")
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        json.dump(rows, out, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
