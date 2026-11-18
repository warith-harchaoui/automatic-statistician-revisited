"""Measure whether a detector's breaks, proposed as dated shifts, help the search.

What this script does
    It searches the two synthetic families of ``doc/research/context_search``
    (a quarterly cycle with no jump; a yearly cycle with a level jump in
    mid-2014) three ways, with the same seeds and light settings:

    * ``grammar``: the 2014 grammar alone;
    * ``spectral``: plus the strongest periodogram peak as a period start
      (``spectral_starts=1``), the cycle the detector also uses;
    * ``breaks``: plus a level shift proposed at each of the three
      strongest breaks the detector finds (``break_starts=3``), which also
      proposes its own model with that cycle at the first level.

    The ``spectral`` arm separates what the cycle brings from what the
    dates bring. One JSON line per run, appended as it ends, so an
    interrupted run resumes; then the whole list.

Usage
    ``python doc/research/changepoints/run.py results.json [seeds]``;
    ``seeds`` defaults to 5. Light settings: three levels, four random
    starts, 100 optimiser iterations, two worker processes.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

from autostat import Experiment, describe, search
from autostat.context import Context
from autostat.features import features

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "context_search", HERE.parent / "context_search" / "run.py"
)
families = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(families)

#: What each arm adds to the 2014 settings.
ARMS = {
    "grammar": {},
    "spectral": {"spectral_starts": 1},
    "breaks": {"break_starts": 3},
}


def main() -> int:
    """Run every family, seed and arm; write the rows to the path given.

    Returns
    -------
    int
        0.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    warnings.filterwarnings("ignore")
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lines = Path(sys.argv[1]).with_suffix(".jsonl")
    rows = (
        [json.loads(line) for line in lines.read_text(encoding="utf-8").splitlines()]
        if lines.exists()
        else []
    )
    done = {(r["family"], r["seed"], r["arm"]) for r in rows}
    for name, make in (("jump", families.jump), ("quarterly", families.quarterly)):
        for seed in range(seeds):
            t, y, mapping = make(seed)
            context = Context.from_dict(mapping)
            for arm, extra in ARMS.items():
                if (name, seed, arm) in done:
                    continue
                experiment = Experiment.abcd_2014(
                    max_depth=3,
                    n_jobs=2,
                    random_seed=seed,
                    n_rand=4,
                    iters=100,
                    **extra,
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
