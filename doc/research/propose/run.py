"""Do a language model's proposals help the search, and can a wrong context mislead them?

What this script does
    It reuses the two synthetic families of ``doc/research/context_search``
    (a quarterly cycle; a yearly cycle with a level jump at a known date)
    and runs the same light search four ways per series:

    * ``grammar``: the grammar alone, as published;
    * ``llm_none``: plus a local language model's proposals, told nothing
      about the business;
    * ``llm_true``: plus proposals told the true context;
    * ``llm_wrong``: plus proposals told a misleading context (a wrong
      period, a wrong date).

    The context reaches the language model only, never the search's random
    starts, so the arms differ by the proposals alone. Features are read
    against the true context. One JSON line per run, appended as it ends,
    then the whole list.

Usage
    ``python doc/research/propose/run.py results.json [seeds] [model]``
    ``seeds`` defaults to 3, ``model`` to ``gemma3:12b`` on the local
    Ollama server.

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
from autostat.propose import Proposer, ollama

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "context_search", HERE.parent / "context_search" / "run.py"
)
families = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(families)


def main() -> int:
    """Run every family, seed and arm; write the rows to the path given.

    Returns
    -------
    int
        0.
    """
    warnings.filterwarnings("ignore")
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ask = ollama(sys.argv[3] if len(sys.argv) > 3 else "gemma3:12b", timeout=1800)
    lines = Path(sys.argv[1]).with_suffix(".jsonl")
    rows = (
        [json.loads(line) for line in lines.read_text(encoding="utf-8").splitlines()]
        if lines.exists()
        else []
    )
    done = {(r["family"], r["seed"], r["arm"]) for r in rows}
    for name, make in (("quarterly", families.quarterly), ("jump", families.jump)):
        for seed in range(seeds):
            t, y, mapping = make(seed)
            truth = Context.from_dict(mapping)
            arms = {
                "grammar": None,
                "llm_none": Proposer(ask),
                "llm_true": Proposer(ask, truth),
                "llm_wrong": Proposer(ask, Context.from_dict(families.WRONG[name])),
            }
            for arm, proposer in arms.items():
                if (name, seed, arm) in done:
                    continue
                experiment = Experiment.abcd_2014(
                    max_depth=3, n_jobs=2, random_seed=seed, n_rand=4, iters=100
                )
                tic = time.perf_counter()
                result = search(t, y, experiment, propose=proposer)
                report = describe(result.best, t, y, name=name, check=False)
                row = features(report, truth)
                best = result.best.kernel.simplified().structure_key()
                history = proposer.history if proposer else []
                proposed = {k for h in history for k in h.get("proposed", [])}
                record = {
                    "family": name,
                    "seed": seed,
                    "arm": arm,
                    "bic": result.best.bic,
                    "structure": best,
                    "best_was_proposed": best in proposed,
                    "proposals": history,
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
