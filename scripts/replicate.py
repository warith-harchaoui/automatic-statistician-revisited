#!/usr/bin/env python3
"""Rerun the published 2014 search on the thirteen benchmark series.

    python scripts/replicate.py                          # all thirteen
    python scripts/replicate.py --only 01-airline --max-depth 3

What this script does
    For each benchmark series it performs two checks of different strength
    against the 2014 results shipped in the package, then writes a table.

    * **Likelihood at the published parameters.** Every level's published
      winner is re-scored at exactly its printed hyperparameters. This tests
      the kernels and the marginal likelihood and involves no randomness, so
      agreement is to rounding.
    * **The search itself.** The port searches from scratch under the
      published settings and its winner's BIC (Bayesian information
      criterion) is compared with the published winner's. Random starts and
      a different optimiser rule out digit-for-digit agreement, so the
      question is whether the port finds structures as good.

What it produces
    ``<out>/comparison.csv``, one ``<name>.json`` search record and one
    report directory per series.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from autostat import Experiment, benchmarks, describe, search
from autostat.model import negative_log_likelihood

LOGGER = logging.getLogger("replicate")


def published_agreement(name: str) -> float:
    """Return the largest NLL difference over the published level winners.

    Parameters
    ----------
    name : str
        Benchmark name.

    Returns
    -------
    float
        Maximum absolute difference, in nats.
    """
    x, y = benchmarks.load(name)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    worst = 0.0
    for model in benchmarks.published(name):
        theta = torch.as_tensor(model.theta_vector())
        with torch.no_grad():
            mine = float(negative_log_likelihood(model, xt, yt, theta))
        worst = max(worst, abs(mine - model.nll))
    return worst


def make_parser() -> argparse.ArgumentParser:
    """Build the parser without running anything.

    Returns
    -------
    argparse.ArgumentParser
        The parser for :func:`main`.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", help="benchmark names (default: all)")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=Path("results/replication"))
    parser.add_argument("--no-check", action="store_true", help="skip model checking")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replication and write the comparison table.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 on completion.
    """
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.only or benchmarks.names():
        x, y = benchmarks.load(name)
        agreement = published_agreement(name)
        final = benchmarks.published(name)[-1]
        LOGGER.info(
            "== %s (n=%d): NLL agreement at published parameters %.2e",
            name,
            len(y),
            agreement,
        )
        started = time.perf_counter()
        result = search(
            x, y, Experiment.abcd_2014(max_depth=args.max_depth, n_jobs=args.n_jobs)
        )
        result.save(args.out / f"{name}.json")
        describe(
            result.best,
            x,
            y,
            name=name,
            out_dir=args.out / name,
            check=not args.no_check,
        )
        rows.append(
            {
                "dataset": name,
                "n": len(y),
                "nll_agreement_at_published_params": agreement,
                "published_bic": final.bic,
                "port_bic": result.best.bic,
                "port_minus_published_bic": result.best.bic - final.bic,
                "published_structure": final.kernel.structure(),
                "port_structure": result.best.kernel.structure(),
                "seconds": time.perf_counter() - started,
            }
        )
        pd.DataFrame(rows).to_csv(args.out / "comparison.csv", index=False)
    LOGGER.info("\n%s", pd.DataFrame(rows).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
