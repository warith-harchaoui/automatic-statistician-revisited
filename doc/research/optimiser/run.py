"""Why does the port forecast better than the 2014 system? Search or optimiser.

What this script does
    The port, searching five levels, beats the forecasts the 2014 system
    published on the extrapolation benchmark (median standardised RMSE 1.51
    against 2.12), with the same grammar, starts and score. The 2014 run kept
    every model of its ten levels in ``*_result.txt`` files
    (``benchmark/published/searches_2014/``, copied from
    ``results/2013-12-10-extrap-GPSS`` of ``gpss-research``). For each of the
    thirteen series this script reads the 2014 winner and:

    * **2014 as published**: scores it in the port's code, with the 2014
      parameters, then forecasts the held-out tail. This checks that the port
      reproduces the 2014 forecast (against ``predictions_2014.parquet``).
    * **2014 refitted**: refits the same structure with the port's optimiser
      (L-BFGS in torch), started from the 2014 parameters, then forecasts.
    * **port**: the port's own depth-5 winner (``results/searches``).

    BIC is computed in the port on the same training part for all three, so
    the three are comparable: a lower BIC from the same structure means the
    2014 optimiser stopped short; a lower BIC from another structure means
    the port's search found a better expression.

Usage
    ``python doc/research/optimiser/run.py`` writes ``results.json`` here.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from autostat import SearchResult, posterior
from autostat.legacy import read_results
from autostat.model import fit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import run_2026  # noqa: E402


def scores(model, s) -> dict:
    """Forecast the held-out tail with one model and score it.

    Parameters
    ----------
    model : autostat.GPModel
        A fitted model.
    s : run_2026.Series
        One benchmark series.

    Returns
    -------
    dict
        BIC on the training part, RMSE and 90 % coverage on the tail.
    """
    mean, var = posterior(model, s.x_train, s.y_train, s.x_test, noise=True)
    sd = np.sqrt(var)
    inside = (s.y_test >= mean - 1.645 * sd) & (s.y_test <= mean + 1.645 * sd)
    return {
        "bic": float(model.bic),
        "rmse": float(np.sqrt(np.mean((mean - s.y_test) ** 2))),
        "coverage": float(np.mean(inside)),
        "structure": str(model.kernel.simplified().structure_key()),
        "mean": mean,
    }


def main() -> None:
    """Score the three forecasts on every series and write the table."""
    published = pd.read_parquet(ROOT / "benchmark/published/predictions_2014.parquet")
    published = published[published["method"] == "ABCD-2014"]
    rows = []
    for s in run_2026.load_series():
        levels = read_results(
            ROOT / "benchmark/published/searches_2014" / f"{s.name}_result.txt"
        )
        winner = min((m for level in levels for m in level), key=lambda m: m.bic)
        # Scored with zero iterations: the 2014 parameters, the port's likelihood.
        as_published = fit(
            winner, s.x_train, s.y_train, iters=0, subset=False, full_iters=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refitted = fit(winner, s.x_train, s.y_train)
        if not refitted.bic <= as_published.bic:
            refitted = as_published
        port = SearchResult.load(
            ROOT / "benchmark/results/searches/abcd_2014-depth5" / f"{s.name}.json"
        ).best
        tail = published[published["dataset"] == s.name].sort_values("step")
        row = {"dataset": s.name, "bic_2014_file": float(winner.bic)}
        for name, model in (
            ("published", as_published),
            ("refitted", refitted),
            ("port", port),
        ):
            score = scores(model, s)
            if name == "published":
                row["max_gap_to_2014_forecast"] = float(
                    np.max(np.abs(score["mean"] - tail["prediction"].to_numpy()))
                )
                row["rmse_2014_forecast"] = float(
                    np.sqrt(np.mean((tail["prediction"].to_numpy() - s.y_test) ** 2))
                )
            score.pop("mean")
            row.update({f"{name}_{k}": v for k, v in score.items()})
        rows.append(row)
        print(
            f"{s.name[:24]:24s} BIC 2014 {row['published_bic']:9.1f} "
            f"refit {row['refitted_bic']:9.1f} port {row['port_bic']:9.1f} | RMSE "
            f"{row['published_rmse']:9.3f} {row['refitted_rmse']:9.3f} "
            f"{row['port_rmse']:9.3f} (2014 file {row['rmse_2014_forecast']:9.3f})",
            flush=True,
        )
    (HERE / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
