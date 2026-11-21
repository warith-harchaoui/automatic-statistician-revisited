#!/usr/bin/env python3
"""Score the methods on the recent benchmark, whose tails lie in 2026.

    python benchmark/recent/score.py

What this script does
    It reads the forecasts ``run_2026.py --suite recent`` wrote to
    ``benchmark/results-recent/`` and scores them as ``benchmark/score.py``
    scores the 2014 series: RMSE per series divided by the best RMSE on that
    series (standardised RMSE, 1 is the best), the median over the series
    with a bootstrap 95 % interval and the methods tied with the leader,
    MASE (the error in units of the training part's mean one-step change),
    and the coverage of the 90 % interval where a method gives one. A method
    is ranked only once it has forecast every series.

What it produces
    ``benchmark/recent/RESULTS.md`` and ``benchmark/results-recent/scores.parquet``.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

from score import resolution, rmse_table  # noqa: E402

RESULTS = BENCH / "results-recent"
DATA = BENCH / "data" / "recent.parquet"


def predictions() -> tuple[pd.DataFrame, dict[str, str]]:
    """Read every method's forecasts beside the held-out values.

    Returns
    -------
    predictions : pandas.DataFrame
        Columns ``dataset, method, step, actual, prediction, lower, upper``.
    families : dict
        Method to family.
    """
    table = pd.read_parquet(DATA)
    actual = table[table["split"] == "test"][["dataset", "step", "y"]].rename(
        columns={"y": "actual"}
    )
    frames, families = [], {}
    for path in sorted(RESULTS.glob("*.json")):
        table_path = path.with_suffix(".parquet")
        if not table_path.exists():
            continue
        frame = pd.read_parquet(table_path)
        if frame.empty:
            continue
        frame["method"] = path.stem
        frames.append(frame.merge(actual, on=["dataset", "step"]))
        families[path.stem] = json.loads(path.read_text())["family"]
    return pd.concat(frames, ignore_index=True), families


def main() -> int:
    """Score everything and write the table.

    Returns
    -------
    int
        0.
    """
    pred, families = predictions()
    rmse = rmse_table(pred)
    complete = rmse.columns[rmse.notna().all()]
    standardised = rmse[complete].div(rmse[complete].min(axis=1), axis=0)
    resolved = resolution(standardised)

    table = pd.read_parquet(DATA)
    train = table[table["split"] == "train"].sort_values(["dataset", "step"])
    scale = train.groupby("dataset")["y"].apply(lambda y: y.diff().abs().mean())
    mae = (
        (pred["actual"] - pred["prediction"])
        .abs()
        .groupby([pred["dataset"], pred["method"]])
        .mean()
        .unstack("method")
    )
    mase = mae.div(scale, axis=0)
    inside = (pred["actual"] >= pred["lower"]) & (pred["actual"] <= pred["upper"])
    coverage = inside.where(pred["lower"].notna()).groupby(pred["method"]).mean()

    rows = []
    for method in complete:
        rows.append(
            {
                "method": method,
                "family": families[method],
                "median": standardised[method].median(),
                "low": resolved.loc[method, "low"],
                "high": resolved.loc[method, "high"],
                "p_best": resolved.loc[method, "p_best"],
                "tied": bool(resolved.loc[method, "tied"]),
                "mean": standardised[method].mean(),
                "won": int((standardised[method] <= 1 + 1e-12).sum()),
                "coverage": coverage.get(method, np.nan),
                "mase": mase[method].median(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("median").reset_index(drop=True)
    standardised.stack().rename("standardised_rmse").reset_index().to_parquet(
        RESULTS / "scores.parquet", index=False
    )
    summary.to_parquet(RESULTS / "summary.parquet", index=False)

    n = len(standardised)
    lines = [
        "# Recent benchmark results",
        "",
        f"{n} public monthly series, trained on 2006 to 2025 and forecast over",
        "the months of 2026 published when the data were built",
        "(`make_data.py`, `provenance.json`). These tails postdate the",
        "release of Chronos-2 and TimesFM 2.5, so no method can have seen them.",
        "",
        "Standardised RMSE: each series' RMSE divided by the best RMSE on that",
        f"series among the methods listed, so 1.00 is the best; median over the {n}",
        "series, lower is better. 95 % interval, P(best) and tied: a bootstrap over",
        "the series (2000 resamples, seed 0), read as in `benchmark/RESULTS.md`.",
        "MASE: mean absolute error over the training part's mean absolute",
        "one-step change, median across series.",
        "",
        "| method | family | median | 95 % interval | P(best) | tied | mean | series won | 90 % coverage | MASE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        cov = "" if np.isnan(r.coverage) else f"{r.coverage:.2f}"
        lines.append(
            f"| {r.method} | {r.family} | {r.median:.2f} | {r.low:.2f} to {r.high:.2f} | "
            f"{r.p_best:.2f} | {'yes' if r.tied else ''} | {r.mean:.2f} | {r.won} | "
            f"{cov} | {r.mase:.2f} |"
        )
    missing = sorted(set(rmse.columns) - set(complete))
    if missing:
        lines += ["", "Not ranked, some series missing: " + ", ".join(missing) + "."]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
