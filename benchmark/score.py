#!/usr/bin/env python3
"""Score the 2014 and 2026 methods on the extrapolation benchmark.

    python benchmark/score.py

What this script does
    It computes, for every method and series, the root mean squared error
    (RMSE) of the forecast of the held-out tail. Each series' RMSEs are then
    divided by the best RMSE on that series, the "standardised RMSE" of the
    AAAI 2014 paper: 1 is the best method on that series, 2 is twice its
    error.

    It first recomputes the paper's own table from the published 2014
    predictions, using only the eight methods the paper compared. It checks
    the result against ``published/paper_table_standardised_rmse.parquet``. The
    full ranking then puts the 2014 and 2026 methods on one scale.

What it produces
    ``results/scores.parquet`` (one row per method and series),
    ``results/summary.parquet`` (one row per method) and ``RESULTS.md``.

    A method run before the move to Parquet left ``results/<method>.csv``;
    it is converted to ``results/<method>.parquet`` (values checked equal)
    and the CSV removed.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("score")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

#: The paper's columns, in its order, mapped to the published method names.
PAPER_COLUMNS = {
    "AABCD": "ABCD-2014",
    "SP": "SP",
    "TCI": "TCI",
    "MKL": "MKL",
    "Eureqa": "Eureqa",
    "CP": "CP",
    "SE": "SE",
    "LN": "LN",
}

#: Families of the 2014 methods, for the combined table.
FAMILIES_2014 = {
    "ABCD-2014": "structure search (2014 published)",
    "CKS-2013": "structure search (2014 published)",
    "SP": "2014 baselines",
    "TCI": "2014 baselines",
    "MKL": "2014 baselines",
    "Eureqa": "symbolic regression (2014 published)",
    "CP": "2014 baselines",
    "SE": "2014 baselines",
    "LN": "2014 baselines",
}


def test_values() -> pd.DataFrame:
    """Read the held-out values of every series.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, step, actual``.
    """
    table = pd.read_parquet(HERE / "data" / "series.parquet")
    test = table[table["split"] == "test"]
    return (
        test[["dataset", "step", "y"]]
        .rename(columns={"y": "actual"})
        .sort_values(["dataset", "step"])
        .reset_index(drop=True)
    )


def rmse_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """RMSE per dataset (rows) and method (columns).

    Parameters
    ----------
    predictions : pandas.DataFrame
        Columns ``dataset, method, step, actual, prediction``.

    Returns
    -------
    pandas.DataFrame
        RMSE; NaN where a method produced no forecast.
    """
    squared = (predictions["actual"] - predictions["prediction"]) ** 2
    grouped = squared.groupby([predictions["dataset"], predictions["method"]]).mean()
    return np.sqrt(grouped).unstack("method")


def mase_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and median absolute scaled error per dataset and method.

    Each error is divided by the mean absolute one-step change of the
    series' training part, so the error reads in units of a naive one-step
    forecast's error (Hyndman and Koehler 2006).

    Parameters
    ----------
    predictions : pandas.DataFrame
        Columns ``dataset, method, step, actual, prediction``.

    Returns
    -------
    mase : pandas.DataFrame
        Mean absolute scaled error, datasets in rows and methods in columns.
    mdase : pandas.DataFrame
        Median absolute scaled error, same layout.
    """
    table = pd.read_parquet(HERE / "data" / "series.parquet")
    train = table[table["split"] == "train"].sort_values(["dataset", "step"])
    scale = train.groupby("dataset")["y"].apply(lambda y: y.diff().abs().mean())
    scaled = (predictions["actual"] - predictions["prediction"]).abs() / predictions[
        "dataset"
    ].map(scale)
    grouped = scaled.groupby([predictions["dataset"], predictions["method"]])
    return grouped.mean().unstack("method"), grouped.median().unstack("method")


def ape_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mean and median absolute percentage error per dataset and method.

    ``100 * |actual - prediction| / |actual|``. A zero actual value has no
    percentage error: it is left out of both statistics and counted.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Columns ``dataset, method, step, actual, prediction``.

    Returns
    -------
    mape : pandas.DataFrame
        Mean APE in per cent, datasets in rows and methods in columns.
    mdape : pandas.DataFrame
        Median APE in per cent, same layout.
    undefined : pandas.DataFrame
        Number of points whose actual value is zero, same layout.
    """
    actual = predictions["actual"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = 100.0 * (actual - predictions["prediction"]).abs() / actual.abs()
    ape = ape.where(actual != 0)
    keys = [predictions["dataset"], predictions["method"]]
    grouped = ape.groupby(keys)
    return (
        grouped.mean().unstack("method"),
        grouped.median().unstack("method"),
        ape.isna().groupby(keys).sum().unstack("method"),
    )


def published() -> pd.DataFrame:
    """Read the 2014 predictions, as published.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, method, step, actual, prediction``.
    """
    return pd.read_parquet(HERE / "published" / "predictions_2014.parquet")


def paper_check() -> float:
    """Recompute the paper's table and return the largest deviation.

    Returns
    -------
    float
        Maximum absolute difference with the published standardised RMSE.
    """
    rmse = rmse_table(published())[list(PAPER_COLUMNS.values())]
    standardised = rmse.div(rmse.min(axis=1), axis=0)
    paper = pd.read_parquet(
        HERE / "published" / "paper_table_standardised_rmse.parquet"
    ).set_index("dataset")
    paper.columns = [PAPER_COLUMNS[c] for c in paper.columns]
    paper = paper.loc[standardised.index, standardised.columns]
    return float(np.abs(standardised.to_numpy() - paper.to_numpy()).max())


def convert_legacy_csv() -> list[str]:
    """Turn result CSV files left by earlier runs into Parquet.

    Returns
    -------
    list of str
        Methods converted.

    Raises
    ------
    RuntimeError
        When a Parquet copy does not reproduce its CSV exactly.
    """
    converted = []
    for csv in sorted(RESULTS.glob("*.csv")):
        if csv.stem in ("scores", "summary"):
            csv.unlink()
            continue
        try:
            frame = pd.read_csv(csv, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            # A method that failed on every series leaves a blank file.
            continue
        target = csv.with_suffix(".parquet")
        frame.to_parquet(target, index=False)
        back = pd.read_parquet(target)
        if not back.equals(frame):
            raise RuntimeError(f"{csv.name}: Parquet copy differs")
        csv.unlink()
        converted.append(csv.stem)
    return converted


def methods_2026() -> tuple[pd.DataFrame, dict[str, str], dict[str, dict]]:
    """Read every 2026 result file and attach the held-out values.

    Returns
    -------
    predictions : pandas.DataFrame
        Columns ``dataset, method, step, actual, prediction, lower, upper``.
    families : dict
        Method to family.
    meta : dict
        Method to its JSON record (time, failures).
    """
    actual = test_values()
    frames, families, meta = [], {}, {}
    for path in sorted(RESULTS.glob("*.json")):
        if path.stem in ("summary",):
            continue
        record = json.loads(path.read_text())
        table = path.with_suffix(".parquet")
        if not table.exists():
            continue
        frame = pd.read_parquet(table)
        if frame.empty:
            continue
        frame["method"] = path.stem
        frames.append(frame.merge(actual, on=["dataset", "step"]))
        families[path.stem] = record["family"]
        meta[path.stem] = record
    return pd.concat(frames, ignore_index=True), families, meta


def main(argv: Sequence[str] | None = None) -> int:
    """Score everything and write the tables.

    Parameters
    ----------
    argv : sequence of str, optional
        Unused; kept for a uniform entry point.

    Returns
    -------
    int
        0 on success, 1 if the paper's table is not reproduced.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    deviation = paper_check()
    LOGGER.info("paper table reproduced to within %.2e", deviation)
    if deviation > 1e-6:
        return 1

    for name in convert_legacy_csv():
        LOGGER.info("converted results/%s.csv to Parquet", name)
    new, families, meta = methods_2026()
    families = {**FAMILIES_2014, **families}
    combined = pd.concat([published(), new], ignore_index=True)
    rmse = rmse_table(combined)
    complete = rmse.columns[rmse.notna().all()]
    standardised = rmse[complete].div(rmse[complete].min(axis=1), axis=0)

    scores = standardised.stack().rename("standardised_rmse").reset_index()
    scores["rmse"] = [
        rmse.loc[d, m] for d, m in zip(scores["dataset"], scores["method"], strict=True)
    ]
    scores.to_parquet(RESULTS / "scores.parquet", index=False)

    interval = (
        combined.dropna(subset=["lower", "upper"])
        if "lower" in combined
        else pd.DataFrame()
    )
    coverage = (
        (
            (interval["actual"] >= interval["lower"])
            & (interval["actual"] <= interval["upper"])
        )
        .groupby(interval["method"])
        .mean()
        if not interval.empty
        else pd.Series(dtype=float)
    )
    mape, mdape, undefined = ape_tables(combined)
    mase, mdase = mase_tables(combined)
    columns = standardised.columns
    summary = pd.DataFrame(
        {
            "family": [families.get(m, "") for m in columns],
            "median_standardised_rmse": standardised.median().to_numpy(),
            "mean_standardised_rmse": standardised.mean().to_numpy(),
            "series_won": (standardised == 1.0).sum().to_numpy(),
            "coverage90": [coverage.get(m, np.nan) for m in columns],
            # Percentage errors: per-series MAPE and MdAPE, then the median
            # and the mean of those across series.
            "median_mase": mase[columns].median().to_numpy(),
            "mean_mase": mase[columns].mean().to_numpy(),
            "median_mdase": mdase[columns].median().to_numpy(),
            "median_mape": mape[columns].median().to_numpy(),
            "mean_mape": mape[columns].mean().to_numpy(),
            "median_mdape": mdape[columns].median().to_numpy(),
            "ape_undefined": undefined[columns].sum().astype(int).to_numpy(),
        },
        index=standardised.columns,
    ).sort_values("median_standardised_rmse")
    summary.rename_axis("method").reset_index().to_parquet(
        RESULTS / "summary.parquet", index=False
    )
    incomplete = sorted(set(rmse.columns) - set(complete))
    lines = [
        "# Extrapolation benchmark results",
        "",
        f"Paper table reproduced from the published 2014 predictions to within {deviation:.1e}.",
        "",
        "Standardised RMSE: each series' RMSE divided by the best RMSE on that",
        "series among every method listed, so 1.00 is the best. Median over the",
        f"{len(standardised)} series; lower is better.",
        "",
        "MASE and MdASE: mean and median absolute error of each series scaled",
        "by the mean absolute one-step change of its training part; MASE is",
        "their median (and mean MASE their mean) across series; lower is better.",
        "",
        "MAPE and MdAPE: mean and median absolute percentage error of each",
        "series, in per cent, then their median (MAPE, MdAPE) and mean (mean",
        "MAPE) across series; lower is better. A zero actual value has no",
        "percentage error and is left out.",
        "",
        "| method | family | median | mean | series won | 90 % coverage "
        "| MASE | mean MASE | MdASE | MAPE | mean MAPE | MdAPE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in summary.iterrows():
        cov = "" if np.isnan(row["coverage90"]) else f"{row['coverage90']:.2f}"
        lines.append(
            f"| {name} | {row['family']} | {row['median_standardised_rmse']:.2f} | "
            f"{row['mean_standardised_rmse']:.2f} | {int(row['series_won'])} | {cov} "
            f"| {row['median_mase']:.2f} | {row['mean_mase']:.2f} "
            f"| {row['median_mdase']:.2f} "
            f"| {row['median_mape']:.1f} | {row['mean_mape']:.1f} "
            f"| {row['median_mdape']:.1f} |"
        )
    if incomplete:
        lines += [
            "",
            "Left out because they failed on at least one series: "
            + ", ".join(incomplete)
            + ".",
        ]
        for name in incomplete:
            for dataset, reason in meta.get(name, {}).get("failures", {}).items():
                lines.append(f"- {name} on {dataset}: {reason}")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    LOGGER.info("\n%s", summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
