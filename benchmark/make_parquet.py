#!/usr/bin/env python3
"""Rebuild the benchmark's Parquet files from a gpss-research checkout.

    git clone https://github.com/jamesrobertlloyd/gpss-research
    python benchmark/make_parquet.py gpss-research            # check, then write
    python benchmark/make_parquet.py gpss-research --record   # new hashes

What this script does
    The benchmark reads three Parquet files, all derived from the original
    repository (commit ``2a64958a018f1668f7b8eedf33c4076a63af7868``, MIT
    licence):

    * ``data/series.parquet``: the thirteen 90/10 splits of
      ``data/tsdlr_9010/<name>-{train,test}.csv``, long format with columns
      ``dataset, split, step, x, y``;
    * ``published/predictions_2014.parquet``: the forecasts the 2014 methods
      made, read from ``results/<folder>/<name>_predictions.mat``, with
      columns ``dataset, method, step, actual, prediction``;
    * ``published/paper_table_standardised_rmse.parquet``: the table behind
      the paper's box plot, ``source/R/postprocessing/extrap_sop_only.csv``,
      with a ``dataset`` column added.

    Before writing anything it checks the SHA-256 of every upstream file it
    reads against ``published/provenance.json``, so a changed upstream file
    stops the rebuild instead of silently changing the benchmark.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("make_parquet")
HERE = Path(__file__).resolve().parent
PROVENANCE = HERE / "published" / "provenance.json"

#: Upstream commit the hashes were recorded from.
COMMIT = "2a64958a018f1668f7b8eedf33c4076a63af7868"

#: Published method name to the upstream results folder holding its forecasts.
#: Each folder was matched to a column of the paper by recomputing it.
FOLDERS = {
    "ABCD-2014": "2013-12-10-extrap-GPSS",
    "SP": "2013-12-17-extrap-SP-bic",
    "TCI": "2013-11-11-extrap-TCI",
    "MKL": "2013-11-11-extrap-MKL",
    "Eureqa": "eureqa-extrap",
    "CP": "2013-11-11-extrap-CP",
    "SE": "2014-01-28-extrap-SE",
    "LN": "2014-01-28-extrap-LN",
    "CKS-2013": "2013-11-08-extrap-GPSS",
}

SPLITS = "data/tsdlr_9010"
PAPER_TABLE = "source/R/postprocessing/extrap_sop_only.csv"


def sha256(path: Path) -> str:
    """Hash a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Hexadecimal SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_files(root: Path) -> list[str]:
    """List every upstream file the rebuild reads, relative to ``root``.

    Parameters
    ----------
    root : pathlib.Path
        The gpss-research checkout.

    Returns
    -------
    list of str
        Sorted relative paths.
    """
    files = [p.relative_to(root) for p in (root / SPLITS).glob("*-t*.csv")]
    for folder in FOLDERS.values():
        files += [
            p.relative_to(root)
            for p in (root / "results" / folder).glob("*_predictions.mat")
        ]
    files.append(Path(PAPER_TABLE))
    return sorted(str(f) for f in files)


def series_frame(root: Path) -> pd.DataFrame:
    """Read the thirteen splits.

    Parameters
    ----------
    root : pathlib.Path
        The gpss-research checkout.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, split, step, x, y``, float64 values exactly as
        ``numpy.loadtxt`` parses them.
    """
    frames = []
    for path in sorted((root / SPLITS).glob("*-train.csv")):
        name = path.name.removesuffix("-train.csv")
        for split in ("train", "test"):
            values = np.loadtxt(
                root / SPLITS / f"{name}-{split}.csv", delimiter=",", ndmin=2
            )
            frames.append(
                pd.DataFrame(
                    {
                        "dataset": name,
                        "split": split,
                        "step": np.arange(len(values)),
                        "x": values[:, 0],
                        "y": values[:, 1],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def predictions_frame(root: Path) -> pd.DataFrame:
    """Read the 2014 forecasts from the MATLAB result files.

    Parameters
    ----------
    root : pathlib.Path
        The gpss-research checkout.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, method, step, actual, prediction``.
    """
    from scipy.io import loadmat

    rows = []
    for method, folder in FOLDERS.items():
        for path in sorted((root / "results" / folder).glob("*_predictions.mat")):
            mat = loadmat(path)
            name = path.name.removesuffix("_predictions.mat")
            pairs = zip(mat["actuals"].ravel(), mat["predictions"].ravel(), strict=True)
            for step, (actual, prediction) in enumerate(pairs):
                rows.append(
                    {
                        "dataset": name,
                        "method": method,
                        "step": step,
                        "actual": float(actual),
                        "prediction": float(prediction),
                    }
                )
    return pd.DataFrame(rows)


def paper_frame(root: Path, datasets: list[str]) -> pd.DataFrame:
    """Read the paper's standardised RMSE table.

    Parameters
    ----------
    root : pathlib.Path
        The gpss-research checkout.
    datasets : list of str
        Dataset names in sorted order, the order of the table's rows.

    Returns
    -------
    pandas.DataFrame
        One row per dataset, one column per method of the paper.
    """
    table = pd.read_csv(root / PAPER_TABLE)
    table.insert(0, "dataset", datasets)
    return table


def main(argv: Sequence[str] | None = None) -> int:
    """Check the upstream hashes, then write the Parquet files.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 on success, 1 when an upstream file is missing or changed.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkout", type=Path, help="gpss-research checkout")
    parser.add_argument(
        "--record", action="store_true", help="write the hashes instead of checking"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    root = args.checkout.resolve()

    hashes = {name: sha256(root / name) for name in upstream_files(root)}
    if args.record:
        PROVENANCE.write_text(
            json.dumps({"commit": COMMIT, "sha256": hashes}, indent=2) + "\n"
        )
        LOGGER.info("recorded %d hashes", len(hashes))
    else:
        expected = json.loads(PROVENANCE.read_text())["sha256"]
        bad = sorted(name for name in expected if hashes.get(name) != expected[name])
        if bad:
            LOGGER.error("missing or changed upstream files: %s", ", ".join(bad))
            return 1
        LOGGER.info("all %d upstream hashes match", len(expected))

    series = series_frame(root)
    series.to_parquet(HERE / "data" / "series.parquet", index=False)
    predictions = predictions_frame(root)
    predictions.to_parquet(HERE / "published" / "predictions_2014.parquet", index=False)
    datasets = sorted(series["dataset"].unique())
    paper_frame(root, datasets).to_parquet(
        HERE / "published" / "paper_table_standardised_rmse.parquet", index=False
    )
    LOGGER.info(
        "wrote %d series rows, %d prediction rows", len(series), len(predictions)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
