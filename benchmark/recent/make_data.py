#!/usr/bin/env python3
"""Build a benchmark the pre-trained forecasters cannot have seen.

    python benchmark/recent/make_data.py

What this script does
    The thirteen series of 2014 have been public for decades, so a
    foundation model may have read their held-out tails while pretraining.
    This script builds a second benchmark whose tails lie in 2026, after the
    release of Chronos-2 (October 2025) and TimesFM 2.5 (September 2025).
    Each series is a public monthly statistic: twenty years of training,
    January 2006 to December 2025, then every month of 2026 published when
    the script runs. A series with fewer than six months of 2026 is left out.

    The series are mostly not seasonally adjusted, so their cycles are there
    to find: US prices, employment, housing, production and sales from FRED
    (the Federal Reserve Bank of St. Louis), then three scientific records,
    carbon dioxide at Mauna Loa (NOAA), the sunspot number (SILSO) and the
    global temperature anomaly (NASA GISTEMP).

    Every download goes through one HTTP session with a pause of 2 to 4
    seconds, drawn at random, between requests, with an honest User-Agent;
    the first refusal stops the script. Each source file's URL, retrieval
    date and SHA-256 go to ``provenance.json``.

What it produces
    ``benchmark/data/recent.parquet`` with columns ``dataset, split, step,
    x, y, season`` (``x`` in decimal years at mid-month) and
    ``benchmark/recent/provenance.json``.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "recent.parquet"
AGENT = "automatic-statistician-revisited benchmark builder (warithmetics@deraison.ai)"

#: FRED series, not seasonally adjusted unless the name says otherwise.
FRED = {
    "cpi-all-items": "CPIAUCNS",
    "unemployment-rate": "UNRATENSA",
    "nonfarm-payrolls": "PAYNSA",
    "housing-starts": "HOUSTNSA",
    "vehicle-sales": "TOTALNSA",
    "electricity-generation": "IPG2211A2N",
    "ice-cream-production": "IPN31152N",
    "retail-sales": "RSAFSNA",
    "motor-vehicle-production": "IPG3361T3N",
    "gasoline-price": "APU000074714",
    "crude-oil-price": "MCOILWTICO",
    "treasury-10-year": "GS10",
}
#: Prices set by markets, with no yearly cycle.
NO_SEASON = {"crude-oil-price", "treasury-10-year"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
SUNSPOT_URL = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

TRAIN_START, TEST_START = 2006, 2026
MIN_TEST = 6


class Refused(RuntimeError):
    """A source refused a request; nothing more is asked of any server."""


def fetch(session: requests.Session, url: str, record: list[dict]) -> str:
    """Download one file after a random pause, recording its provenance.

    Parameters
    ----------
    session : requests.Session
        The one session every request goes through.
    url : str
        The file.
    record : list of dict
        Provenance rows, appended to.

    Returns
    -------
    str
        The file's text.

    Raises
    ------
    Refused
        On any HTTP error or network failure.
    """
    time.sleep(random.uniform(2.0, 4.0))
    try:
        response = session.get(url, timeout=60)
    except requests.RequestException as error:
        raise Refused(f"{url}: {error}") from error
    if response.status_code != 200:
        raise Refused(f"{url}: HTTP {response.status_code}")
    record.append(
        {
            "url": url,
            "retrieved": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(response.content).hexdigest(),
        }
    )
    return response.text


def monthly(dates: pd.Series, values: pd.Series) -> pd.DataFrame:
    """Put a monthly series on decimal years at mid-month.

    Parameters
    ----------
    dates : pandas.Series
        Timestamps of the months.
    values : pandas.Series
        The values.

    Returns
    -------
    pandas.DataFrame
        Columns ``x`` and ``y``, months without a value dropped.
    """
    dates = pd.to_datetime(dates)
    frame = pd.DataFrame(
        {
            "x": dates.dt.year + (dates.dt.month - 0.5) / 12,
            "y": pd.to_numeric(values, errors="coerce"),
        }
    )
    return frame.dropna().reset_index(drop=True)


def read_fred(text: str) -> pd.DataFrame:
    """Read a FRED graph CSV (a date column, then the series)."""
    table = pd.read_csv(io.StringIO(text))
    return monthly(table.iloc[:, 0], table.iloc[:, 1])


def read_co2(text: str) -> pd.DataFrame:
    """Read NOAA's monthly Mauna Loa record (comment lines start with #)."""
    table = pd.read_csv(io.StringIO(text), comment="#")
    dates = pd.to_datetime(dict(year=table["year"], month=table["month"], day=1))
    return monthly(dates, table["average"])


def read_sunspots(text: str) -> pd.DataFrame:
    """Read SILSO's monthly total sunspot number (semicolon-separated)."""
    table = pd.read_csv(io.StringIO(text), sep=";", header=None)
    dates = pd.to_datetime(dict(year=table[0], month=table[1], day=1))
    values = table[3].where(table[3] >= 0)
    return monthly(dates, values)


def read_gistemp(text: str) -> pd.DataFrame:
    """Read GISTEMP's global anomaly table (a year per row, a month per column)."""
    table = pd.read_csv(io.StringIO(text), skiprows=1, na_values=["***", "****"])
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    long = table.melt(id_vars="Year", value_vars=months, var_name="m", value_name="y")
    long["month"] = long["m"].map({m: i + 1 for i, m in enumerate(months)})
    dates = pd.to_datetime(dict(year=long["Year"], month=long["month"], day=1))
    return monthly(dates, long["y"]).sort_values("x").reset_index(drop=True)


def split(name: str, frame: pd.DataFrame, season: int) -> pd.DataFrame | None:
    """Cut a series into its training years and its 2026 tail.

    Parameters
    ----------
    name : str
        Dataset name.
    frame : pandas.DataFrame
        Columns ``x``, ``y``.
    season : int
        Seasonal period in samples.

    Returns
    -------
    pandas.DataFrame or None
        Rows of ``recent.parquet``; ``None`` when 2026 holds fewer than
        :data:`MIN_TEST` months or the training years have gaps.
    """
    train = frame[(frame["x"] >= TRAIN_START) & (frame["x"] < TEST_START)]
    test = frame[frame["x"] >= TEST_START]
    if len(test) < MIN_TEST or len(train) != 12 * (TEST_START - TRAIN_START):
        print(f"left out {name}: {len(train)} training months, {len(test)} in 2026")
        return None
    rows = []
    for part, label in ((train, "train"), (test, "test")):
        for step, (x, y) in enumerate(zip(part["x"], part["y"], strict=True)):
            rows.append(
                {
                    "dataset": name,
                    "split": label,
                    "step": step,
                    "x": x,
                    "y": y,
                    "season": season,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    """Download every source, split it and write the benchmark.

    Returns
    -------
    int
        0 on success, 1 when a source refused (nothing is written).
    """
    session = requests.Session()
    session.headers["User-Agent"] = AGENT
    record: list[dict] = []
    frames = []
    try:
        for name, code in FRED.items():
            text = fetch(session, FRED_URL.format(code), record)
            frames.append(split(name, read_fred(text), 1 if name in NO_SEASON else 12))
        frames.append(
            split("co2-mauna-loa", read_co2(fetch(session, CO2_URL, record)), 12)
        )
        frames.append(
            split(
                "sunspot-number", read_sunspots(fetch(session, SUNSPOT_URL, record)), 1
            )
        )
        frames.append(
            split(
                "global-temperature",
                read_gistemp(fetch(session, GISTEMP_URL, record)),
                1,
            )
        )
    except Refused as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 1
    table = pd.concat([f for f in frames if f is not None], ignore_index=True)
    table.to_parquet(OUT, index=False)
    (HERE / "provenance.json").write_text(json.dumps(record, indent=2))
    tails = table[table["split"] == "test"].groupby("dataset").size()
    print(
        f"{tails.size} series; months of 2026 per series: {tails.min()} to {tails.max()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
