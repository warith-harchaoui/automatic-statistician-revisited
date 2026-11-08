"""Turning a pandas series into search inputs.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autostat import prepare


@pytest.fixture
def hourly() -> pd.Series:
    """Return 5,000 increasing hourly values from 2024.

    Returns
    -------
    pandas.Series
        Datetime-indexed, timezone-aware.
    """
    index = pd.date_range("2024-01-01", periods=5000, freq="h", tz="UTC")
    return pd.Series(np.arange(5000.0), index=index)


def test_a_datetime_index_becomes_decimal_years(hourly):
    """2024-01-01 maps near 2024 on the original's 365-day year.

    Parameters
    ----------
    hourly : pandas.Series
        Fixture: 5,000 hourly points, datetime-indexed.
    """
    x, _, _ = prepare(hourly, max_points=500)
    assert x[0, 0] == pytest.approx(2024.0, abs=0.05)


def test_a_datetime_index_reports_the_year_unit(hourly):
    """The report then speaks in years, months and days.

    Parameters
    ----------
    hourly : pandas.Series
        Fixture: 5,000 hourly points, datetime-indexed.
    """
    assert prepare(hourly)[2] == "year"


def test_a_long_series_is_averaged_into_max_points_bins(hourly):
    """5,000 points become 500 bins.

    Parameters
    ----------
    hourly : pandas.Series
        Fixture: 5,000 hourly points, datetime-indexed.
    """
    x, y, _ = prepare(hourly, max_points=500)
    assert (x.shape, y.shape) == ((500, 1), (500,))


def test_binning_keeps_an_increasing_series_increasing(hourly):
    """Bin means preserve order.

    Parameters
    ----------
    hourly : pandas.Series
        Fixture: 5,000 hourly points, datetime-indexed.
    """
    assert np.all(np.diff(prepare(hourly, max_points=500)[1]) > 0)


def test_missing_values_are_dropped():
    """NaN values never reach the search."""
    series = pd.Series([1.0, np.nan, 3.0, 4.0], index=[0.0, 1.0, 2.0, 3.0])
    assert len(prepare(series)[1]) == 3


def test_a_series_with_two_values_is_refused():
    """Three finite values is the minimum."""
    with pytest.raises(ValueError, match="three"):
        prepare(pd.Series([1.0, 2.0]))
