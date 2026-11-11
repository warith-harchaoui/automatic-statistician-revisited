"""Tests of the business context and of the features read against it.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import pytest

from autostat import benchmarks, describe
from autostat import kernels as ff
from autostat.context import Context, decimal_year, years
from autostat.features import kind, switches


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1y", 1.0), ("12mo", 1.0), ("7d", 7 / 365.25), ("24h", 24 / 8766), (0.5, 0.5)],
)
def test_durations_read_in_years(text, expected):
    """A period with a suffix becomes years, the time axis of the reports."""
    assert years(text) == pytest.approx(expected)


def test_an_unreadable_duration_says_how_to_write_it():
    """A typo fails loudly rather than becoming a wrong period."""
    with pytest.raises(ValueError, match="7d"):
        years("seven days")


def test_dates_read_as_decimal_years():
    """An ISO date lands on the decimal-year axis."""
    assert decimal_year("2020-01-01") == 2020.0
    assert decimal_year("2021-07-02") == pytest.approx(2021.5, abs=0.002)


def test_a_context_reads_from_a_mapping():
    """Cycles and events keep their names, periods in years, dates in years."""
    context = Context.from_dict(
        {
            "name": "sales",
            "positive": True,
            "cycles": [{"name": "weekly", "period": "7d", "tolerance": 0.03}],
            "events": [{"name": "Covid lockdown", "at": "2020-03-17"}],
        }
    )
    assert context.positive
    assert context.cycles[0].period == pytest.approx(7 / 365.25)
    assert context.cycles[0].tolerance == 0.03
    assert context.events[0].at == pytest.approx(2020.208, abs=0.001)


def test_components_are_told_apart_by_their_base_kernels():
    """A growing cycle is a cycle, a slope a trend, white noise noise."""
    assert kind(ff.ProductKernel([ff.LinearKernel(), ff.PeriodicKernel()])) == "cycle"
    assert kind(ff.LinearKernel()) == "trend"
    assert kind(ff.SqExpKernel()) == "smooth"
    assert kind(ff.NoiseKernel()) == "noise"


def test_a_change_window_switches_at_its_two_edges():
    """An event matches the date something changed, not the window's middle."""
    window = ff.ChangeWindowKernel(
        0,
        location=1680.0,
        steepness=1.0,
        width=4.29,
        operands=[ff.ConstKernel(), ff.NoiseKernel()],
    )
    low, high = switches(window)
    assert low < 1680 < high
    assert high - low == pytest.approx(73.0, abs=0.1)


@pytest.fixture(scope="module")
def airline_report():
    """The published 2014 airline model, described without its checks."""
    x, y = benchmarks.load("01-airline")
    return describe(
        benchmarks.published("01-airline")[-1], x, y, name="airline", check=False
    )


def test_the_airline_features_find_the_yearly_cycle(airline_report):
    """The expected yearly cycle is present, a quarter of the level in size."""
    from autostat.features import features

    context = Context.from_dict(
        {
            "positive": True,
            "cycles": [
                {"name": "yearly", "period": "1y"},
                {"name": "weekly", "period": "7d"},
            ],
        }
    )
    row = features(airline_report, context)
    assert row["cycle_yearly_present"] is True
    assert row["cycle_yearly_period"] == pytest.approx(1.0, abs=0.01)
    assert 20 < row["cycle_yearly_size_pct"] < 35
    assert row["cycle_weekly_present"] is False
    assert row["cycle_weekly_period"] is None
    assert row["unexpected_cycles"] == 0
    shares = [row[f"share_{k}"] for k in ("trend", "cycle", "smooth", "noise")]
    assert sum(s for s in shares if s is not None) == pytest.approx(100.0)
    assert all(s is None or s >= 0 for s in shares)


def test_without_context_only_the_generic_columns(airline_report):
    """No context, no per-cent sizes and no business columns."""
    from autostat.features import features

    row = features(airline_report)
    assert row["main_cycle_period"] == pytest.approx(1.0, abs=0.01)
    assert row["trend_slope_pct_per_unit"] is None
    assert not any(k.startswith(("cycle_", "event_")) for k in row)
