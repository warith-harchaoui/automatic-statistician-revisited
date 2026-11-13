"""Tests of the business context and of the features read against it.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math

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
    """A period with a suffix becomes years, the time axis of the reports.

    Parameters
    ----------
    text : str or float
        The duration as written.
    expected : float
        Its length in years.
    """
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
    """The published 2014 airline model, described without its checks.

    Returns
    -------
    Report
        Its description, data included.
    """
    x, y = benchmarks.load("01-airline")
    return describe(
        benchmarks.published("01-airline")[-1], x, y, name="airline", check=False
    )


def test_the_airline_features_find_the_yearly_cycle(airline_report):
    """The expected yearly cycle is present, a quarter of the level in size.

    Parameters
    ----------
    airline_report : Report
        The fixture.
    """
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
    """No context, no per-cent sizes and no business columns.

    Parameters
    ----------
    airline_report : Report
        The fixture.
    """
    from autostat.features import features

    row = features(airline_report)
    assert row["main_cycle_period"] == pytest.approx(1.0, abs=0.01)
    assert row["trend_slope_pct_per_unit"] is None
    assert not any(k.startswith(("cycle_", "event_")) for k in row)


def test_two_equally_good_structures_split_the_vote():
    """A periodic and a smooth model with the same BIC give a coin flip.

    Two models of three parameters each, fitted equally well: the data do
    not choose, so the probability of a cycle is one half.
    """
    from autostat.features import structure_probabilities
    from autostat.model import GPModel

    cycle = ff.PeriodicKernel(0, lengthscale=0.0, period=0.0, sf=0.0)
    smooth = ff.RQKernel(0, lengthscale=0.0, sf=0.0, alpha=0.0)
    models = [GPModel(cycle, nll=10.0, ndata=100), GPModel(smooth, nll=10.0, ndata=100)]
    context = Context.from_dict({"cycles": [{"name": "yearly", "period": "1y"}]})
    row = structure_probabilities(models, context)
    assert row["p_periodic"] == pytest.approx(0.5)
    assert row["p_cycle_yearly"] == pytest.approx(0.5)
    assert row["p_entropy"] == pytest.approx(math.log(2))


def test_the_published_airline_search_is_sure_of_its_yearly_cycle():
    """Every structure that counts in the 2014 airline search has the cycle."""
    from autostat.features import structure_probabilities

    context = Context.from_dict({"cycles": [{"name": "yearly", "period": "1y"}]})
    row = structure_probabilities(benchmarks.published("01-airline"), context)
    assert row["p_cycle_yearly"] == pytest.approx(1.0, abs=1e-6)
    assert row["p_changepoint"] < 1e-6


def test_the_call_centre_drop_is_measured_and_a_placebo_is_not():
    """The 1973 drop is negative with an interval clear of 0; 1968 is not."""
    from autostat.features import event_effect

    x, y = benchmarks.load("07-call-centre")
    model = benchmarks.published("07-call-centre")[-1]
    drop = event_effect(model, x, y, 1973.36, 1.0)
    placebo = event_effect(model, x, y, 1968.0, 1.0)
    assert drop["high"] < 0
    assert placebo["low"] < 0 < placebo["high"]
    assert event_effect(model, x, y, x.min() + 0.1, 1.0) is None


def test_a_component_with_no_signal_is_described():
    """A component whose posterior mean vanishes no longer crashes describe."""
    import numpy as np

    from autostat.model import GPModel

    t = np.linspace(2000, 2010, 60)
    y = np.sin(2 * np.pi * t) + 0.1 * np.cos(7 * t)
    kernel = ff.SumKernel(
        [
            ff.PeriodicKernel(0, lengthscale=0.0, period=0.0, sf=0.0),
            ff.ChangePointKernel(
                0,
                location=2050.0,
                steepness=3.0,
                operands=[ff.ZeroKernel(), ff.NoiseKernel(sf=-2.0)],
            ),
        ]
    )
    report = describe(
        GPModel(kernel, nll=0.0, ndata=60), t, y, name="flat", check=False
    )
    assert len(report.stats) == len(report.parts)


def test_a_known_event_adds_a_level_shift_to_each_kept_model():
    """With an event date, each kept model is also proposed with a shift.

    The fitted terms keep their parameters; a model that already has as
    many shifts as there are events gets no more; without dates, nothing.
    """
    from autostat import Experiment
    from autostat.model import GPModel
    from autostat.search import _level_shifts

    fitted = GPModel(
        ff.SumKernel(
            [ff.LinearKernel(0, location=2010.0, sf=0.5), ff.NoiseKernel(sf=0.1)]
        ),
        nll=10.0,
        ndata=100,
    )
    assert _level_shifts([fitted], Experiment()) == []
    (shifted,) = _level_shifts([fitted], Experiment(location_hints=[2014.5]))
    assert shifted.kernel.structure_key() == "CP(Zero, C) + Lin + WN"
    assert shifted.kernel.operands[0].location == 2010.0
    assert _level_shifts([shifted], Experiment(location_hints=[2014.5])) == []


@pytest.mark.slow
def test_a_short_search_finds_the_jump_on_the_date_it_is_told():
    """One level is too few for the grammar; the last stage adds the shift."""
    import numpy as np

    from autostat import Experiment, search

    rng = np.random.default_rng(0)
    t = 2010 + np.arange(60) / 12
    y = 1.0 * (t - 2010) + 6 * (t > 2012.5) + rng.normal(0, 0.5, t.size)
    exp = Experiment.abcd_2014(
        max_depth=1, n_rand=3, iters=100, n_jobs=1, location_hints=[2012.5]
    )
    best = search(t, y, exp).best
    (shift,) = [k for k in best.kernel.operands if isinstance(k, ff.ChangePointKernel)]
    assert isinstance(shift.operands[0], ff.ZeroKernel)
    assert abs(shift.location - 2012.5) < 0.15


def test_two_vanishing_constants_merge_without_a_domain_error():
    """Constants too small for exp still add up, on the log scale."""
    kernel = ff.SumKernel(
        [ff.ConstKernel(sf=-400.0), ff.ConstKernel(sf=-400.0), ff.NoiseKernel(sf=0.0)]
    )
    (const,) = [
        k
        for k in kernel.collapse_additive_idempotency().operands
        if isinstance(k, ff.ConstKernel)
    ]
    assert const.sf == pytest.approx(-400.0 + 0.5 * math.log(2))
