"""Tests of the break detector and of the shifts it proposes to the search.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autostat import Experiment, search
from autostat import kernels as ff
from autostat.breaks import detect, proposals, shift
from autostat.model import GPModel
from autostat.propose import parse
from autostat.search import data_shape


def monthly(jump: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw ten years with a trend, a yearly cycle and a jump in mid-2014.

    Parameters
    ----------
    jump : float
        Size of the jump, in noise units.
    seed : int, optional
        Random seed of the noise.

    Returns
    -------
    tuple of numpy.ndarray
        Times in decimal years and values.
    """
    t = 2010 + np.arange(120) / 12
    noise = np.random.default_rng(seed).normal(0, 1.0, t.size)
    y = 20 + (t - 2010) + jump * (t > 2014.5) + 3 * np.sin(2 * np.pi * t) + noise
    return t, y


@pytest.mark.parametrize("seed", range(3))
def test_the_jump_is_found_and_the_cycle_is_not_cut(seed):
    """With the cycle in the cost, one break at the jump and none without one.

    Parameters
    ----------
    seed : int
        Random seed of the noise.
    """
    t, y = monthly(8.0, seed)
    assert detect(t, y, 3, period=1.0) == [pytest.approx(2014.54, abs=0.01)]
    t, y = monthly(0.0, seed)
    assert detect(t, y, 3, period=1.0) == []


def test_breaks_come_strongest_first_and_at_most_count():
    """Two steps are ranked by size; ``count`` caps the list."""
    x = np.arange(300.0)
    y = 5.0 * (x >= 100) + 2.0 * (x >= 200)
    y += np.random.default_rng(1).normal(0, 0.3, x.size)
    assert detect(x, y, 5) == [pytest.approx(99.5), pytest.approx(199.5)]
    assert detect(x, y, 1) == [pytest.approx(99.5)]
    assert detect(x, y, 0) == []
    assert detect(x[:5], y[:5], 3) == []


def test_the_search_reads_the_breaks_only_when_asked():
    """``break_starts`` fills the data summary; the presets leave it empty."""
    t, y = monthly(8.0)
    blind = data_shape(t[:, None], y, Experiment.abcd_2014())
    asked = data_shape(t[:, None], y, Experiment.abcd_2014(break_starts=2))
    assert (blind["break_hints"], blind["break_period"]) == ([], None)
    assert asked["break_hints"] == [pytest.approx(2014.54, abs=0.01)]
    assert asked["break_period"] == pytest.approx(1.0, abs=0.02)


def test_a_shift_is_proposed_once_per_date():
    """The detector's model first; then the best model plus a shift, if new."""
    t, y = monthly(8.0)
    shape = data_shape(t[:, None], y, Experiment.abcd_2014(break_starts=1))
    first = proposals([], shape, 0)
    assert [k.structure_key() for k in first] == ["CP(Zero, C) + Lin + Per + WN"]
    switch = next(k for k in first[0].operands if isinstance(k, ff.ChangePointKernel))
    assert switch.location == shape["break_hints"][0]
    best = GPModel(parse("Lin + SE + WN"))
    assert [k.structure_key() for k in proposals([best], shape, 1)] == [
        "CP(Zero, C) + Lin + SE + WN"
    ]
    held = GPModel(ff.SumKernel([parse("SE + WN"), shift(2014.5)]).canonical())
    assert proposals([held], shape, 1) == []


def test_a_detected_break_is_fitted_where_it_is():
    """A search asked for breaks keeps a shift at the jump's date."""
    t, y = monthly(8.0)
    exp = Experiment.abcd_2014(
        max_depth=1, n_jobs=1, n_rand=3, iters=50, break_starts=1
    )
    result = search(t, y, exp)
    locations = [
        k.location
        for m in result.levels[0]
        for k in [m.kernel, *getattr(m.kernel, "operands", [])]
        if isinstance(k, ff.ChangePointKernel)
    ]
    assert any(abs(t0 - 2014.5) < 0.1 for t0 in locations)
    assert math.isfinite(result.best.bic)
