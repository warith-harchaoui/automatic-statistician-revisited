"""The search finds structure, refuses bad input and keeps both presets.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest

from autostat import Experiment, search
from autostat import kernels as ff
from autostat.model import GPModel, fit
from conftest import nll_at_published


def trend_plus_cycle() -> tuple[np.ndarray, np.ndarray]:
    """Simulate a noisy yearly sinusoid on a linear trend.

    Returns
    -------
    tuple of numpy.ndarray
        Ten years of 120 monthly points.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(2000, 2010, 120)
    y = 0.3 * (x - 2000) + np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size)
    return x, y


def test_fitting_lowers_the_negative_log_likelihood(airline):
    """The optimiser improves on its starting point."""
    x, y = airline
    start = GPModel(
        ff.SumKernel(
            [
                ff.NoiseKernel(sf=0.0),
                ff.SqExpKernel(dimension=0, lengthscale=0.0, sf=0.0),
            ]
        )
    )
    assert fit(start, x, y, iters=50, subset=False).nll < nll_at_published(start, x, y)


def test_search_refuses_inputs_and_targets_of_different_lengths():
    """Mismatched data fail before any fitting."""
    with pytest.raises(ValueError, match="rows"):
        search(np.arange(5.0), np.arange(4.0))


def test_search_refuses_missing_values():
    """NaN in the targets fails before any fitting."""
    with pytest.raises(ValueError, match="finite"):
        search(np.arange(5.0), np.array([1.0, 2.0, np.nan, 4.0, 5.0]))


def test_the_2014_preset_equals_the_defaults():
    """abcd_2014() is the published configuration, which the defaults encode."""
    assert Experiment.abcd_2014() == Experiment()


def test_the_2013_preset_models_noise_in_the_likelihood():
    """cks_2013() puts observation noise in the likelihood, as the 2013 code did."""
    assert Experiment.cks_2013().likelihood_noise


@pytest.mark.slow
def test_the_2014_search_finds_the_yearly_cycle():
    """Three levels under the published settings find a periodic component.

    With three random starts instead of nine the search stops at WN + SE
    (BIC about -8) although WN + Per + Lin reaches -168: the period is the
    parameter that needs a lucky start, which is why the original paid for
    nine.
    """
    x, y = trend_plus_cycle()
    result = search(x, y, Experiment.abcd_2014(max_depth=3, n_jobs=4, iters=100))
    assert "Per" in result.best.kernel.structure()


@pytest.mark.slow
def test_the_2013_search_never_builds_a_changepoint():
    """The 2013 grammar has no changepoint or change-window operator.

    On this series the 2013 preset stops at SE (BIC about -13) where the
    2014 preset finds the cycle: two random starts with a spread of 4 on the
    log scale rarely draw a period near one year. That is a property of the
    preset, documented in the README, not a defect this test hides.
    """
    x, y = trend_plus_cycle()
    result = search(x, y, Experiment.cks_2013(max_depth=2, n_jobs=4))
    structures = {m.kernel.structure() for level in result.levels for m in level}
    assert not any("CP" in s or "CW" in s for s in structures)
