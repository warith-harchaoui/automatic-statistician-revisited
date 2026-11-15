"""Tests of the forecast averaged over structures, weighted by BIC.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest

from autostat import benchmarks, mixture_posterior, posterior
from autostat import kernels as ff
from autostat.model import GPModel, bic_weights


def _data():
    """Draw a short yearly cycle with a little noise.

    Returns
    -------
    x, y : numpy.ndarray
        Inputs in years and targets.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(2000, 2004, 48)
    return x, np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(48)


def _cycle(nll: float) -> GPModel:
    """Build a periodic model with a fixed score.

    Parameters
    ----------
    nll : float
        Its negative log likelihood.

    Returns
    -------
    GPModel
        A yearly cycle.
    """
    return GPModel(
        ff.PeriodicKernel(0, lengthscale=0.0, period=0.0, sf=0.0), nll=nll, ndata=48
    )


def _smooth(nll: float) -> GPModel:
    """Build a smooth model with the same number of parameters.

    Parameters
    ----------
    nll : float
        Its negative log likelihood.

    Returns
    -------
    GPModel
        A rational quadratic.
    """
    return GPModel(
        ff.RQKernel(0, lengthscale=0.0, sf=0.0, alpha=0.0), nll=nll, ndata=48
    )


def test_one_structure_forecasts_as_the_best_model():
    """With one structure the mixture is that structure's posterior."""
    x, y = _data()
    grid = np.linspace(2004, 2005, 12)
    model = _cycle(10.0)
    mean, var = mixture_posterior([model], x, y, grid, noise=True)
    best_mean, best_var = posterior(model, x, y, grid, noise=True)
    np.testing.assert_allclose(mean, best_mean)
    np.testing.assert_allclose(var, best_var)


def test_disagreeing_structures_widen_the_band():
    """Two structures the data cannot tell apart: average mean, wider variance.

    The law of total variance adds the spread of the two means to the
    average of their variances.
    """
    x, y = _data()
    grid = np.linspace(2004, 2006, 24)
    cycle, smooth = _cycle(10.0), _smooth(10.0)
    mean, var = mixture_posterior([cycle, smooth], x, y, grid)
    m1, v1 = posterior(cycle, x, y, grid)
    m2, v2 = posterior(smooth, x, y, grid)
    np.testing.assert_allclose(mean, (m1 + m2) / 2)
    np.testing.assert_allclose(var, (v1 + v2) / 2 + ((m1 - m2) / 2) ** 2)
    assert np.all(var >= (v1 + v2) / 2 - 1e-12)


def test_a_structure_refitted_twice_votes_once_with_its_better_fit():
    """The same structure found at two levels keeps its lower BIC only."""
    chosen, weights = bic_weights([_cycle(12.0), _cycle(10.0), _smooth(10.0)])
    assert len(chosen) == 2
    assert {m.nll for m in chosen} == {10.0}
    np.testing.assert_allclose(weights, [0.5, 0.5])


def test_a_negligible_structure_is_left_out():
    """A structure 40 nats behind weighs less than the floor and is dropped."""
    chosen, weights = bic_weights([_cycle(10.0), _smooth(50.0)], floor=1e-4)
    assert len(chosen) == 1
    assert weights == pytest.approx([1.0])


def test_no_finite_score_fails_loudly():
    """A list with no usable model says so rather than returning nothing."""
    with pytest.raises(ValueError, match="finite BIC"):
        mixture_posterior([], *_data())


def test_the_published_airline_search_averages_to_a_finite_forecast():
    """The 2014 airline levels mix into a finite forecast near the best one."""
    x, y = benchmarks.load("01-airline")
    models = benchmarks.published("01-airline")
    grid = np.linspace(x.max(), x.max() + 2, 24)
    mean, var = mixture_posterior(models, x, y, grid, noise=True)
    best = min(models, key=lambda m: m.bic)
    best_mean, _ = posterior(best, x, y, grid, noise=True)
    assert np.all(np.isfinite(mean)) and np.all(var > 0)
    assert np.max(np.abs(mean - best_mean)) < 0.1 * np.ptp(y)
