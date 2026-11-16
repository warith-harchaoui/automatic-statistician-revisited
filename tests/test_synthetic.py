"""Tests of the series drawn from the grammar.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autostat import kernels as ff
from autostat import synthetic as sy
from autostat.features import leaves

X = np.linspace(2000, 2010, 80)


def test_the_same_seed_draws_the_same_series():
    """A seed fixes the structure, its parameters and the values."""
    y1, k1 = sy.draw(X, np.random.default_rng(3))
    y2, k2 = sy.draw(X, np.random.default_rng(3))
    np.testing.assert_array_equal(y1, y2)
    assert k1.pretty_print() == k2.pretty_print()


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_a_structure_uses_only_the_base_kernels_asked_for(steps):
    """No base kernel outside the vocabulary appears, whatever the depth.

    Parameters
    ----------
    steps : int
        Grammar rewrites.
    """
    rng = np.random.default_rng(steps)
    for _ in range(10):
        kernel = sy.random_structure(rng, steps, base_kernels="SE,Per")
        kinds = {type(k) for k in leaves(kernel) if not k.is_operator}
        # The 2014 rules bring Const as a scale factor (``*-const``) and
        # Noise as the start; Zero is the empty side of a switch.
        allowed = {ff.SqExpKernel, ff.PeriodicKernel}
        allowed |= {ff.ConstKernel, ff.NoiseKernel, ff.ZeroKernel}
        assert kinds <= allowed


def test_every_parameter_is_set_and_every_value_finite():
    """Twenty draws give finite series of the right length, fully parameterised."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        y, kernel = sy.draw(X, rng)
        assert y.shape == X.shape and np.all(np.isfinite(y))
        assert np.all(np.isfinite(kernel.param_vector))


def test_the_samples_have_the_kernels_covariance():
    """Many draws from one smooth kernel reproduce its covariance."""
    x = np.linspace(0, 1, 5)
    kernel = ff.SqExpKernel(0, lengthscale=np.log(0.5), sf=0.0)
    y, drawn = sy.sample(kernel, x, np.random.default_rng(0), n=20000, noise=0.0)
    xt = ff.as_input(x[:, None])
    theta = torch.as_tensor(drawn.param_vector, dtype=torch.float64)
    expected = drawn.cov(xt, xt, theta, same=True).numpy()
    np.testing.assert_allclose(np.cov(y.T), expected, atol=0.05)


def test_set_parameters_are_kept():
    """A period given by the caller survives the draw of the others."""
    kernel = ff.PeriodicKernel(0, period=0.0)
    _, drawn = sy.sample(kernel, X, np.random.default_rng(0))
    assert drawn.period == 0.0
    assert drawn.lengthscale is not None and drawn.sf is not None


@pytest.mark.slow
def test_the_search_recovers_the_cycle_that_drew_a_series():
    """A yearly cycle drawn by the grammar is found again by the search.

    Twelve points a year over ten years; with eight a year and two starts,
    level 1 found no cycle, so the test keeps the margin it needs.
    """
    from autostat import Experiment, search
    from autostat.features import periods

    kernel = ff.PeriodicKernel(0, lengthscale=0.0, period=0.0, sf=0.0)
    x = np.linspace(2000, 2010, 120)
    y, _ = sy.sample(kernel, x, np.random.default_rng(0), noise=0.1)
    exp = Experiment.abcd_2014(max_depth=2, n_jobs=1, n_rand=3)
    found = periods(search(x, y[0], exp).best.kernel)
    assert any(abs(p - 1.0) < 0.05 for p in found)


def test_no_draw_is_out_of_the_search_bounds():
    """Every drawn period fits a few samples per cycle and a few cycles in the span."""
    from autostat.search import Experiment, data_shape

    shape = data_shape(
        X[:, None], np.random.default_rng(0).standard_normal(80), Experiment()
    )
    rng = np.random.default_rng(5)
    for _ in range(30):
        _, kernel = sy.draw(X, rng)
        assert not kernel.out_of_bounds(shape)
