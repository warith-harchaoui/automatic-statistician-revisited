"""Kernel values, gradients and structural algebra.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import special

from autostat import kernels as ff
from autostat.model import GPModel, negative_log_likelihood

X = np.linspace(0, 3, 7)


def se(lengthscale: float = 0.0, sf: float = 0.5) -> ff.SqExpKernel:
    """Build an SE kernel on dimension 0.

    Parameters
    ----------
    lengthscale, sf : float
        Log parameters.

    Returns
    -------
    SqExpKernel
        The kernel.
    """
    return ff.SqExpKernel(dimension=0, lengthscale=lengthscale, sf=sf)


@pytest.mark.parametrize("ell", [0.3, 1.0, 4.0])
def test_the_periodic_kernel_equals_gpml_closed_form(ell):
    """CovPeriodicNoDC through i0e equals the textbook Bessel quotient.

    Parameters
    ----------
    ell : float
        Periodic lengthscale.
    """
    k = ff.PeriodicKernel(
        dimension=0, lengthscale=math.log(ell), period=math.log(0.7), sf=0.2
    )
    d = 2 * np.pi * np.abs(X[:, None] - X[None, :]) / 0.7
    b0 = special.i0(1 / ell**2)
    expected = (
        math.exp(0.4) * (np.exp(np.cos(d) / ell**2) - b0) / (math.exp(1 / ell**2) - b0)
    )
    np.testing.assert_allclose(k(X), expected, rtol=1e-10, atol=1e-12)


def test_a_very_long_periodic_lengthscale_gives_a_cosine():
    """Beyond the limit the periodic kernel is cos(2 pi r / p), as in GPML."""
    k = ff.PeriodicKernel(dimension=0, lengthscale=math.log(1e5), period=0.0, sf=0.0)
    np.testing.assert_allclose(
        k(X), np.cos(2 * np.pi * (X[:, None] - X[None, :])), atol=1e-12
    )


def test_autograd_gradients_match_finite_differences(airline, airline_final):
    """Autograd replaces GPML's hand-written derivatives without changing them.

    Parameters
    ----------
    airline : tuple of numpy.ndarray
        Fixture: the airline inputs and passenger counts.
    airline_final : GPModel
        Fixture: the final published airline model.
    """
    x, y = airline
    model = airline_final
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    theta = torch.as_tensor(model.theta_vector()).clone().requires_grad_(True)
    negative_log_likelihood(model, xt, yt, theta).backward()
    eps = 1e-6
    numeric = []
    for i in range(theta.numel()):
        up, down = theta.detach().clone(), theta.detach().clone()
        up[i] += eps
        down[i] -= eps
        diff = negative_log_likelihood(model, xt, yt, up) - negative_log_likelihood(
            model, xt, yt, down
        )
        numeric.append(float(diff) / (2 * eps))
    np.testing.assert_allclose(theta.grad.numpy(), numeric, rtol=1e-4, atol=1e-4)


def test_se_times_se_collapses_to_one_se_with_combined_lengthscale():
    """exp(-r^2/2a^2) exp(-r^2/2b^2) is one SE with 1/l^2 = 1/a^2 + 1/b^2."""
    product = ff.ProductKernel([se(), se()]).simplified()
    assert product.lengthscale == pytest.approx(-0.5 * math.log(2))


def test_se_times_se_adds_the_log_scales():
    """The product's log scale is the sum of the factors' log scales."""
    assert ff.ProductKernel([se(), se()]).simplified().sf == pytest.approx(1.0)


def test_noise_plus_noise_collapses_to_one_noise_with_summed_variance():
    """Two white noises add their variances."""
    noise = ff.SumKernel([ff.NoiseKernel(sf=0.0), ff.NoiseKernel(sf=0.0)]).simplified()
    assert noise.sf == pytest.approx(0.5 * math.log(2))


def test_a_constant_factor_becomes_a_scale():
    """C x SE is an SE with a larger scale."""
    scaled = ff.ProductKernel([ff.ConstKernel(sf=1.0), se()]).simplified()
    assert (type(scaled), scaled.sf) == (ff.SqExpKernel, pytest.approx(1.5))


def test_noise_absorbs_a_stationary_factor():
    """WN x SE is white noise."""
    assert isinstance(
        ff.ProductKernel([ff.NoiseKernel(sf=0.0), se()]).simplified(), ff.NoiseKernel
    )


def test_a_product_does_not_count_redundant_scales():
    """SE x Per has 2 + 3 parameters but only 4 effective ones."""
    per = ff.PeriodicKernel(dimension=0, lengthscale=0.0, period=0.0, sf=0.0)
    assert ff.ProductKernel([se(), per]).effective_params == 4


def test_additive_form_distributes_a_product_over_a_sum():
    """(SE + Lin) x Per becomes Lin x Per + Per x SE."""
    lin = ff.LinearKernel(dimension=0, sf=0.0, location=0.0)
    per = ff.PeriodicKernel(dimension=0, lengthscale=0.0, period=0.0, sf=0.0)
    additive = ff.ProductKernel([ff.SumKernel([se(), lin]), per]).additive_form()
    assert additive.structure() == "Lin x Per + Per x SE"


def test_serialisation_round_trips_a_changepoint_model():
    """to_dict and from_dict replace the original's repr and eval."""
    from autostat import benchmarks

    model = benchmarks.published("07-call-centre")[-1]
    assert repr(GPModel.from_dict(model.to_dict()).kernel) == repr(model.kernel)


def test_from_dict_rejects_an_unknown_kernel_type():
    """Deserialisation names what it does not know."""
    with pytest.raises(ValueError, match="unknown kernel type"):
        ff.from_dict({"type": "MagicKernel"})


def test_base_kernels_reject_an_unknown_name():
    """A typo in a kernel list fails loudly instead of being ignored."""
    with pytest.raises(ValueError, match="unknown base kernels"):
        list(ff.base_kernels(1, "SE,Perr"))


def test_a_changepoint_needs_exactly_two_operands():
    """A changepoint switches between two kernels, no more, no fewer."""
    with pytest.raises(ValueError, match="exactly two"):
        ff.ChangePointKernel(dimension=0, operands=[se()])
