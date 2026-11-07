"""The AutoGP port against AutoGP.jl itself, number for number, through juliacall.

These tests run only where ``juliacall`` is installed (the benchmark's
virtual environment); the first run downloads Julia and AutoGP.jl v0.1.19.
Each compares a deterministic quantity the two implementations must share:
trace scores, covariance matrices, gradients, the densities of structure
moves drawn by Julia, predictive distributions.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

import math

import numpy as np
import pytest

juliacall = pytest.importorskip("juliacall")

import torch  # noqa: E402  (juliacall must load before torch)

from autostat import benchmarks  # noqa: E402
from autostat.smc import AutoGP, Particle, moves  # noqa: E402
from autostat.smc import julia as J  # noqa: E402
from autostat.smc.inference import _score  # noqa: E402
from autostat.smc.tree import (  # noqa: E402
    FIELDS,
    Config,
    Scaling,
    covariance,
    log_likelihood,
    log_prior,
)

pytestmark = pytest.mark.slow

_LOG_2PI = math.log(2 * math.pi)
_LOG_HALF = math.log(0.5)


@pytest.fixture(scope="module")
def data():
    """Airline, scaled as AutoGP scales it.

    Returns
    -------
    tuple
        Raw times, raw values, scaled times, scaled values, scaling.
    """
    x, y = benchmarks.load("01-airline")
    x, y = np.ravel(x), np.ravel(y)
    scaling = Scaling.fit(x, y)
    return x, y, scaling.t(x), scaling.y(y), scaling


def test_scores_and_covariances_match_on_prior_draws(data):
    """30 structures drawn by Julia: same joint density, same kernel matrix."""
    _, _, t, y, _ = data
    config = Config()
    tt, yy = torch.tensor(t), torch.tensor(y)
    for seed in range(30):
        trace = J.simulate(seed, t, y)
        tree, z_noise = J.tree_from_trace(trace)
        ours = (
            log_prior(tree, config)
            - 0.5 * (z_noise**2 + _LOG_2PI)
            + float(log_likelihood(tree, config, tt, yy, z_noise))
        )
        assert ours == pytest.approx(J.score(trace), rel=1e-12, abs=1e-9)
        np.testing.assert_allclose(
            covariance(tree, config, tt).numpy(), J.covariance(trace), atol=1e-12
        )


def test_structure_moves_have_the_same_densities(data):
    """200 moves drawn by Julia: same forward, reverse and model ratios."""
    _, _, t, y, _ = data
    config = Config()
    tt, yy = torch.tensor(t), torch.tensor(y)
    kinds = set()
    for seed in range(200):
        trace = J.simulate(seed % 50, t, y)
        old, z_noise = J.tree_from_trace(trace)
        move = J.propose(1000 + seed, trace)
        new, _ = J.tree_from_trace(move["new_trace"])
        a, b = move["a"], move["b"]
        kinds.add(move["kind"])
        if move["kind"] == "replace":
            ours = moves.replace_densities(old, new, a, config)
        elif move["kind"] == "detach":
            assert new == old.replaced(a, old.moved(b, a))
            ours = moves.detach_densities(old, new, a, b, config)
        else:
            ours = moves.attach_densities(old, new, a, b, config)
        # Gen's proposal densities include the 1/2 of choosing the move type.
        assert ours.log_q_forward + _LOG_HALF == pytest.approx(
            move["forward"], abs=1e-9
        )
        assert ours.log_q_reverse + _LOG_HALF == pytest.approx(
            move["reverse"], abs=1e-9
        )
        weight = _score(Particle(new, z_noise), config, tt, yy) - _score(
            Particle(old, z_noise), config, tt, yy
        )
        if math.isfinite(move["weight"]):
            assert weight == pytest.approx(move["weight"], rel=1e-9, abs=1e-8)
        else:
            assert weight == move["weight"]
    assert kinds == {"replace", "detach", "attach"}


def test_julia_moves_kernel_parameters_by_the_prior_gradient_only(data):
    """Gen's gradient for every kernel parameter is ``-z``: no likelihood term.

    This is why :class:`autostat.smc.HMCConfig` defaults to
    ``gradient="prior"``.
    """
    _, _, t, y, _ = data
    for seed in range(20):
        gradients, _ = J.gradients(J.simulate(seed, t, y))
        for value, gradient in gradients.values():
            assert gradient == pytest.approx(-value, abs=1e-12)
    assert not bool(
        J.julia().seval("Gen.accepts_output_grad(AutoGP.Model.covariance_prior)")
    )


def test_the_noise_gradient_includes_the_likelihood(data):
    """For the noise, Gen's gradient is the full one, as the port's."""
    _, _, t, y, _ = data
    config = Config()
    tt, yy = torch.tensor(t), torch.tensor(y)
    for seed in range(20):
        trace = J.simulate(seed, t, y)
        tree, z_noise = J.tree_from_trace(trace)
        _, julia_gradient = J.gradients(trace)
        z = torch.tensor(z_noise, dtype=torch.float64, requires_grad=True)
        value = log_likelihood(tree, config, tt, yy, z) - 0.5 * z * z
        (ours,) = torch.autograd.grad(value, z)
        assert float(ours) == pytest.approx(julia_gradient, rel=1e-6, abs=1e-8)


def test_predictive_distributions_match(data):
    """Same predictive mean and covariance, 20 structures, 15 future times."""
    x, y, t, yy, scaling = data
    t_new = np.linspace(1.0, 1.2, 15)
    x_new = (t_new - scaling.t_intercept) / scaling.t_slope
    model = AutoGP(x, y, n_particles=1, seed=0)
    for seed in range(20):
        trace = J.simulate(seed, t, yy)
        tree, z_noise = J.tree_from_trace(trace)
        mean_j, cov_j = J.predict_mvn(trace, t_new)
        mean, cov = model._predict_particle(Particle(tree, z_noise), x_new, None)
        scale = max(1.0, np.abs(cov_j).max())
        np.testing.assert_allclose(
            mean, mean_j, atol=1e-9 * max(1, np.abs(mean_j).max())
        )
        np.testing.assert_allclose(cov, cov_j, atol=1e-9 * scale)


def test_round_trip_through_julia_keeps_the_tree(data):
    """Python tree to Gen trace and back is the identity."""
    _, _, t, y, _ = data
    for seed in range(10):
        tree, z_noise = J.tree_from_trace(J.simulate(seed, t, y))
        back, z_back = J.tree_from_trace(J.trace_from_tree(tree, z_noise, t, y))
        assert back == tree
        assert z_back == z_noise
        assert all(len(tree[i].z) == len(FIELDS[tree[i].kind]) for i in tree)
