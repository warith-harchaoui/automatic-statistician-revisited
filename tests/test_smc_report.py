"""AutoGP structures translated to 2014 kernels: same covariance, same likelihood.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

import math

import numpy as np
import pytest
import torch

from autostat import kernels as ff
from autostat.describe import translate_product
from autostat.smc import AutoGP, Config, Particle, Scaling, linear_schedule
from autostat.smc.report import (
    PosteriorReport,
    describe_posterior,
    to_kernel,
    to_model,
)
from autostat.smc.tree import covariance, log_likelihood, sample_tree


@pytest.fixture(scope="module")
def series():
    """A trend plus a yearly cycle, in decimal years.

    Returns
    -------
    tuple of numpy.ndarray
        Times and values.
    """
    rng = np.random.default_rng(3)
    x = 1990 + np.arange(96) / 12
    y = 50 + 2 * (x - 1990) + 5 * np.sin(2 * np.pi * x) + rng.normal(0, 1, len(x))
    return x, y


def test_translated_kernels_have_the_same_covariance(series):
    """200 prior structures: the 2014 kernel equals AutoGP's, rescaled.

    Parameters
    ----------
    series : tuple of numpy.ndarray
        Fixture: times in decimal years and values.
    """
    x, _ = series
    config, scaling = Config(), Scaling.fit(*series)
    t = torch.tensor(scaling.t(x))
    xs = torch.tensor(x).reshape(-1, 1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        tree = sample_tree(rng, config)
        kernel = to_kernel(tree, config, scaling)
        ours = kernel.cov(xs, xs, torch.as_tensor(kernel.param_vector), True)
        theirs = covariance(tree, config, t) / scaling.y_slope**2
        np.testing.assert_allclose(
            ours.numpy(),
            theirs.numpy(),
            rtol=1e-9,
            atol=1e-9 * float(theirs.abs().max()),
        )


def test_translated_models_have_the_same_likelihood(series):
    """The 2014 NLL is AutoGP's on the scaled axis, plus the Jacobian of the scaling.

    Parameters
    ----------
    series : tuple of numpy.ndarray
        Fixture: times in decimal years and values.
    """
    x, y = series
    config, scaling = Config(), Scaling.fit(x, y)
    t, yy = torch.tensor(scaling.t(x)), torch.tensor(scaling.y(y))
    rng = np.random.default_rng(1)
    for _ in range(50):
        particle = Particle(sample_tree(rng, config), float(rng.standard_normal()))
        model = to_model(particle, config, scaling, x, y)
        scaled = float(log_likelihood(particle.tree, config, t, yy, particle.z_noise))
        if not math.isfinite(scaled):
            continue
        expected = -(scaled + len(x) * math.log(scaling.y_slope))
        assert model.nll == pytest.approx(expected, rel=1e-8, abs=1e-6)


def test_a_rough_gamma_exponential_is_called_rough():
    """Exponent 1: 'rough', with the exponent in the text."""
    kernel = ff.GammaExpKernel(dimension=0, lengthscale=0.0, gamma=1.0, sf=0.0)
    summary, desc, _ = translate_product(kernel, np.linspace(0, 10, 50), 0, 0.0, "")
    assert "rough" in summary
    assert "1.00" in desc[-1]


def test_a_gamma_exponential_of_exponent_two_reads_as_smooth():
    """Exponent 2 is a squared exponential: no 'rough'."""
    kernel = ff.GammaExpKernel(dimension=0, lengthscale=0.0, gamma=2.0, sf=0.0)
    summary, _, _ = translate_product(kernel, np.linspace(0, 10, 50), 0, 0.0, "")
    assert "rough" not in summary
    assert "smooth" in summary


@pytest.mark.slow
def test_a_posterior_report_is_written(series, tmp_path):
    """A small fit yields a summary page and one checked report per structure.

    Parameters
    ----------
    series : tuple of numpy.ndarray
        Fixture: times in decimal years and values.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    x, y = series
    model = AutoGP(x, y, n_particles=4, seed=0, n_jobs=1)
    model.fit_smc(linear_schedule(len(x), 0.25), n_mcmc=10, n_hmc=5)
    report = describe_posterior(model, name="toy", out_dir=tmp_path, top=2)
    assert (tmp_path / "report.md").exists()
    assert report.reports
    assert (tmp_path / "structure_1" / "report.md").exists()
    assert report.reports[0].checks is not None
    assert abs(sum(w for w, _, _ in report.structures) - 1) < 1e-9
    assert len(report.bics) == len(report.structures)
    text = (tmp_path / "report.md").read_text()
    if len(report.structures) > 1:
        assert (tmp_path / report.bic_figure).exists()
        assert "BIC" in " ".join(report.bic_sentences())
    else:
        assert "no BIC to compare" in text


def test_the_summary_says_whether_bic_agrees_with_the_posterior():
    """BIC ranking the heaviest structure first agrees; ranking another disagrees."""
    structures = [(0.6, "a", "A"), (0.4, "b", "B")]
    agree = PosteriorReport("toy", structures, bics=[100.0, 104.0])
    assert "BIC agrees" in " ".join(agree.bic_sentences())
    differ = PosteriorReport("toy", structures, bics=[104.0, 100.0])
    text = " ".join(differ.bic_sentences())
    assert "ranks structure 2 first (weight 0.400), 4.0 below" in text
    assert agree.markdown().startswith('---\nauthor: "Some Math empowered')
    assert r"\author{Some Math empowered" in agree.latex()
    single = PosteriorReport("toy", structures[:1], bics=[100.0])
    assert single.bic_sentences() == ["One structure only: there is no BIC to compare."]
