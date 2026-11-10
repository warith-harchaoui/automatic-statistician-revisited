"""The formula a search found: written exactly, read back exactly.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autostat import benchmarks, describe
from autostat import kernels as ff
from autostat.formula import kernel_tex, number, polynomial_tex, switch_weight


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (1949.0, "1949"),
        (0.85134, "0.8513"),
        (31.58, "31.58"),
        (2.31e-5, r"2.31 \times 10^{-5}"),
    ],
)
def test_numbers_are_written_with_four_significant_digits(value, text):
    """Plain decimals in a readable range, powers of ten outside it.

    Parameters
    ----------
    value : float
        The number.
    text : str
        Its LaTeX.
    """
    assert number(value) == text


def test_a_product_has_one_amplitude():
    """sf^2 of every factor multiply into a single coefficient."""
    product = ff.ProductKernel(
        [
            ff.LinearKernel(dimension=0, sf=math.log(2.0), location=1950.0),
            ff.NoiseKernel(sf=math.log(3.0)),
        ]
    )
    assert kernel_tex(product) == r"36\,(t - 1950)(t' - 1950) \cdot \delta_{t t'}"


def test_a_change_of_regime_is_named_once():
    """Two components switched by the same window share one weight w_1."""
    switches: dict = {}
    for amplitude in (1.0, 2.0):
        window = ff.ChangeWindowKernel(
            dimension=0,
            location=1680.0,
            steepness=math.log(0.13),
            width=math.log(73.0),
            operands=[ff.ZeroKernel(), ff.ConstKernel(sf=math.log(amplitude))],
        )
        assert "w_{1}(t)" in kernel_tex(window, switches)
    assert list(switches.values()) == ["w_{1}"]


def test_the_window_weight_is_one_inside_and_zero_outside():
    """w(t) rises at the window's start and falls at its end."""
    key = (1.0, 1680.0, math.log(1.0), math.log(70.0))
    inside, before, after = switch_weight(key, np.array([1680.0, 1500.0, 1900.0]))
    assert inside == pytest.approx(1.0, abs=1e-6)
    assert (before, after) == (
        pytest.approx(0.0, abs=1e-6),
        pytest.approx(0.0, abs=1e-6),
    )


def test_polynomials_read_signs_correctly():
    """A negative leading term keeps its minus sign, without a stray space."""
    assert polynomial_tex(np.array([-0.5, 2.0]), 1949.0) == r"-0.5 + 2\,(t - 1949)"


@pytest.mark.slow
def test_the_airline_trend_is_a_straight_line_through_its_posterior_mean(
    airline, airline_final
):
    """The linear component's closed form reproduces its posterior mean.

    Parameters
    ----------
    airline : tuple of numpy.ndarray
        Fixture: the airline series.
    airline_final : GPModel
        Fixture: the published final model.
    """
    x, y = airline
    report = describe(airline_final, x, y, name="airline", check=False)
    found = report.formula()
    assert found.explicit[0] is not None
    assert found.explicit[0].startswith(r"f_{1}(t) = 92.68 + 31.58\,(t - 1949)")
    assert "The formula found" in report.markdown()
    assert r"\begin{dmath*}" in report.latex()


def test_every_published_model_can_be_written():
    """The writer knows every kernel the thirteen 2014 models use."""
    for name in benchmarks.names():
        kernel = benchmarks.published(name)[-1].simplified().kernel
        assert kernel_tex(kernel)


def test_a_periodic_component_is_written_as_its_fourier_series():
    """A sine of amplitude 8 plus a trend comes back as a cosine of amplitude near 8."""
    from autostat.describe import components
    from autostat.formula import formula
    from autostat.model import GPModel, fit

    rng = np.random.default_rng(0)
    t = np.arange(96) / 12.0 + 2016.0
    y = 100 + 6 * (t - 2016) + 8 * np.sin(2 * np.pi * t) + rng.normal(0, 2, t.size)
    kernel = ff.SumKernel(
        [
            ff.LinearKernel(dimension=0, sf=1.8, location=2000.0),
            ff.NoiseKernel(sf=0.6),
            ff.PeriodicKernel(dimension=0, lengthscale=2.0, period=0.0, sf=1.7),
        ]
    )
    model = fit(GPModel(kernel, ndata=t.size), t[:, None], y)
    parts = components(model)
    found = formula(model, parts, list(range(len(parts))), t[:, None], y)
    (series,) = [e for e in found.explicit if e and r"\approx" in e]
    amplitude = float(series.split(r"\approx ")[1].split(r"\cos")[0])
    assert abs(amplitude - 8) < 1.0


def test_a_tiny_lengthscale_squares_without_a_double_superscript():
    """A length scale in scientific notation is parenthesised before ^2."""
    kernel = ff.SqExpKernel(dimension=0, lengthscale=math.log(5.787e-4), sf=0.0)
    text = kernel_tex(kernel)
    assert r"\left(5.787 \times 10^{-4}\right)^2" in text


@pytest.mark.slow
def test_every_published_kernel_compiles_in_latex(tmp_path):
    """The 116 kernels of every level of the 13 published searches typeset.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    import shutil
    import subprocess

    if shutil.which("pdflatex") is None:
        pytest.skip("no pdflatex")
    equations = sorted(
        {
            kernel_tex(model.simplified().kernel)
            for name in benchmarks.names()
            for model in benchmarks.published(name)
        }
    )
    body = "\n".join(rf"\begin{{dmath*}}k = {e}\end{{dmath*}}" for e in equations)
    (tmp_path / "k.tex").write_text(
        r"\documentclass{article}\usepackage{amsmath}\usepackage{breqn}"
        rf"\begin{{document}}{body}\end{{document}}"
    )
    done = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "k.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, [
        line for line in done.stdout.splitlines() if line.startswith("!")
    ]


def test_a_missing_or_failing_pysr_leaves_the_covariance(monkeypatch):
    """Without a usable PySR the component simply has no closed form.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the worker's process.
    """
    import subprocess

    from autostat import formula as module

    for status in (3, 1):
        finished = subprocess.CompletedProcess([], status, "", "")
        monkeypatch.setattr(module.subprocess, "run", lambda *a, r=finished, **k: r)
        with pytest.warns(UserWarning, match="PySR"):
            assert module.symbolic_form(np.arange(3.0), np.arange(3.0)) is None


def test_only_smooth_components_go_to_symbolic_regression():
    """Noise and quasi-periodic components have no short curve to read."""
    from autostat.formula import is_smooth

    smooth = ff.SqExpKernel(dimension=0, lengthscale=0.0, sf=0.0)
    periodic = ff.PeriodicKernel(dimension=0, lengthscale=0.0, period=0.0, sf=0.0)
    assert is_smooth(smooth)
    assert not is_smooth(ff.NoiseKernel(sf=0.0))
    assert not is_smooth(ff.ProductKernel([smooth, periodic]))
