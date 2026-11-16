"""Series drawn from the grammar: random structures and Gaussian process samples.

What this module does
    It walks the search grammar at random to build a kernel expression, gives
    the expression parameters with the same data-dependent draws the search
    starts from. It then samples series from the Gaussian process prior the
    expression defines. Each series comes with the expression that made it.

Why it exists as a separate module
    Kernel composition is the standard synthetic prior of the 2024 to 2026
    foundation models (KernelSynth for Chronos, CauKer, TempoPFN). Here the
    same grammar that searches also generates, which gives three uses:
    recovery tests (does the search find the structure that drew the
    series?), training data for amortised initialisation, a sampler for
    others.

What it consumes
    Input locations, a random generator and the grammar's vocabulary.

What it produces
    :func:`random_structure` gives a kernel expression; :func:`sample` gives
    series and the parameterised kernel; :func:`draw` does both.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import kernels as ff
from .grammar import Rule, expand_kernels
from .model import cholesky
from .search import Experiment, data_shape

#: The base kernels of the published 2014 searches.
BASE = "SE,Per,Lin,Const,Noise"


def random_structure(
    rng: np.random.Generator,
    steps: int = 2,
    base_kernels: str = BASE,
    rules: list[Rule] | None = None,
) -> ff.Kernel:
    """Walk the grammar at random from one base kernel.

    Each step replaces the expression by one of its distinct neighbours,
    chosen uniformly, so ``steps`` levels of the search could reach it.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    steps : int, optional
        Grammar rewrites after the first base kernel.
    base_kernels : str, optional
        Comma-separated base kernel identifiers.
    rules : list of Rule, optional
        Rewrite rules; the 2014 set when omitted.

    Returns
    -------
    Kernel
        A canonical expression without parameters.
    """
    bases = list(ff.base_kernels(1, base_kernels))
    kernel = bases[rng.integers(len(bases))].canonical()
    for _ in range(steps):
        neighbours = expand_kernels(1, [kernel], base_kernels, rules)
        if not neighbours:
            break
        kernel = neighbours[rng.integers(len(neighbours))]
    return kernel


def sample(
    kernel: ff.Kernel,
    x: Any,
    rng: np.random.Generator,
    n: int = 1,
    sd: float = 1.0,
    noise: float = 0.1,
) -> tuple[np.ndarray, ff.Kernel]:
    """Draw series from the Gaussian process the kernel defines.

    Unset parameters are drawn as the search draws its starts, against a
    series of unit spread on ``x``: lengthscales and periods scale with the
    span of ``x``, so one expression yields series of every kind.

    Parameters
    ----------
    kernel : Kernel
        An expression; its set parameters are kept.
    x : array_like
        Input locations, shape ``(m,)`` or ``(m, 1)``.
    rng : numpy.random.Generator
        Source of randomness.
    n : int, optional
        Series drawn from the same parameters.
    sd : float, optional
        Spread of the parameter draws, on the log scale.
    noise : float, optional
        Standard deviation of white noise added to every value.

    Returns
    -------
    y : numpy.ndarray
        Shape ``(n, m)``.
    kernel : Kernel
        A copy of the expression with every parameter set.

    Raises
    ------
    ValueError
        When the covariance has no Cholesky factor even with jitter.
    """
    xs = np.asarray(x, dtype=float).reshape(len(x), -1)
    unit = rng.standard_normal(len(xs))
    shape = data_shape(xs, unit / unit.std(), Experiment())
    drawn = kernel.copy()
    drawn.initialise_params(rng, sd, shape)
    xt = ff.as_input(xs)
    theta = torch.as_tensor(drawn.param_vector, dtype=torch.float64)
    with torch.no_grad():
        cov = drawn.cov(xt, xt, theta, same=True)
        chol = cholesky(0.5 * (cov + cov.T))
    if chol is None:
        raise ValueError(f"{drawn.pretty_print()} has no Cholesky factor on x")
    z = rng.standard_normal((len(xs), n))
    y = (chol.numpy() @ z).T + noise * rng.standard_normal((n, len(xs)))
    return y, drawn


def draw(
    x: Any,
    rng: np.random.Generator,
    steps: int = 2,
    base_kernels: str = BASE,
    noise: float = 0.1,
    tries: int = 10,
) -> tuple[np.ndarray, ff.Kernel]:
    """Draw one structure and one series from it.

    Parameters
    ----------
    x : array_like
        Input locations.
    rng : numpy.random.Generator
        Source of randomness.
    steps : int, optional
        Grammar rewrites, as in :func:`random_structure`.
    base_kernels : str, optional
        Comma-separated base kernel identifiers.
    noise : float, optional
        Standard deviation of the added white noise.
    tries : int, optional
        Structures drawn before giving up when parameters make the
        covariance singular.

    Returns
    -------
    y : numpy.ndarray
        Shape ``(m,)``.
    kernel : Kernel
        The parameterised expression that drew it.

    Raises
    ------
    ValueError
        When no structure in ``tries`` gives a usable covariance.
    """
    for _ in range(tries):
        structure = random_structure(rng, steps, base_kernels)
        try:
            y, kernel = sample(structure, x, rng, noise=noise)
        except ValueError:
            continue
        if np.all(np.isfinite(y)):
            return y[0], kernel
    raise ValueError(f"no usable structure in {tries} tries")
