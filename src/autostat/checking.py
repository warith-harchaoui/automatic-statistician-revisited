"""Model checking: test whether each fitted component looks like a draw from its prior.

What this module does
    For every additive component of a fitted model it computes posterior
    predictive p-values that compare what the data say about the component
    (its posterior) with what the model assumed (its prior):

    * **QQ** (quantile-quantile): the largest and smallest gaps between the
      sorted component and the prior quantiles;
    * **ACF** (autocorrelation function): the depth and position of its
      minimum;
    * **periodogram**: the height and position of its largest peak;
    * **MMD** (maximum mean discrepancy): a kernel two-sample test between the
      data and data in which the component is replaced by a prior draw.

    A small p-value flags a property of the data the component's prior does
    not reproduce: a heavier tail than Gaussian, a dip in the autocorrelation
    at a lag the kernel does not expect, a spectral peak it misses.

Why it exists as a separate module
    The 2014 reports carry a "model checking" section computed by a separate
    MATLAB file, ``checking_stats.m``, built on the MMD code of Lloyd and
    Ghahramani's model-criticism work (``github.com/jamesrobertlloyd/kmc-research``,
    NIPS 2015). Checking is optional and slower than the description, so it
    stays apart.

What it consumes
    A fitted :class:`autostat.model.GPModel`, its additive components, the
    data and a random generator.

What it produces
    One :class:`ComponentCheck` per component.

Faithfulness
    The statistics, the sample count (1,000), the bandwidth selection for the
    MMD test (five-fold cross-validated kernel density likelihood over fifty
    candidate widths) and the one-sided p-value convention are the
    original's. One step is reinterpreted: ``checking_stats.m`` builds the
    MMD's "posterior" sample with a covariance between resampled points that
    mixes two index sets. This port states the intent instead: remove the
    component's posterior mean at the resampled inputs and add a draw from
    the component's prior there.

Author
------
Warith Harchaoui <warith@deraison.ai>. The ported design is (c) 2013-2015
James Robert Lloyd and David Duvenaud, MIT licence, reproduced in
``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from . import kernels as ff
from .model import GPModel, posterior

#: Draws per statistic, as in the original.
SAMPLES = 1000
#: Candidate bandwidths and folds for the MMD kernel width.
MMD_WIDTHS = 50
MMD_FOLDS = 5


@dataclass
class ComponentCheck:
    """Posterior predictive p-values for one component.

    Every value is a two-sided p-value in ``[0, 1]``; small means the data
    disagree with the component's prior on that statistic.

    Parameters
    ----------
    qq_max, qq_min : float
        Largest and smallest gap between sorted values and prior quantiles.
    acf_min, acf_min_loc : float
        Depth and lag of the autocorrelation minimum.
    pxx_max, pxx_max_loc : float
        Height and frequency of the periodogram's largest peak.
    mmd : float
        Kernel two-sample test, one-sided as in the original.
    """

    qq_max: float
    qq_min: float
    acf_min: float
    acf_min_loc: float
    pxx_max: float
    pxx_max_loc: float
    mmd: float

    def as_dict(self) -> dict[str, float]:
        """Return the p-values keyed by statistic.

        Returns
        -------
        dict
            Seven entries.
        """
        return dict(self.__dict__)


def _two_sided(p: float) -> float:
    """Convert a cumulative probability to a two-sided p-value.

    Parameters
    ----------
    p : float
        Fraction of prior draws below the posterior's.

    Returns
    -------
    float
        ``2 min(p, 1 - p)``, the original's ``convert_cum_prob_to_p_value``.
    """
    return float(min(2 * p, 2 * (1 - p)))


def _draw(
    mean: np.ndarray, cov: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw Gaussian samples from a covariance that is only nearly positive definite.

    The original called ``non_singular`` before ``chol``. An eigenvalue clip
    does the same job and also accepts the rank-deficient covariances of
    linear and constant components.

    Parameters
    ----------
    mean : numpy.ndarray
        Shape ``(n,)``.
    cov : numpy.ndarray
        Shape ``(n, n)``, symmetric.
    size : int
        Number of draws.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, size)``.
    """
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    return mean[:, None] + root @ rng.standard_normal((len(mean), size))


def _exceedance(prior: np.ndarray, post: np.ndarray, rng: np.random.Generator) -> float:
    """Return the fraction of prior statistics below the posterior ones.

    The original breaks ties with a tiny Gaussian perturbation of the
    posterior statistics; this port does the same.

    Parameters
    ----------
    prior, post : numpy.ndarray
        Statistics from prior and posterior draws, same length.
    rng : numpy.random.Generator
        Source of the tie-breaking noise.

    Returns
    -------
    float
        A cumulative probability.
    """
    scale = 1e-4 * np.max(np.abs(post)) if np.max(np.abs(post)) > 0 else 1e-12
    return float(np.mean(prior < post + scale * rng.standard_normal(post.shape)))


def _qq_stats(
    samples: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Largest and smallest gap between each sorted sample and the reference quantiles.

    Parameters
    ----------
    samples : numpy.ndarray
        Shape ``(n, draws)``.
    reference : numpy.ndarray
        Prior quantiles, shape ``(n,)``.

    Returns
    -------
    tuple of numpy.ndarray
        Maxima and minima, each of shape ``(draws,)``.
    """
    gaps = np.sort(samples, axis=0) - reference[:, None]
    return gaps.max(axis=0), gaps.min(axis=0)


def _acf(samples: np.ndarray) -> np.ndarray:
    """Sample autocorrelation at every lag, as MATLAB's ``autocorr``.

    Parameters
    ----------
    samples : numpy.ndarray
        Shape ``(n, draws)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, draws)``, lag 0 first.
    """
    n = samples.shape[0]
    centred = samples - samples.mean(axis=0)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=0)
    raw = np.fft.irfft(spectrum * np.conj(spectrum), axis=0)[:n]
    return raw / np.where(raw[0] > 0, raw[0], 1.0)


def _periodogram_db(samples: np.ndarray) -> np.ndarray:
    """One-sided periodogram in decibels, as MATLAB's ``periodogram`` default.

    Parameters
    ----------
    samples : numpy.ndarray
        Shape ``(n, draws)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(frequencies, draws)``.
    """
    n = samples.shape[0]
    nfft = max(256, 1 << int(np.ceil(np.log2(n))))
    _, power = signal.periodogram(samples, nfft=nfft, detrend=False, axis=0)
    return 10 * np.log10(np.maximum(power, 1e-300))


def _grid(x: np.ndarray) -> np.ndarray:
    """Return an evenly spaced grid for the spectral statistics.

    The original used the inputs themselves when they were evenly spaced,
    a grid of the implied size when the gaps were whole multiples of the
    smallest one; otherwise a grid of ``n`` points.

    Parameters
    ----------
    x : numpy.ndarray
        One input dimension.

    Returns
    -------
    numpy.ndarray
        Grid, shape ``(m,)``.
    """
    xs = np.sort(x)
    gaps = np.diff(xs)
    multiples = np.round(gaps / gaps.min() * 10) / 10
    if np.all(multiples == 1):
        return xs
    if np.all(multiples == np.round(multiples)):
        count = int(multiples.sum()) + 1
    else:
        count = len(x)
    return np.linspace(xs[0], xs[-1], count)


def _rbf(a: np.ndarray, b: np.ndarray, width: float) -> np.ndarray:
    """Gaussian kernel matrix with the MMD code's convention ``exp(-d^2 / 2 w^2)``.

    Parameters
    ----------
    a, b : numpy.ndarray
        Point sets, shapes ``(m, d)`` and ``(n, d)``.
    width : float
        Bandwidth.

    Returns
    -------
    numpy.ndarray
        Shape ``(m, n)``.
    """
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-np.maximum(d2, 0.0) / (2 * width**2))


def _mmd_width(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    """Pick the MMD bandwidth by cross-validated kernel density likelihood.

    Fifty candidate widths up to ``sqrt(0.5)`` times the median pairwise
    distance, scored by five-fold held-out log density of both samples.

    Parameters
    ----------
    a, b : numpy.ndarray
        The two samples, already standardised.
    rng : numpy.random.Generator
        Shuffles the folds.

    Returns
    -------
    float
        The best width.
    """
    distances = np.sqrt(
        np.concatenate(
            [np.sum((p[:, None] - p[None, :]) ** 2, -1).ravel() for p in (a, b)]
        )
    )
    median = np.sort(distances)[len(distances) // 2]
    widths = np.arange(1, MMD_WIDTHS + 1) * np.sqrt(0.5) * median / MMD_WIDTHS
    dim = a.shape[1]
    folds = [np.array_split(rng.permutation(len(p)), MMD_FOLDS) for p in (a, b)]
    best_width, best_score = widths[0], -np.inf
    for width in widths:
        if width <= 0:
            continue
        score = 0.0
        for sample, parts in zip((a, b), folds, strict=True):
            for held in parts:
                train = np.setdiff1d(np.arange(len(sample)), held)
                density = _rbf(sample[train], sample[held], width).sum(0)
                density = density / len(sample) / width**dim
                score += float(np.sum(np.log(np.maximum(density, 1e-300))))
        if score > best_score:
            best_width, best_score = width, score
    return float(best_width)


def mmd_test(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, shuffles: int = SAMPLES
) -> float:
    """Permutation p-value of the biased MMD statistic between two samples.

    Parameters
    ----------
    a, b : numpy.ndarray
        Samples, shapes ``(m, d)`` and ``(n, d)``.
    rng : numpy.random.Generator
        Source of randomness.
    shuffles : int, optional
        Permutations.

    Returns
    -------
    float
        Fraction of permuted statistics above the observed one.
    """
    scale = a.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    a, b = a / scale, b / scale
    width = _mmd_width(a, b, rng)
    joint = np.vstack([a, b])
    gram = _rbf(joint, joint, width)
    m = len(a)

    def statistic(order: np.ndarray) -> float:
        """Compute the biased MMD for one assignment of points to samples.

        Parameters
        ----------
        order : numpy.ndarray
            Permutation; the first ``m`` indices form the first sample.

        Returns
        -------
        float
            The statistic.
        """
        first, second = order[:m], order[m:]
        return float(
            gram[np.ix_(first, first)].mean()
            - 2 * gram[np.ix_(first, second)].mean()
            + gram[np.ix_(second, second)].mean()
        )

    observed = statistic(np.arange(len(joint)))
    permuted = np.array(
        [statistic(rng.permutation(len(joint))) for _ in range(shuffles)]
    )
    return float(np.mean(observed < permuted))


def check_component(
    model: GPModel,
    component: ff.Kernel,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    samples: int = SAMPLES,
) -> ComponentCheck:
    """Compute every posterior predictive p-value for one component.

    Parameters
    ----------
    model : GPModel
        Fitted model containing the component.
    component : Kernel
        One additive summand, with its parameters.
    x : numpy.ndarray
        Inputs, shape ``(n, 1)``.
    y : numpy.ndarray
        Targets, shape ``(n,)``.
    rng : numpy.random.Generator
        Source of randomness.
    samples : int, optional
        Draws per statistic.

    Returns
    -------
    ComponentCheck
        Seven p-values.
    """
    n = len(y)
    prior_cov = component(x)
    post_mean, post_cov = posterior(model, x, y, component=component, full_cov=True)
    zero = np.zeros(n)

    # QQ: compare sorted draws with quantiles of the pooled prior draws.
    reference_pool = _draw(zero, prior_cov, samples, rng).ravel()
    levels = np.linspace(0, 1, n + 2)[1:-1]
    reference = np.quantile(reference_pool, levels)
    prior_max, prior_min = _qq_stats(_draw(zero, prior_cov, samples, rng), reference)
    post_max, post_min = _qq_stats(_draw(post_mean, post_cov, samples, rng), reference)
    qq_max = _two_sided(_exceedance(-prior_max, -post_max, rng))
    qq_min = _two_sided(_exceedance(prior_min, post_min, rng))

    # ACF and periodogram on an evenly spaced grid.
    grid = _grid(x[:, 0])[:, None]
    grid_prior = component(grid)
    grid_mean, grid_cov = posterior(
        model, x, y, x_star=grid, component=component, full_cov=True
    )
    prior_draws = _draw(np.zeros(len(grid)), grid_prior, samples, rng)
    post_draws = _draw(grid_mean, grid_cov, samples, rng)
    prior_acf, post_acf = _acf(prior_draws), _acf(post_draws)
    acf_min = _two_sided(_exceedance(prior_acf.min(0), post_acf.min(0), rng))
    acf_min_loc = _two_sided(
        _exceedance(
            prior_acf.argmin(0).astype(float), post_acf.argmin(0).astype(float), rng
        )
    )
    prior_pxx, post_pxx = _periodogram_db(prior_draws), _periodogram_db(post_draws)
    pxx_max = _two_sided(_exceedance(prior_pxx.max(0), post_pxx.max(0), rng))
    pxx_max_loc = _two_sided(
        _exceedance(
            prior_pxx.argmax(0).astype(float), post_pxx.argmax(0).astype(float), rng
        )
    )

    # MMD: data against data whose component is replaced by a prior draw,
    # both resampled with replacement.
    pick = rng.integers(0, n, n)
    swapped = y[pick] - post_mean[pick] + _draw(zero, prior_cov, 1, rng)[pick, 0]
    data = rng.integers(0, n, n)
    first = np.column_stack([x[data, 0], y[data]])
    second = np.column_stack([x[pick, 0], swapped])
    mmd = mmd_test(first, second, rng, shuffles=samples)

    return ComponentCheck(
        qq_max, qq_min, acf_min, acf_min_loc, pxx_max, pxx_max_loc, mmd
    )


def check_model(
    model: GPModel,
    components: list[ff.Kernel],
    x: Any,
    y: Any,
    seed: int = 0,
    samples: int = SAMPLES,
) -> list[ComponentCheck]:
    """Check every component of a fitted model.

    Parameters
    ----------
    model : GPModel
        Fitted model.
    components : list of Kernel
        Its additive components, in report order.
    x, y : array_like
        The data it was fitted to.
    seed : int, optional
        Seeds every draw, so a report is reproducible.
    samples : int, optional
        Draws per statistic.

    Returns
    -------
    list of ComponentCheck
        One per component, same order.
    """
    rng = np.random.default_rng(seed)
    xs = np.asarray(x, float).reshape(len(np.ravel(y)), -1)
    ys = np.asarray(y, float).ravel()
    return [check_component(model, c, xs, ys, rng, samples) for c in components]
