"""Sequential Monte Carlo over kernel structures, AutoGP's inference, in Python.

What this module does
    It approximates the posterior over Gaussian process structures and their
    parameters with a population of weighted particles. The data enter a few
    at a time (data annealing); after each batch, particles are reweighted
    by how well they predicted it, resampled when the weights degenerate,
    then moved by Markov chain Monte Carlo (MCMC): structure moves from
    :mod:`autostat.smc.moves`, each accepted one followed by Hamiltonian
    Monte Carlo (HMC) on the parameters and on the noise. New observations
    can be added later without starting over.

    >>> model = AutoGP(t, y, n_particles=8, seed=0)
    >>> model.fit_smc(schedule=linear_schedule(len(t), 0.10), n_mcmc=75, n_hmc=10)
    >>> mean, bounds = model.predict(t_new, quantiles=(0.025, 0.975))

Why it exists
    The 2013 and 2014 searches keep one structure, chosen greedily by an
    information criterion. AutoGP (Saad et al., ICML 2023) keeps a weighted
    set of structures and averages their forecasts, which widens intervals
    where the data cannot tell structures apart; it is also 10 to 100 times
    faster. AutoGP.jl is written in Julia; this is a Python port of it.

What it consumes
    Times and values as arrays. Times may be dates converted to numbers.

What it produces
    :class:`AutoGP` objects: particle weights, structures, forecasts and
    their quantiles, log marginal likelihood estimates.

Faithfulness
    Port of ``inference_smc_anneal_data.jl``, ``Schedule.jl`` and ``api.jl``
    from AutoGP.jl v0.1.19 (Apache-2.0, (c) 2023 Google LLC). HMC follows
    ``Gen.hmc``: unit-mass momenta, ``L`` leapfrog steps of size ``eps``.
    Random draws differ from Julia's, so runs agree in distribution, not
    draw for draw; ``tests/test_smc_julia.py`` checks the deterministic
    parts number for number.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy import special

from . import moves
from .tree import (
    Config,
    Scaling,
    Tree,
    covariance,
    log_likelihood,
    log_prior,
    noise_variance,
    parameter_slots,
    sample_tree,
    with_parameters,
)


@dataclass(frozen=True)
class Particle:
    """One hypothesis: a structure with its parameters and the noise.

    Parameters
    ----------
    tree : Tree
        Kernel structure and parameters.
    z_noise : float
        Observation noise on the standard normal scale.
    """

    tree: Tree
    z_noise: float


@dataclass(frozen=True)
class HMCConfig:
    """Leapfrog settings, ``hmc_config`` in AutoGP.jl.

    Parameters
    ----------
    L_param, eps_param : int, float
        Steps and step size for the kernel parameters.
    L_noise, eps_noise : int, float
        Steps and step size for the noise.
    n_exit : int
        Stop a round of HMC after this many rejections in a row.
    gradient : str
        What drives the leapfrog steps of the kernel parameters. ``"prior"``
        (the default) reproduces AutoGP.jl: its ``covariance_prior`` is not
        declared differentiable in Gen, so the likelihood's gradient never
        reaches these parameters and the steps follow the prior's gradient
        ``-z`` alone; the acceptance test still uses the full density, so
        the chain stays exact. ``"full"`` adds the likelihood's gradient, as
        HMC intends. The noise always gets the full gradient, in both.
    """

    L_param: int = 10
    eps_param: float = 0.02
    L_noise: int = 10
    eps_noise: float = 0.02
    n_exit: int = 1
    gradient: str = "prior"

    def __post_init__(self) -> None:
        """Reject an unknown gradient mode."""
        if self.gradient not in ("prior", "full"):
            raise ValueError("gradient must be 'prior' or 'full'")


def _tensor(values: np.ndarray) -> torch.Tensor:
    """Float64 tensor."""
    return torch.as_tensor(np.asarray(values, dtype=float), dtype=torch.float64)


def _score(
    particle: Particle, config: Config, t: torch.Tensor, y: torch.Tensor
) -> float:
    """Log joint density at fixed noise: structure prior plus likelihood.

    Parameters
    ----------
    particle : Particle
        The hypothesis.
    config : Config
        The prior.
    t, y : torch.Tensor
        Scaled data.

    Returns
    -------
    float
        ``log p(tree) + log p(y | tree, noise)``.
    """
    prior = log_prior(particle.tree, config)
    if not math.isfinite(prior):
        return -math.inf
    with torch.no_grad():
        return prior + float(
            log_likelihood(particle.tree, config, t, y, particle.z_noise)
        )


def _hmc(
    particle: Particle,
    config: Config,
    t: torch.Tensor,
    y: torch.Tensor,
    rng: np.random.Generator,
    noise: bool,
    steps: int,
    eps: float,
    prior_gradient: bool = False,
) -> tuple[Particle, bool]:
    """One HMC transition on the kernel parameters or on the noise, ``Gen.hmc``.

    Parameters
    ----------
    particle : Particle
        Current state.
    config : Config
        The prior.
    t, y : torch.Tensor
        Scaled data.
    rng : numpy.random.Generator
        Randomness.
    noise : bool
        Move the noise (True) or the kernel parameters (False).
    steps, eps : int, float
        Leapfrog steps and step size.
    prior_gradient : bool, optional
        Drive the steps by the prior's gradient alone, as AutoGP.jl does
        for kernel parameters (see :class:`HMCConfig`).

    Returns
    -------
    tuple
        The new state and whether it was accepted.
    """
    tree = particle.tree
    if noise:
        start = np.array([particle.z_noise])
    else:
        start = np.array([tree[i].z[k] for i, k in parameter_slots(tree)])

    def log_density(values: np.ndarray, gradient: bool) -> tuple[float, np.ndarray]:
        z = _tensor(values).requires_grad_(gradient)
        if noise:
            ll = log_likelihood(tree, config, t, y, z[0])
        else:
            ll = log_likelihood(tree, config, t, y, particle.z_noise, z=z)
        value = ll - 0.5 * (z * z).sum()
        if not torch.isfinite(value):
            return -math.inf, np.full(len(values), np.nan)
        if not gradient:
            return float(value), -values
        (grad,) = torch.autograd.grad(value, z)
        return float(value.detach()), grad.numpy()

    def step_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
        if prior_gradient:
            return math.nan, -values
        return log_density(values, True)

    old_value, _ = log_density(start, False)
    _, grad = step_gradient(start)
    values = start.copy()
    momenta = rng.standard_normal(len(values))
    old_kinetic = 0.5 * momenta @ momenta
    new_value = old_value
    for _ in range(steps):
        momenta = momenta + 0.5 * eps * grad
        values = values + eps * momenta
        new_value, grad = step_gradient(values)
        if not (prior_gradient or math.isfinite(new_value)):
            return particle, False
        momenta = momenta + 0.5 * eps * grad
    if prior_gradient:
        new_value, _ = log_density(values, False)
    alpha = new_value - old_value - 0.5 * momenta @ momenta + old_kinetic
    if not math.log(rng.random()) < alpha:
        return particle, False
    if noise:
        return replace(particle, z_noise=float(values[0])), True
    return replace(particle, tree=with_parameters(tree, values)), True


def rejuvenate_parameters(
    particle: Particle,
    config: Config,
    t: torch.Tensor,
    y: torch.Tensor,
    rng: np.random.Generator,
    n_hmc: int,
    hmc: HMCConfig,
) -> tuple[Particle, int, int]:
    """Up to ``n_hmc`` rounds of HMC, ``rejuvenate_particle_parameters``.

    Each round moves the kernel parameters, then the noise when it is
    inferred. A run of ``hmc.n_exit`` rejected parameter moves ends it.

    Parameters
    ----------
    particle : Particle
        Current state.
    config : Config
        The prior.
    t, y : torch.Tensor
        Scaled data.
    rng : numpy.random.Generator
        Randomness.
    n_hmc : int
        Maximum number of rounds.
    hmc : HMCConfig
        Leapfrog settings.

    Returns
    -------
    tuple
        New state, accepted parameter moves, rounds run.
    """
    accepted = rejected = trials = 0
    for _ in range(n_hmc):
        particle, ok = _hmc(
            particle,
            config,
            t,
            y,
            rng,
            False,
            hmc.L_param,
            hmc.eps_param,
            prior_gradient=hmc.gradient == "prior",
        )
        if config.noise is None:
            particle, _ = _hmc(
                particle, config, t, y, rng, True, hmc.L_noise, hmc.eps_noise
            )
        trials += 1
        accepted += ok
        rejected = 0 if ok else rejected + 1
        if rejected == hmc.n_exit:
            break
    return particle, accepted, trials


def rejuvenate_structure(
    particle: Particle,
    config: Config,
    t: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_mcmc: int,
    n_hmc: int,
    hmc: HMCConfig,
) -> tuple[Particle, dict[str, int]]:
    """``n_mcmc`` structure moves, each accepted one followed by HMC.

    Parameters
    ----------
    particle : Particle
        Current state.
    config : Config
        The prior.
    t, y : numpy.ndarray
        Scaled data.
    seed : int
        Seed of this particle's randomness for this call.
    n_mcmc, n_hmc : int
        Structure moves; HMC rounds after each accepted one.
    hmc : HMCConfig
        Leapfrog settings.

    Returns
    -------
    tuple
        New state and counts ``mh``, ``hmc``, ``hmc_trials``.
    """
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    t_, y_ = _tensor(t), _tensor(y)
    stats = {"mh": 0, "hmc": 0, "hmc_trials": 0}
    current = _score(particle, config, t_, y_)
    for _ in range(n_mcmc):
        move = moves.propose(rng, particle.tree, config)

        def score(tree: Tree, current: Particle = particle) -> float:
            return _score(replace(current, tree=tree), config, t_, y_)

        alpha = moves.log_acceptance(move, score, current)
        if not math.log(rng.random()) < alpha:
            continue
        stats["mh"] += 1
        particle = replace(particle, tree=move.tree)
        particle, ok, trials = rejuvenate_parameters(
            particle, config, t_, y_, rng, n_hmc, hmc
        )
        stats["hmc"] += ok
        stats["hmc_trials"] += trials
        current = _score(particle, config, t_, y_)
    return particle, stats


def _log_likelihoods(
    particles: list[Particle], config: Config, t: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Log likelihood of the data under each particle.

    Parameters
    ----------
    particles : list of Particle
        The population.
    config : Config
        The prior.
    t, y : numpy.ndarray
        Scaled data.

    Returns
    -------
    numpy.ndarray
        One value per particle.
    """
    t_, y_ = _tensor(t), _tensor(y)
    with torch.no_grad():
        return np.array(
            [
                float(log_likelihood(p.tree, config, t_, y_, p.z_noise))
                for p in particles
            ]
        )


def linear_schedule(n: int, percent: float) -> list[int]:
    """Add about ``n * percent`` observations per step, ``Schedule.linear_schedule``.

    Parameters
    ----------
    n : int
        Number of observations.
    percent : float
        Fraction added per step, in ``(0, 1)``.

    Returns
    -------
    list of int
        Cumulative counts, ending at ``n``.
    """
    if not (n > 0 and 0 < percent < 1):
        raise ValueError("need n > 0 and 0 < percent < 1")
    # Julia's round sends halves to even, as Python's does.
    step = round(percent * n)
    checkpoints = list(range(step, n + 1, step))
    remaining = n - checkpoints[-1]
    if remaining == 0:
        return checkpoints
    if remaining < step / 2:
        checkpoints[-1] = n
        return checkpoints
    return [*checkpoints, n]


def logarithmic_schedule(n: int, base: float, start: int) -> list[int]:
    """Add ``start * base**i`` observations at step ``i``, ``Schedule.logarithmic_schedule``.

    Parameters
    ----------
    n : int
        Number of observations.
    base : float
        Growth factor, at least 1.
    start : int
        Observations added at the first step.

    Returns
    -------
    list of int
        Cumulative counts, ending at ``n``.
    """
    if not (n > 0 and base >= 1 and 0 < start <= n):
        raise ValueError("need n > 0, base >= 1 and 0 < start <= n")
    checkpoints: list[int] = []
    total, i = 0, 0
    while True:
        block = start * base**i
        if n < total + block:
            break
        total += round(block)
        i += 1
        checkpoints.append(total)
    return checkpoints if checkpoints[-1] == n else [*checkpoints, n]


class AutoGP:
    """A population of weighted Gaussian process structures for one series.

    Parameters
    ----------
    t, y : array-like
        Times and values.
    n_particles : int, optional
        Population size.
    config : Config, optional
        The prior; AutoGP.jl's defaults when omitted.
    seed : int, optional
        Seed of every random draw.
    n_jobs : int, optional
        Worker processes for the particles; -1 uses every core.
    """

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        n_particles: int = 8,
        config: Config | None = None,
        seed: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.t = np.asarray(t, dtype=float).copy()
        self.y = np.asarray(y, dtype=float).copy()
        if self.t.shape != self.y.shape or self.t.ndim != 1 or len(self.t) < 2:
            raise ValueError("t and y must be 1-D arrays of the same length >= 2")
        if not (np.isfinite(self.t).all() and np.isfinite(self.y).all()):
            raise ValueError("t and y must be finite")
        self.config = config or Config()
        self.scaling = Scaling.fit(self.t, self.y)
        self.n_jobs = n_jobs
        self._seeds = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(self._seeds.spawn(1)[0])
        self.particles = self._prior_particles(n_particles)
        self.log_weights = np.zeros(n_particles)
        self.log_ml_est = 0.0
        self.history: list[dict[str, float]] = []

    # Population bookkeeping -------------------------------------------------

    def _prior_particles(self, n: int) -> list[Particle]:
        """Draw ``n`` particles from the prior."""
        out = []
        for _ in range(n):
            tree = sample_tree(self._rng, self.config)
            if self.config.noise is None:
                z_noise = float(self._rng.standard_normal())
            else:
                z_noise = self.config.untransform("noise", self.config.noise)
            out.append(Particle(tree, z_noise))
        return out

    @property
    def n_particles(self) -> int:
        """Population size."""
        return len(self.particles)

    def particle_weights(self) -> np.ndarray:
        """Normalised weights."""
        return np.exp(self.log_weights - special.logsumexp(self.log_weights))

    def effective_sample_size(self) -> float:
        """``1 / sum(w**2)`` of the normalised weights."""
        w = self.particle_weights()
        return float(1.0 / (w @ w))

    def log_marginal_likelihood_estimate(self) -> float:
        """The running estimate, as ``AutoGP.log_marginal_likelihood_estimate``.

        Returns
        -------
        float
            Updated at each resampling. Add :meth:`log_mean_weight` for the
            estimate that includes the steps since the last one.
        """
        return self.log_ml_est

    def log_mean_weight(self) -> float:
        """``log mean exp(log_weights)``, the part not yet in :attr:`log_ml_est`."""
        return float(special.logsumexp(self.log_weights) - math.log(self.n_particles))

    def maybe_resample(self, ess_threshold: float) -> bool:
        """Multinomial resampling when the effective sample size is below a threshold.

        Parameters
        ----------
        ess_threshold : float
            Resample when the effective sample size is below it.

        Returns
        -------
        bool
            Whether the population was resampled.
        """
        if not self.effective_sample_size() < ess_threshold:
            return False
        parents = self._rng.choice(
            self.n_particles, size=self.n_particles, p=self.particle_weights()
        )
        self.log_ml_est += self.log_mean_weight()
        self.particles = [self.particles[i] for i in parents]
        self.log_weights = np.zeros(self.n_particles)
        return True

    def _scaled(self, order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Scaled data, optionally reordered."""
        t, y = self.scaling.t(self.t), self.scaling.y(self.y)
        return (t, y) if order is None else (t[order], y[order])

    def _reweight(self, t_old, y_old, t_new, y_new) -> None:
        """Add each particle's predictive log likelihood of the new data."""
        self.log_weights += _log_likelihoods(
            self.particles, self.config, t_new, y_new
        ) - _log_likelihoods(self.particles, self.config, t_old, y_old)

    def _rejuvenate(
        self, t: np.ndarray, y: np.ndarray, n_mcmc: int, n_hmc: int, hmc: HMCConfig
    ) -> dict[str, int]:
        """Run the structure moves on every particle, in parallel."""
        seeds = [
            int(s.generate_state(1)[0]) for s in self._seeds.spawn(self.n_particles)
        ]
        results = Parallel(
            n_jobs=min(self.n_jobs, self.n_particles)
            if self.n_jobs > 0
            else self.n_jobs
        )(
            delayed(rejuvenate_structure)(p, self.config, t, y, s, n_mcmc, n_hmc, hmc)
            for p, s in zip(self.particles, seeds, strict=True)
        )
        self.particles = [p for p, _ in results]
        totals = {"mh": 0, "hmc": 0, "hmc_trials": 0}
        for _, stats in results:
            for key in totals:
                totals[key] += stats[key]
        return totals

    # Public inference -------------------------------------------------------

    def fit_smc(
        self,
        schedule: list[int],
        n_mcmc: int | list[int],
        n_hmc: int | list[int],
        shuffle: bool = True,
        adaptive_resampling: bool = True,
        adaptive_rejuvenation: bool = False,
        hmc: HMCConfig | None = None,
        verbose: bool = False,
    ) -> AutoGP:
        """Sequential Monte Carlo with data annealing, ``AutoGP.fit_smc!``.

        Parameters
        ----------
        schedule : list of int
            Cumulative number of observations at each step, ending at ``n``.
        n_mcmc, n_hmc : int or list of int
            Structure moves and HMC rounds per step.
        shuffle : bool, optional
            Add the observations in random order (AutoGP's default) rather
            than in time order.
        adaptive_resampling : bool, optional
            Resample when the effective sample size drops below half the
            population; otherwise whenever it is below the population.
        adaptive_rejuvenation : bool, optional
            Move particles only after a resampling.
        hmc : HMCConfig, optional
            Leapfrog settings; ``n_exit=1`` as in ``fit_smc!``.
        verbose : bool, optional
            Print each step.

        Returns
        -------
        AutoGP
            ``self``, fitted.
        """
        n = len(self.t)
        if not (schedule and schedule[0] >= 1 and schedule[-1] == n):
            raise ValueError("schedule must start at >= 1 and end at len(t)")
        if any(b <= a for a, b in itertools.pairwise(schedule)):
            raise ValueError("schedule must increase")
        steps = len(schedule)
        n_mcmc = [n_mcmc] * steps if isinstance(n_mcmc, int) else list(n_mcmc)
        n_hmc = [n_hmc] * steps if isinstance(n_hmc, int) else list(n_hmc)
        hmc = hmc or HMCConfig(n_exit=1)
        order = self._rng.permutation(n) if shuffle else np.arange(n)
        t, y = self._scaled(order)
        previous = 0
        for step, (count, mcmc, rounds) in enumerate(
            zip(schedule, n_mcmc, n_hmc, strict=True)
        ):
            self._reweight(t[:previous], y[:previous], t[:count], y[:count])
            previous = count
            resampled = False
            if count < n:
                threshold = (
                    self.n_particles / 2 if adaptive_resampling else self.n_particles
                )
                resampled = self.maybe_resample(threshold)
            stats = {"mh": 0, "hmc": 0, "hmc_trials": 0}
            if not adaptive_rejuvenation or resampled:
                stats = self._rejuvenate(t[:count], y[:count], mcmc, rounds, hmc)
            record = {
                "step": step,
                "observations": count,
                "ess": self.effective_sample_size(),
                "resampled": resampled,
                **stats,
            }
            self.history.append(record)
            if verbose:
                print(record)
        return self

    def add_data(self, t: np.ndarray, y: np.ndarray) -> None:
        """Append observations and reweight, ``AutoGP.add_data!``.

        The scaling stays the one fitted on the first data, as in Julia.
        Follow with :meth:`maybe_resample` and :meth:`mcmc_structure` to move
        the particles.

        Parameters
        ----------
        t, y : array-like
            New times and values.
        """
        t_old, y_old = self._scaled()
        self.t = np.concatenate([self.t, np.asarray(t, dtype=float)])
        self.y = np.concatenate([self.y, np.asarray(y, dtype=float)])
        t_new, y_new = self._scaled()
        self._reweight(t_old, y_old, t_new, y_new)

    def mcmc_structure(
        self, n_mcmc: int, n_hmc: int, hmc: HMCConfig | None = None
    ) -> dict[str, int]:
        """Structure moves on every particle, ``AutoGP.mcmc_structure!``.

        Parameters
        ----------
        n_mcmc, n_hmc : int
            Structure moves; HMC rounds after each accepted one.
        hmc : HMCConfig, optional
            Leapfrog settings; ``n_exit=2`` as in Julia.

        Returns
        -------
        dict
            Acceptance counts.
        """
        t, y = self._scaled()
        return self._rejuvenate(t, y, n_mcmc, n_hmc, hmc or HMCConfig(n_exit=2))

    # Prediction --------------------------------------------------------------

    def _predict_particle(
        self, particle: Particle, t_new: np.ndarray, noise_pred: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance at new times on the scaled axis, ``Inference.predict_mvn``."""
        t, y = (_tensor(v) for v in self._scaled())
        s = _tensor(self.scaling.t(t_new))
        with torch.no_grad():
            noise = float(noise_variance(self.config, particle.z_noise))
            k_tt = covariance(particle.tree, self.config, t) + noise * torch.eye(
                len(t), dtype=torch.float64
            )
            k_ts = covariance(particle.tree, self.config, t, s)
            k_ss = covariance(particle.tree, self.config, s)
            chol = torch.linalg.cholesky(0.5 * (k_tt + k_tt.T))
            mean = k_ts.T @ torch.cholesky_solve(y[:, None], chol)[:, 0]
            cov = k_ss - k_ts.T @ torch.cholesky_solve(k_ts, chol)
            cov = 0.5 * (cov + cov.T) + (
                noise if noise_pred is None else noise_pred
            ) * torch.eye(len(s), dtype=torch.float64)
        return mean.numpy(), cov.numpy()

    def predict_particles(
        self, t_new: np.ndarray, noise_pred: float | None = None
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Each particle's weight, mean and covariance on the original scale.

        Parameters
        ----------
        t_new : array-like
            Times to predict.
        noise_pred : float, optional
            Noise variance added to the predictive covariance, on the scaled
            axis; the particle's own noise when omitted.

        Returns
        -------
        list of tuple
            ``(weight, mean, covariance)`` per particle.
        """
        t_new = np.asarray(t_new, dtype=float)
        out = []
        for w, p in zip(self.particle_weights(), self.particles, strict=True):
            mean, cov = self._predict_particle(p, t_new, noise_pred)
            mean, _ = self.scaling.y_back(mean, np.diag(cov))
            out.append((float(w), mean, cov / self.scaling.y_slope**2))
        return out

    def predict(
        self,
        t_new: np.ndarray,
        quantiles: tuple[float, ...] = (),
        noise_pred: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mixture mean and quantiles at new times.

        Parameters
        ----------
        t_new : array-like
            Times to predict.
        quantiles : tuple of float, optional
            Levels in ``(0, 1)``.
        noise_pred : float, optional
            See :meth:`predict_particles`.

        Returns
        -------
        tuple of numpy.ndarray
            Mean ``(m,)`` and quantiles ``(m, len(quantiles))`` of the
            weighted mixture of the particles' Gaussian predictions.
        """
        parts = self.predict_particles(t_new, noise_pred)
        weights = np.array([w for w, _, _ in parts])
        means = np.stack([m for _, m, _ in parts])
        stds = np.sqrt(np.stack([np.clip(np.diag(c), 0, None) for _, _, c in parts]))
        mean = weights @ means
        bounds = (
            np.stack(
                [_mixture_quantile(weights, means, stds, q) for q in quantiles], axis=1
            )
            if quantiles
            else np.empty((len(mean), 0))
        )
        return mean, bounds

    def predictive_log_density(self, t_new: np.ndarray, y_new: np.ndarray) -> float:
        """Log density of new values under the mixture, jointly over the points.

        Parameters
        ----------
        t_new, y_new : array-like
            Held-out times and values.

        Returns
        -------
        float
            ``log sum_i w_i N(y_new; mean_i, cov_i)``, ``AutoGP.predict_proba``.
        """
        y_new = np.asarray(y_new, dtype=float)
        terms = []
        for w, mean, cov in self.predict_particles(t_new):
            chol = np.linalg.cholesky(cov)
            r = np.linalg.solve(chol, y_new - mean)
            terms.append(
                math.log(w)
                - 0.5 * r @ r
                - np.log(np.diag(chol)).sum()
                - 0.5 * len(y_new) * math.log(2 * math.pi)
            )
        return float(special.logsumexp(terms))

    def structures(self) -> list[tuple[float, str]]:
        """Each particle's weight and formula, heaviest first.

        Returns
        -------
        list of tuple
            ``(weight, formula)``.
        """
        pairs = [
            (float(w), p.tree.formula())
            for w, p in zip(self.particle_weights(), self.particles, strict=True)
        ]
        return sorted(pairs, key=lambda pair: -pair[0])


def _mixture_quantile(
    weights: np.ndarray, means: np.ndarray, stds: np.ndarray, q: float
) -> np.ndarray:
    """Quantile of a mixture of normals at every time, by bisection.

    Parameters
    ----------
    weights : numpy.ndarray
        ``(k,)`` mixture weights.
    means, stds : numpy.ndarray
        ``(k, m)`` component moments.
    q : float
        Level in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        ``(m,)`` quantiles, to about 1e-12 of the range.
    """
    if not 0 < q < 1:
        raise ValueError("quantile levels must be in (0, 1)")
    spread = 40 * stds.max(axis=0) + 1e-12
    lo = means.min(axis=0) - spread
    hi = means.max(axis=0) + spread
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = weights @ special.ndtr((mid - means) / np.where(stds > 0, stds, 1e-300))
        below = cdf < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)
