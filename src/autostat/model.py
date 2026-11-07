"""A Gaussian process model and the optimiser that scores it.

What this module does
    It holds a Gaussian process (GP) model (a mean, a kernel and a Gaussian
    likelihood), computes its negative log marginal likelihood, optimises its
    hyperparameters and returns posterior means and covariances, for the
    whole model or for one additive component.

Why it exists as a separate module
    In the original, this work crossed a language boundary three times. The
    Python 2 code wrote one MATLAB script per kernel (``gpml.py``), ran it in
    a subprocess, optimised with GPML's ``minimize.m`` and read the result
    back from a ``.mat`` file. Here it is one function call.

What it consumes
    A kernel from :mod:`autostat.kernels` and data as arrays.

What it produces
    Fitted :class:`GPModel` objects with their score; posterior moments.

Numerical choices
    The marginal likelihood is the textbook one, computed through a Cholesky
    factor in float64. L-BFGS (limited-memory Broyden-Fletcher-Goldfarb-Shanno)
    minimises it with a strong-Wolfe line search on autograd gradients. The original optimised for ``iters`` line searches on
    a random 250-point subset, then ``full_iters`` more on all the data. That
    two-stage schedule is kept because the published results came from it.

Author
------
Warith Harchaoui <warith@deraison.ai>. The ported design is (c) 2013-2015
James Robert Lloyd, David Duvenaud and Roger Grosse, MIT licence, reproduced
in ``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import kernels as ff

_LOG_2PI = math.log(2 * math.pi)

#: Relative diagonal jitter tried in order until the Cholesky factor exists.
#: The first attempt adds **none**, as GPML's ``infDelta`` did. A jitter
#: relative to the mean diagonal is large when a constant kernel dominates
#: that diagonal: on the solar series it moved the published likelihood by 61
#: nats before the first entry was set to zero. Only a factorisation that
#: fails outright (one GPML scored NaN and discarded) is retried with jitter.
JITTERS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)


@dataclass
class GPModel:
    """A mean, a kernel and a Gaussian likelihood, plus the fit's score.

    Parameters
    ----------
    kernel : Kernel
        The covariance function.
    mean_const : float or None, optional
        ``None`` for a zero mean (the original's ``MeanZero``), otherwise the
        constant of ``MeanConst``. Every published configuration used zero.
    lik_sf : float, optional
        Log noise standard deviation. ``-inf``, the default and the 2014
        setting, means no likelihood noise: observation noise is then a
        white-noise *kernel* the search discovers.
    nll : float, optional
        Negative log marginal likelihood at the fitted parameters.
    ndata : int, optional
        Number of observations ``nll`` was computed on.
    """

    kernel: ff.Kernel
    mean_const: float | None = None
    lik_sf: float = -math.inf
    nll: float = math.nan
    ndata: int = 0

    def copy(self) -> GPModel:
        """Return a deep copy.

        Returns
        -------
        GPModel
            Shares no mutable state with ``self``.
        """
        return GPModel(
            self.kernel.copy(), self.mean_const, self.lik_sf, self.nll, self.ndata
        )

    # Scores

    @property
    def bic(self) -> float:
        """Bayesian information criterion ``2 nll + p log n``.

        ``p`` counts kernel parameters only, as the original did.

        Returns
        -------
        float
            Lower is better.
        """
        return 2 * self.nll + self.kernel.effective_params * math.log(self.ndata)

    @property
    def aic(self) -> float:
        """Akaike information criterion ``2 nll + 2 p``.

        Returns
        -------
        float
            Lower is better.
        """
        return 2 * self.nll + 2 * self.kernel.effective_params

    @property
    def pl2(self) -> float:
        """Per-datum penalised likelihood the authors also tried.

        Returns
        -------
        float
            ``nll / n + p / (2 n)``.
        """
        return self.nll / self.ndata + self.kernel.effective_params / (2 * self.ndata)

    def score(self, criterion: str = "bic") -> float:
        """Return the search criterion; lower is better.

        Parameters
        ----------
        criterion : {"bic", "aic", "pl2", "nll"}
            Which score.

        Returns
        -------
        float
            The score.

        Raises
        ------
        KeyError
            On an unknown criterion.
        """
        scores = {"bic": self.bic, "aic": self.aic, "pl2": self.pl2, "nll": self.nll}
        return scores[criterion.lower()]

    # Structure

    def simplified(self) -> GPModel:
        """Return the same model with its kernel simplified.

        Returns
        -------
        GPModel
            A copy.
        """
        out = self.copy()
        out.kernel = out.kernel.simplified()
        return out

    def additive_form(self) -> GPModel:
        """Return the same model with its kernel in additive form.

        Returns
        -------
        GPModel
            A copy.
        """
        out = self.copy()
        out.kernel = out.kernel.additive_form()
        return out

    def out_of_bounds(self, shape: dict[str, Any]) -> bool:
        """Report whether the fitted kernel left its sensible range.

        Parameters
        ----------
        shape : dict
            Data summary.

        Returns
        -------
        bool
            True when out of bounds.
        """
        return self.kernel.out_of_bounds(shape)

    def __repr__(self) -> str:
        """Render the original's ``GPModel(...)`` line format.

        Returns
        -------
        str
            Parseable by :func:`autostat.legacy.parse_model`.
        """
        mean = (
            "MeanZero()"
            if self.mean_const is None
            else f"MeanConst(c={self.mean_const!r})"
        )
        return (
            f"GPModel(mean={mean}, "
            f"kernel={self.kernel!r}, likelihood=LikGauss(sf={self.lik_sf!r}), "
            f"nll={self.nll!r}, ndata={self.ndata})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Describe the model as JSON-serialisable data.

        Returns
        -------
        dict
            Inverted by :meth:`from_dict`; ``bic`` and ``structure`` are
            included for readers.
        """
        return {
            "kernel": self.kernel.to_dict(),
            "mean_const": self.mean_const,
            "lik_sf": self.lik_sf,
            "nll": self.nll,
            "ndata": self.ndata,
            "bic": self.bic if self.ndata else None,
            "structure": self.kernel.structure(),
        }

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> GPModel:
        """Rebuild a model from :meth:`to_dict` output.

        Parameters
        ----------
        spec : dict
            Serialised model.

        Returns
        -------
        GPModel
            Equal to the serialised model.
        """
        return cls(
            ff.from_dict(spec["kernel"]),
            spec.get("mean_const"),
            spec.get("lik_sf", -math.inf),
            spec.get("nll", math.nan),
            spec.get("ndata", 0),
        )

    # Parameters as one vector

    def theta_vector(self) -> np.ndarray:
        """Concatenate mean, kernel and likelihood parameters.

        Returns
        -------
        numpy.ndarray
            The optimiser's view of the model.
        """
        parts = [self.kernel.param_vector]
        if self.mean_const is not None:
            parts.insert(0, np.array([self.mean_const]))
        if math.isfinite(self.lik_sf):
            parts.append(np.array([self.lik_sf]))
        return np.concatenate(parts)

    def load_theta_vector(self, theta: np.ndarray) -> None:
        """Write a vector from :meth:`theta_vector` back into the model.

        Parameters
        ----------
        theta : numpy.ndarray
            Same layout as :meth:`theta_vector`.
        """
        i = 0
        if self.mean_const is not None:
            self.mean_const = float(theta[0])
            i = 1
        n = self.kernel.num_params
        self.kernel.load_param_vector(theta[i : i + n])
        if math.isfinite(self.lik_sf):
            self.lik_sf = float(theta[i + n])

    def split_theta(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Split a parameter tensor into mean, kernel and likelihood parts.

        Parameters
        ----------
        theta : torch.Tensor
            Same layout as :meth:`theta_vector`.

        Returns
        -------
        tuple
            Mean (a scalar, zero for a zero mean), kernel parameters and the
            log noise or ``None`` without likelihood noise.
        """
        i = 0
        mean = torch.zeros((), dtype=torch.float64)
        if self.mean_const is not None:
            mean = theta[0]
            i = 1
        n = self.kernel.num_params
        lik = theta[i + n] if math.isfinite(self.lik_sf) else None
        return mean, theta[i : i + n], lik


def _targets(y: Any) -> torch.Tensor:
    """Convert targets to a float64 vector.

    Parameters
    ----------
    y : array_like
        Targets.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)``.
    """
    return torch.as_tensor(np.asarray(y, float).ravel(), dtype=torch.float64)


def _train_cov(
    model: GPModel, x: torch.Tensor, theta_k: torch.Tensor, lik: torch.Tensor | None
) -> torch.Tensor:
    """Build the training covariance, likelihood noise included.

    Parameters
    ----------
    model : GPModel
        Supplies the kernel.
    x : torch.Tensor
        Training inputs.
    theta_k : torch.Tensor
        Kernel parameters.
    lik : torch.Tensor or None
        Log noise standard deviation, if any.

    Returns
    -------
    torch.Tensor
        Shape ``(n, n)``.
    """
    k = model.kernel.cov(x, x, theta_k, same=True)
    if lik is not None:
        k = k + torch.exp(2 * lik) * torch.eye(x.shape[0], dtype=torch.float64)
    return k


def cholesky(k: torch.Tensor) -> torch.Tensor | None:
    """Factorise a covariance, adding jitter only if the plain factor fails.

    Parameters
    ----------
    k : torch.Tensor
        Symmetric matrix.

    Returns
    -------
    torch.Tensor or None
        Lower Cholesky factor, or ``None`` when every jitter in
        :data:`JITTERS` fails.
    """
    scale = torch.clamp(torch.diagonal(k).mean().detach(), min=1e-300)
    eye = torch.eye(k.shape[0], dtype=torch.float64)
    for jitter in JITTERS:
        chol, info = torch.linalg.cholesky_ex(k + jitter * scale * eye)
        if int(info) == 0:
            return chol
    return None


def negative_log_likelihood(
    model: GPModel, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Compute the negative log marginal likelihood at ``theta``.

    Parameters
    ----------
    model : GPModel
        Supplies the structure.
    x, y : torch.Tensor
        Inputs ``(n, D)`` and targets ``(n,)``.
    theta : torch.Tensor
        All parameters, laid out as :meth:`GPModel.theta_vector`.

    Returns
    -------
    torch.Tensor
        Scalar; ``+inf`` when the covariance has no Cholesky factor.
    """
    mean, theta_k, lik = model.split_theta(theta)
    k = _train_cov(model, x, theta_k, lik)
    if not torch.isfinite(k).all():
        return torch.tensor(math.inf, dtype=torch.float64)
    chol = cholesky(k)
    if chol is None:
        return torch.tensor(math.inf, dtype=torch.float64)
    resid = (y - mean)[:, None]
    alpha = torch.cholesky_solve(resid, chol)
    n = y.shape[0]
    return (
        0.5 * (resid * alpha).sum()
        + torch.log(torch.diagonal(chol)).sum()
        + 0.5 * n * _LOG_2PI
    )


def _minimise(model: GPModel, x: torch.Tensor, y: torch.Tensor, iters: int) -> None:
    """Run L-BFGS from the model's parameters and keep the best point seen.

    Parameters
    ----------
    model : GPModel
        Updated in place with the best parameters found.
    x, y : torch.Tensor
        Training data.
    iters : int
        Maximum optimiser iterations.
    """
    theta0 = model.theta_vector()
    if theta0.size == 0 or iters <= 0:
        return
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    best = {"value": math.inf, "theta": theta0.copy()}
    optimiser = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=iters,
        max_eval=int(iters * 1.25) + 1,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        """Evaluate the objective and its gradient, recording the best point.

        Returns
        -------
        torch.Tensor
            The objective at the current parameters.
        """
        optimiser.zero_grad()
        value = negative_log_likelihood(model, x, y, theta)
        if not torch.isfinite(value):
            # A huge finite value lets the line search back off from one bad
            # step instead of aborting the whole run.
            return (
                torch.tensor(1e25, dtype=torch.float64, requires_grad=True)
                + 0 * theta.sum()
            )
        value.backward()
        grad_ok = theta.grad is not None and torch.isfinite(theta.grad).all()
        if value.item() < best["value"] and grad_ok:
            best["value"] = value.item()
            best["theta"] = theta.detach().numpy().copy()
        return value

    # A numerically hopeless model makes the optimiser itself raise. The best
    # parameters seen so far are kept either way.
    with contextlib.suppress(RuntimeError):
        optimiser.step(closure)
    model.load_theta_vector(best["theta"])


def fit(
    model: GPModel,
    x: Any,
    y: Any,
    iters: int = 250,
    subset: bool = True,
    subset_size: int = 250,
    full_iters: int = 10,
    seed: int = 0,
) -> GPModel:
    """Optimise a model's parameters and attach its negative log likelihood.

    Parameters
    ----------
    model : GPModel
        Starting point; every parameter set.
    x, y : array_like
        Inputs and targets.
    iters : int, optional
        Optimiser iterations, on the subset when ``subset`` is on.
    subset : bool, optional
        Optimise on a random subset first, as the published runs did.
    subset_size : int, optional
        Size of that subset.
    full_iters : int, optional
        Iterations on the full data afterwards.
    seed : int, optional
        Seeds the subset draw.

    Returns
    -------
    GPModel
        A fitted copy, ``nll`` computed on all the data (``nan`` on failure).
    """
    # One thread per fit: the search runs many fits in parallel processes.
    torch.set_num_threads(1)
    out = model.copy()
    xt, yt = ff.as_input(x), _targets(y)
    n = yt.shape[0]
    if subset and n > subset_size:
        rng = np.random.default_rng(seed)
        idx = torch.as_tensor(np.sort(rng.choice(n, subset_size, replace=False)))
        _minimise(out, xt[idx], yt[idx], iters)
        _minimise(out, xt, yt, full_iters)
    else:
        _minimise(out, xt, yt, iters)
    with torch.no_grad():
        theta = torch.as_tensor(out.theta_vector())
        value = negative_log_likelihood(out, xt, yt, theta)
    out.nll = float(value) if torch.isfinite(value) else math.nan
    out.ndata = n
    return out


def posterior(
    model: GPModel,
    x: Any,
    y: Any,
    x_star: Any = None,
    component: ff.Kernel | None = None,
    noise: bool = False,
    full_cov: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and (co)variance of the model or of one component.

    For a component ``k_j`` of ``k = sum_j k_j`` the posterior given the data
    has mean ``K_j(*, X) K(X, X)^-1 (y - m)`` and covariance
    ``K_j(*, *) - K_j(*, X) K^-1 K_j(X, *)``. This is the decomposition the
    original computed in ``component_stats_and_plots.m``.

    Parameters
    ----------
    model : GPModel
        Fitted model.
    x, y : array_like
        Training data.
    x_star : array_like, optional
        Where to evaluate. ``None`` means the training inputs themselves,
        where a white-noise component *is* correlated with the data (it is
        the identity there and zero elsewhere), as the original evaluated
        ``data_mean``.
    component : Kernel, optional
        One summand of the model's kernel, with its own parameters. The whole
        kernel when omitted.
    noise : bool, optional
        Add likelihood noise to the variance (whole model only).
    full_cov : bool, optional
        Return the full covariance matrix instead of its diagonal.

    Returns
    -------
    mean : numpy.ndarray
        Shape ``(m,)``.
    var : numpy.ndarray
        Shape ``(m,)``, or ``(m, m)`` with ``full_cov``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When the training covariance has no Cholesky factor.
    """
    at_data = x_star is None
    xt = ff.as_input(x)
    xs = xt if at_data else ff.as_input(x_star)
    yt = _targets(y)
    with torch.no_grad():
        mean0, theta_k, lik = model.split_theta(torch.as_tensor(model.theta_vector()))
        chol = cholesky(_train_cov(model, xt, theta_k, lik))
        if chol is None:
            raise np.linalg.LinAlgError("training covariance is not positive definite")
        part = component if component is not None else model.kernel
        theta_p = torch.as_tensor(part.param_vector, dtype=torch.float64)
        k_sx = part.cov(xs, xt, theta_p, same=at_data)
        k_ss = part.cov(xs, xs, theta_p, same=True)
        alpha = torch.cholesky_solve((yt - mean0)[:, None], chol)
        mean = (k_sx @ alpha).ravel() + (mean0 if component is None else 0.0)
        v = torch.cholesky_solve(k_sx.T, chol)
        cov = k_ss - k_sx @ v
        if noise and component is None and lik is not None:
            cov = cov + torch.exp(2 * lik) * torch.eye(
                cov.shape[0], dtype=torch.float64
            )
    if full_cov:
        return mean.numpy(), 0.5 * (cov + cov.T).numpy()
    return mean.numpy(), np.clip(torch.diagonal(cov).numpy(), 0.0, None)
