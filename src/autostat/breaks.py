"""Breaks a detector finds in the data, proposed to the search as dated shifts.

What this module does
    :func:`detect` finds the dates where a series' level jumps. Between two
    breaks it fits a line and, when the periodogram shows a cycle, two
    harmonics of its period: the trend and the cycle sit inside the cost,
    so a cycle is not mistaken for a run of jumps. The segmentation is
    exact (optimal partitioning with PELT pruning); the penalty is BIC's,
    the parameters of one more segment times ``log n`` times a robust
    estimate of the noise variance. :func:`proposals` turns the dates into
    structures with a level shift starting at each one, which
    :func:`autostat.search.search` fits beside the grammar's neighbours.

Why it exists as a separate module
    The 2014 grammar reaches a jump only by turning a smooth term into a
    changepoint, then placing it; on the synthetic jump family it missed
    the jump six times of six. A language model told the date finds it,
    one left to guess misleads the search (``doc/research/propose``). A
    detector gives the date from the data alone. Detectors are weak on
    their own (on TCPDBench the best barely beat "no change"), so they only
    propose here; BIC decides. A constant-mean detector on the raw values
    or on STL components cuts every cycle; with the trend and the cycle in
    the cost it found the jump 5 times of 5 with no false break
    (``doc/research/changepoints``).

What it consumes
    Inputs and targets of one-dimensional data; for :func:`proposals`, the
    best models of the search and the data summary that carries the dates.

What it produces
    Dates, strongest first; unfitted kernels with a starting location.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import kernels as ff


def noise_sd(y: np.ndarray) -> float:
    """Estimate the noise spread from second differences, robustly.

    Second differences remove a level and a slope; the median absolute
    deviation ignores the few large values a jump leaves.

    Parameters
    ----------
    y : numpy.ndarray
        Values ``(n,)``, in time order.

    Returns
    -------
    float
        The standard deviation of white noise that would give them.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    if y.size < 3:
        return float(np.std(y))
    return float(np.median(np.abs(np.diff(y, 2))) / 0.6745 / math.sqrt(6))


def _design(x: np.ndarray, period: float | None) -> np.ndarray:
    """Build the regressors of a line and two harmonics of one period.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs ``(n,)``.
    period : float or None
        The cycle's period; ``None`` for a line alone.

    Returns
    -------
    numpy.ndarray
        Regressors ``(n, 2)`` or ``(n, 6)``.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    span = float(x.max() - x.min()) or 1.0
    columns = [np.ones_like(x), (x - x.mean()) / span]
    if period:
        for j in (1, 2):
            angle = 2 * np.pi * j * x / period
            columns += [np.sin(angle), np.cos(angle)]
    return np.column_stack(columns)


class _Costs:
    """Residual sums of squares of segment regressions from running sums.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """

    def __init__(self, design: np.ndarray, y: np.ndarray) -> None:
        """Accumulate the sums every segment cost is read from.

        Parameters
        ----------
        design : numpy.ndarray
            Regressors ``(n, p)``.
        y : numpy.ndarray
            Targets ``(n,)``.
        """
        n, p = design.shape
        self.xx = np.zeros((n + 1, p, p))
        self.xx[1:] = np.cumsum(design[:, :, None] * design[:, None, :], axis=0)
        self.xy = np.zeros((n + 1, p))
        self.xy[1:] = np.cumsum(design * y[:, None], axis=0)
        self.yy = np.concatenate([[0.0], np.cumsum(y * y)])
        self.ridge = 1e-9 * np.eye(p)

    def __call__(self, starts: np.ndarray, end: int) -> np.ndarray:
        """Return the cost of every segment ``[start, end)``.

        Parameters
        ----------
        starts : numpy.ndarray
            Segment starts ``(k,)``.
        end : int
            Common segment end, excluded.

        Returns
        -------
        numpy.ndarray
            Residual sums of squares ``(k,)``.
        """
        xx = self.xx[end] - self.xx[starts]
        xy = self.xy[end] - self.xy[starts]
        yy = self.yy[end] - self.yy[starts]
        scale = np.trace(xx, axis1=1, axis2=2)[:, None, None] + 1.0
        beta = np.linalg.solve(xx + scale * self.ridge, xy[:, :, None])[:, :, 0]
        return np.maximum(yy - np.einsum("kp,kp->k", beta, xy), 0.0)


def _segment(costs: _Costs, n: int, penalty: float, min_size: int) -> list[int]:
    """Find the exact penalised segmentation, PELT-pruned.

    Parameters
    ----------
    costs : _Costs
        Segment costs.
    n : int
        Number of points.
    penalty : float
        Cost of one more segment.
    min_size : int
        Fewest points in a segment.

    Returns
    -------
    list of int
        Indices where a new segment starts, in order.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates = np.array([0])
    for end in range(min_size, n + 1):
        usable = candidates[candidates <= end - min_size]
        if usable.size:
            total = best[usable] + costs(usable, end) + penalty
            k = int(np.argmin(total))
            best[end], last[end] = total[k], usable[k]
            # PELT: a start that already costs more than the best plus one
            # penalty can never be the last break of a later optimum.
            candidates = np.concatenate(
                [
                    candidates[candidates > end - min_size],
                    usable[total - penalty <= best[end]],
                ]
            )
        if end + min_size <= n:
            candidates = np.append(candidates, end)
    breaks, end = [], n
    while end > 0:
        end = int(last[end])
        if end > 0:
            breaks.append(end)
    return sorted(breaks)


def detect(
    x: np.ndarray, y: np.ndarray, count: int, period: float | None = None
) -> list[float]:
    """Find the dates where a series' level jumps, strongest first.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs ``(n,)``.
    y : numpy.ndarray
        Targets ``(n,)``.
    count : int
        Dates returned at most.
    period : float, optional
        A cycle fitted inside every segment; ``None`` fits a line alone.

    Returns
    -------
    list of float
        Each break halfway between the two points it separates, ranked by
        how much cost merging its two segments would add.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    design = _design(x, period)
    n, p = design.shape
    sd = noise_sd(y)
    if count <= 0 or n < 2 * (p + 2) or sd <= 0:
        return []
    costs = _Costs(design, y)
    breaks = _segment(costs, n, (p + 1) * math.log(n) * sd**2, p + 2)
    bounds = [0, *breaks, n]
    gains = [
        costs(np.array([bounds[i - 1]]), bounds[i + 1])[0]
        - costs(np.array([bounds[i - 1]]), bounds[i])[0]
        - costs(np.array([bounds[i]]), bounds[i + 1])[0]
        for i in range(1, len(bounds) - 1)
    ]
    ranked = sorted(zip(gains, breaks, strict=True), reverse=True)[:count]
    return [float((x[b - 1] + x[b]) / 2) for _, b in ranked]


def steps(
    x: np.ndarray, y: np.ndarray, dates: list[float], period: float | None = None
) -> dict[str, Any]:
    """Fit the detector's model once: a line, the cycle and a step per date.

    The coefficients give a structure proposed from the breaks its starting
    values, so that it is not left to random starts: with them the true
    structure of the synthetic jump family fits to a BIC near 400; from
    random starts it ends anywhere between 400 and 600.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs ``(n,)``.
    y : numpy.ndarray
        Targets ``(n,)``.
    dates : list of float
        Where the steps are.
    period : float, optional
        The cycle's period; ``None`` for none.

    Returns
    -------
    dict
        ``slope`` and ``zero`` (where the line crosses zero) of the line,
        ``amplitude`` of the cycle (0 without one), ``sizes`` of the steps
        in the order of ``dates`` and ``noise``, the residual spread.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    centre = float(x.mean())
    columns = [np.ones_like(x), x - centre]
    if period:
        for j in (1, 2):
            angle = 2 * np.pi * j * x / period
            columns += [np.sin(angle), np.cos(angle)]
    harmonics = len(columns) - 2
    columns += [(x > date).astype(float) for date in dates]
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    level, slope = float(beta[0]), float(beta[1])
    span = float(np.ptp(x)) or 1.0
    # A flat line crosses zero far away: keep its location where the
    # search would draw one, the slope then small.
    zero = centre - level / slope if slope else centre
    zero = float(np.clip(zero, x.min() - span, x.max() + span))
    return {
        "slope": slope,
        "zero": zero,
        "amplitude": float(np.hypot.reduce(beta[2 : 2 + harmonics]))
        if harmonics
        else 0.0,
        "sizes": [float(b) for b in beta[2 + harmonics :]],
        "noise": float(np.std(residual)),
    }


def _log(value: float) -> float:
    """Return the log of a positive scale, floored for a zero one.

    Parameters
    ----------
    value : float
        A scale, any sign.

    Returns
    -------
    float
        ``log(max(|value|, 1e-6))``.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    return math.log(max(abs(value), 1e-6))


def shift(
    location: float, size: float | None = None, steepness: float | None = None
) -> ff.ChangePointKernel:
    """Return a level shift, ``CP(Zero, C)``, starting at a date.

    Parameters
    ----------
    location : float
        Where the switch starts its optimisation.
    size : float, optional
        The step's size, which starts the constant's scale.
    steepness : float, optional
        Log of the switch's sharpness, as :class:`ChangePointKernel` reads it.

    Returns
    -------
    ChangePointKernel
        Unfitted but for what is given.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    constant = ff.ConstKernel(sf=None if size is None else _log(size))
    return ff.ChangePointKernel(
        0, location=location, steepness=steepness, operands=[ff.ZeroKernel(), constant]
    )


def _terms(kernel: ff.Kernel) -> list[ff.Kernel]:
    """Split a kernel into its additive terms, without the empty one.

    Parameters
    ----------
    kernel : Kernel
        Any kernel.

    Returns
    -------
    list of Kernel
        Copies of its terms.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    terms = kernel.operands if isinstance(kernel, ff.SumKernel) else [kernel]
    return [k.copy() for k in terms if not isinstance(k, ff.NoneKernel)]


def proposals(models: list[Any], shape: dict[str, Any], depth: int) -> list[ff.Kernel]:
    """Propose the detector's model, then a level shift at every detected date.

    At the first level the detector's own model is proposed, every
    parameter started from its fit (:func:`steps`): a line, the cycle when
    it found one, white noise and a shift at each date. At every level the
    best model so far is proposed with one more shift, its size and
    sharpness started too, unless the model already holds a switch near
    that date.

    Parameters
    ----------
    models : list of GPModel
        Best models so far, best first.
    shape : dict
        Data summary; ``break_hints`` holds the dates, ``break_period`` the
        cycle of the detector or ``None``, ``break_fit`` the output of
        :func:`steps`, ``x_min_abs_diff`` the log of the closest spacing.
    depth : int
        The level about to be fitted.

    Returns
    -------
    list of Kernel
        Unfitted structures with starting values.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    dates = shape.get("break_hints") or []
    if not dates:
        return []
    period = shape.get("break_period")
    fit = shape["break_fit"]
    step = math.exp(shape["x_min_abs_diff"][0])
    # A switch about two samples wide; tanh reaches 0.96 one sample away.
    steep = math.log(1.0 / step)
    shifts = [
        shift(date, size, steep) for date, size in zip(dates, fit["sizes"], strict=True)
    ]
    out: list[ff.Kernel] = []
    if depth == 0:
        base = [
            ff.LinearKernel(dimension=0, sf=_log(fit["slope"]), location=fit["zero"]),
            ff.NoiseKernel(sf=_log(fit["noise"])),
        ]
        if period:
            # A sine of amplitude A has variance A^2 / 2; a lengthscale of
            # one keeps the periodic kernel close to a sine.
            cycle = ff.PeriodicKernel(
                dimension=0,
                lengthscale=0.0,
                period=math.log(period),
                sf=_log(fit["amplitude"] / math.sqrt(2)),
            )
            base.append(cycle)
        out.append(ff.SumKernel([*base, *shifts]).canonical())
    for model in models[:1]:
        terms = _terms(model.kernel)
        for new in shifts:
            # Within three of the closest spacings, a switch is already there.
            held = any(
                isinstance(k, ff.ChangePointKernel)
                and k.location is not None
                and abs(k.location - new.location) <= 3 * step
                for k in terms
            )
            if not held:
                out.append(ff.SumKernel([*terms, new.copy()]).canonical())
    return out
