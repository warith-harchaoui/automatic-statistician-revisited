"""From a fitted kernel to a written report: decompose, measure, describe, check.

What this module does
    It splits a fitted model into additive components, orders them by how
    much each reduces extrapolation error, measures what each explains,
    describes each in English and optionally checks each against its prior.
    It writes a Markdown report with figures.

Why it exists as a separate module
    In the original this work lived in four files in two languages: the
    decomposition and statistics of ``component_stats_and_plots.m`` and
    ``order_by_mae_reduction.m`` (MATLAB), the model checking of
    ``checking_stats.m`` (MATLAB, ported in :mod:`autostat.checking`) and
    the natural-language rules of ``translation.py`` (Python 2).

What it consumes
    A fitted :class:`autostat.model.GPModel` and the data it was fitted to.

What it produces
    A :class:`Report`. On request it writes ``report.md`` with PNG (Portable Network
    Graphics) figures in place of the original's LaTeX document.

The sentences
    They are the original's. The authors wrote them as a rule system over the
    kernel's structure and fitted parameters; rewording them would make this
    a different tool. Two typos (``conitnue``, ``eigth``) are corrected.

Author
------
Warith Harchaoui <warith@deraison.ai>. The ported design is (c) 2013-2015
James Robert Lloyd, David Duvenaud and Roger Grosse, MIT licence, reproduced
in ``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from . import kernels as ff
from .checking import ComponentCheck, check_model
from .model import GPModel, posterior

POLY_NAMES = ["linear", "quadratic", "cubic", "quartic", "quintic"]
ORDINALS = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
]
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
#: A component counts as active where its envelope exceeds this.
ENV_THRESH = 0.99
#: Seconds in the original's year, which ignored leap days.
SECONDS_PER_YEAR = 365.0 * 24 * 3600

Description = tuple[str, list[str], list[str]]


# Decomposition and statistics


def components(model: GPModel) -> list[ff.Kernel]:
    """Split a model's kernel into simplified additive components.

    Parameters
    ----------
    model : GPModel
        A fitted model.

    Returns
    -------
    list of Kernel
        Summands whose sum is the simplified kernel.
    """
    kernel = model.kernel.simplified().canonical()
    parts = ff.SumKernel(kernel.break_into_summands()).simplified().canonical()
    return list(parts.operands) if isinstance(parts, ff.SumKernel) else [parts]


def _sum(parts: list[ff.Kernel]) -> ff.Kernel:
    """Sum kernels, returning a single kernel unchanged.

    Parameters
    ----------
    parts : list of Kernel
        At least one kernel.

    Returns
    -------
    Kernel
        The sum.
    """
    return parts[0] if len(parts) == 1 else ff.SumKernel([p.copy() for p in parts])


def _contiguous_folds(n: int, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split ``range(n)`` into contiguous validation blocks, as the original.

    Parameters
    ----------
    n : int
        Number of points.
    folds : int
        Number of blocks.

    Returns
    -------
    list of tuple
        ``(train, valid)`` index arrays per fold.
    """
    edges = [int(np.floor(n * f / folds)) for f in range(folds + 1)]
    out = []
    for f in range(folds):
        valid = np.arange(max(edges[f] - 1, 0), edges[f + 1])
        train = np.setdiff1d(np.arange(n), valid)
        out.append((train, valid))
    return out


def order_by_mae(
    model: GPModel,
    parts: list[ff.Kernel],
    x: np.ndarray,
    y: np.ndarray,
    folds: int = 10,
) -> dict[str, Any]:
    """Order components greedily by how much each reduces extrapolation error.

    Each of ten contiguous blocks is predicted from the rest with the full
    model's covariance and, as cross-covariance, only the sum of the
    components chosen so far. Hyperparameters are not refitted, as in the
    original's ``order_by_mae_reduction.m``.

    Parameters
    ----------
    model : GPModel
        Fitted model.
    parts : list of Kernel
        Its components.
    x, y : numpy.ndarray
        Data.
    folds : int, optional
        Number of contiguous blocks.

    Returns
    -------
    dict
        ``order`` (indices into ``parts``), ``maes``, ``mae_reductions`` (per
        cent) and ``mav_data`` (mean absolute value of the data).
    """
    splits = _contiguous_folds(len(y), folds)
    chosen: list[int] = []
    maes: list[float] = []
    reductions: list[float] = []
    previous = float(np.mean(np.abs(y)))
    mav = previous
    for _ in parts:
        best_j, best_mae = -1, math.inf
        for j in range(len(parts)):
            if j in chosen:
                continue
            cumulative = _sum([parts[i] for i in [*chosen, j]])
            errors = []
            for train, valid in splits:
                mean, _ = posterior(
                    model, x[train], y[train], x[valid], component=cumulative
                )
                errors.append(np.mean(np.abs(y[valid] - mean)))
            mae = float(np.mean(errors))
            if mae < best_mae:
                best_j, best_mae = j, mae
        chosen.append(best_j)
        maes.append(best_mae)
        reductions.append((1 - best_mae / previous) * 100)
        previous = best_mae
    return {
        "order": chosen,
        "maes": maes,
        "mae_reductions": reductions,
        "mav_data": mav,
    }


def component_stats(
    model: GPModel,
    parts: list[ff.Kernel],
    order: list[int],
    x: np.ndarray,
    y: np.ndarray,
) -> list[dict[str, float]]:
    """Measure each component and each cumulative sum, in report order.

    Parameters
    ----------
    model : GPModel
        Fitted model.
    parts : list of Kernel
        Its components.
    order : list of int
        Report order, from :func:`order_by_mae`.
    x, y : numpy.ndarray
        Data.

    Returns
    -------
    list of dict
        Per component: ``snr`` (signal-to-noise ratio, in decibels),
        ``var_explained`` and ``cum_var`` (per cent of the data's variance),
        ``cum_resid_var`` (per cent of the previous residual's variance),
        ``monotonic`` (-1, 0 or 1) and ``gradient``.
    """
    out = []
    resid = y
    for rank, j in enumerate(order):
        part = parts[j]
        mean, var = posterior(model, x, y, component=part)
        # The envelope says where a changepoint lets the component act.
        envelope = np.diag(part.cp_structure()(x))
        active = envelope > ENV_THRESH
        diffs = np.diff(mean)[active[:-1] & active[1:]]
        if diffs.size and np.all(diffs > 0):
            monotonic = 1
        elif diffs.size and np.all(diffs < 0):
            monotonic = -1
        else:
            monotonic = 0
        xs, ms = x[active, 0], mean[active]
        gradient = float((ms[-1] - ms[0]) / (xs[-1] - xs[0])) if ms.size > 1 else 0.0
        cumulative = _sum([parts[i] for i in order[: rank + 1]])
        cum_mean, _ = posterior(model, x, y, component=cumulative)
        out.append(
            {
                "snr": 10 * math.log10(np.sum(mean**2) / max(np.sum(var), 1e-300)),
                "var_explained": (1 - np.var(y - mean) / np.var(y)) * 100,
                "cum_var": (1 - np.var(y - cum_mean) / np.var(y)) * 100,
                "cum_resid_var": (1 - np.var(y - cum_mean) / np.var(resid)) * 100,
                "monotonic": monotonic,
                "gradient": gradient,
            }
        )
        resid = y - cum_mean
    return out


# Natural language, ported from translation.py


def english_length(val: float, unit: str) -> str:
    """Write a duration in the most readable unit.

    Parameters
    ----------
    val : float
        Duration in input units.
    unit : str
        ``"year"`` for decimal years, anything else for unitless inputs.

    Returns
    -------
    str
        E.g. ``"8.1 months"``.
    """
    if unit != "year":
        return f"{val:g}"
    steps = (
        (0.75, 1, "years"),
        (2 / 12, 12, "months"),
        (2 / 52, 52, "weeks"),
        (2 / 365, 365, "days"),
        (2 / (24 * 365), 24 * 365, "hours"),
        (2 / (60 * 24 * 365), 60 * 24 * 365, "minutes"),
    )
    for limit, scale, name in steps:
        if val > limit:
            return f"{val * scale:0.1f} {name}"
    return f"{val * 3600 * 24 * 365:0.1f} seconds"


def english_point(val: float, unit: str, x: np.ndarray) -> str:
    """Write a point in time at a precision suited to the data's span.

    Parameters
    ----------
    val : float
        Position in input units.
    unit : str
        ``"year"`` or unitless.
    x : numpy.ndarray
        All inputs, to measure the span.

    Returns
    -------
    str
        A year, a month, a day or a time.
    """
    if unit != "year":
        return f"{val:g}"
    span = float(np.ptp(x))
    t = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=(val - 1970) * SECONDS_PER_YEAR)
    if span > 20:
        return f"{t.year:4d}"
    if span > 2:
        return f"{MONTHS[t.month - 1]} {t.year:4d}"
    if span > 2 / 12:
        return f"{t.day:02d} {MONTHS[t.month - 1]} {t.year:4d}"
    return f"{t:%H:%M:%S} {t.day:02d} {MONTHS[t.month - 1]} {t.year:4d}"


def _is_off(k: ff.Kernel) -> bool:
    """Tell whether a changepoint side is switched off.

    Parameters
    ----------
    k : Kernel
        One operand of a changepoint or change window.

    Returns
    -------
    bool
        True for a zero or deleted operand.
    """
    return isinstance(k, (ff.ZeroKernel, ff.NoneKernel))


def region_of_influence(
    k: ff.Kernel, intervals: tuple[tuple[float, float], ...] = ((-math.inf, math.inf),)
) -> tuple[list[tuple[float, float]], ff.Kernel]:
    """Strip changepoint operators and return where the remaining kernel acts.

    Parameters
    ----------
    k : Kernel
        One additive component.
    intervals : tuple of (float, float), optional
        Region already established by enclosing operators.

    Returns
    -------
    intervals : list of (float, float)
        Where the component acts; empty when the operators cancel out.
    base : Kernel
        The component without its changepoints.
    """
    current = list(intervals)
    if isinstance(k, ff.ChangePointKernel):
        if _is_off(k.operands[0]):
            local, base = [(k.location, math.inf)], k.operands[1]
        else:
            local, base = [(-math.inf, k.location)], k.operands[0]
    elif isinstance(k, ff.ChangeWindowKernel):
        half = math.exp(k.width) / 2
        if _is_off(k.operands[0]):
            local, base = [(k.location - half, k.location + half)], k.operands[1]
        else:
            local = [(-math.inf, k.location - half), (k.location + half, math.inf)]
            base = k.operands[0]
    elif isinstance(k, ff.ProductKernel) and any(op.is_operator for op in k.operands):
        # Move the other factors inside the operator, then recurse on it.
        i = next(i for i, op in enumerate(k.operands) if op.is_operator)
        op = k.operands[i].copy()
        others = k.operands[:i] + k.operands[i + 1 :]
        slot = 1 if _is_off(op.operands[0]) else 0
        for other in others:
            op.operands[slot] = op.operands[slot] * other
        op.operands[slot] = op.operands[slot].canonical()
        return region_of_influence(op, tuple(current))
    else:
        return current, k
    merged = [
        (max(a[0], b[0]), min(a[1], b[1]))
        for a in current
        for b in local
        if not (a[1] < b[0] or a[0] > b[1])
    ]
    return region_of_influence(base, tuple(merged))


def _parametric_window(
    x: np.ndarray,
    unit: str,
    lin_count: int,
    lin_location: float | None,
    quantity: str,
    component: str,
    qualifier: str = "",
) -> tuple[str, str, str]:
    """Describe how linear factors scale the amplitude of a component.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs.
    unit : str
        Input unit.
    lin_count : int
        Number of linear factors.
    lin_location : float or None
        Location of the last linear factor.
    quantity : str
        What is scaled, e.g. ``"amplitude"``.
    component : str
        What it belongs to, e.g. ``"function"``.
    qualifier : str, optional
        Adverb inserted by the caller.

    Returns
    -------
    tuple of str
        Summary fragment, description sentence, extrapolation sentence.
    """
    description = f"The {quantity} of the {component} "
    extrap = f"The {quantity} of the {component} is assumed to continue to "
    if lin_count == 1 and lin_location is not None:
        if lin_location < np.min(x):
            return (
                f"with {qualifier}linearly increasing {quantity}",
                description + f"increases {qualifier}linearly",
                extrap + f"increase {qualifier}linearly",
            )
        where = english_point(lin_location, unit, x)
        if lin_location > np.max(x):
            return (
                f"with {qualifier}linearly decreasing {quantity}",
                description + f"decreases {qualifier}linearly",
                extrap + f"decrease {qualifier}linearly until {where} after which the "
                f"{quantity} of the {component} is assumed to start increasing "
                f"{qualifier}linearly",
            )
        return (
            f"with {quantity} increasing {qualifier}linearly away from {where}",
            description + f"increases {qualifier}linearly away from {where}",
            extrap + f"increase {qualifier}linearly",
        )
    if lin_count <= len(POLY_NAMES):
        name = POLY_NAMES[lin_count - 1]
        return (
            f"with {qualifier}{name}ly varying {quantity}",
            description + f"varies {qualifier}{name}ly",
            extrap + f"vary {qualifier}{name}ly",
        )
    return (
        f"with {quantity} given {qualifier}by a polynomial of degree {lin_count}",
        description + f"is given {qualifier}by a polynomial of degree {lin_count}",
        extrap + f"vary according to a polynomial of degree {lin_count}",
    )


def _within_period(k: ff.PeriodicKernel, unit: str) -> str:
    """Describe the shape of a periodic kernel within one period.

    Parameters
    ----------
    k : PeriodicKernel
        Fitted periodic kernel.
    unit : str
        Input unit.

    Returns
    -------
    str
        One sentence.
    """
    if k.lengthscale > 2:
        return (
            "The shape of this function within each period is very smooth and "
            "resembles a sinusoid"
        )
    # The original's approximate conversion to a lengthscale in input units.
    per_lengthscale = 0.5 * math.exp(k.lengthscale + k.period) / math.pi
    return (
        "The shape of this function within each period has a typical lengthscale of "
        + english_length(per_lengthscale, unit)
    )


@dataclass
class _Census:
    """Counts and summaries of the base kernels in one product."""

    se: int = 0
    lin: int = 0
    per: int = 0
    noise: int = 0
    unknown: int = 0
    lengthscale: float = math.inf
    lin_location: float | None = None
    periodic: list[ff.PeriodicKernel] = field(default_factory=list)


def _census(kernels: list[ff.Kernel]) -> _Census:
    """Count the base kernels of a product and combine their lengthscales.

    Parameters
    ----------
    kernels : list of Kernel
        Factors of one product.

    Returns
    -------
    _Census
        Counts, the combined SE lengthscale (in input units) and the periodic
        factors.
    """
    c = _Census()
    log_ell = math.inf
    for k in kernels:
        if isinstance(k, ff.SqExpKernel):
            c.se += 1
            log_ell = -0.5 * math.log(
                math.exp(-2 * log_ell) + math.exp(-2 * k.lengthscale)
            )
        elif isinstance(k, ff.LinearKernel):
            c.lin += 1
            c.lin_location = k.location
        elif isinstance(k, ff.PeriodicKernel):
            c.per += 1
            c.periodic.append(k)
        elif isinstance(k, ff.NoiseKernel):
            c.noise += 1
        elif not isinstance(k, ff.ConstKernel):
            c.unknown += 1
    c.lengthscale = math.exp(log_ell)
    return c


def _describe_smooth(
    c: _Census, domain: float, monotonic: int, unit: str
) -> tuple[str, list[str], list[str]]:
    """Describe a product of SE kernels, possibly scaled by linear factors.

    Parameters
    ----------
    c : _Census
        The product's census.
    domain : float
        Span of the inputs.
    monotonic : int
        -1, 0 or 1, from :func:`component_stats`.
    unit : str
        Input unit.

    Returns
    -------
    tuple
        Summary, description sentences, extrapolation sentences.
    """
    trend = {1: " monotonically increasing", -1: " monotonically decreasing"}
    adjective = trend.get(monotonic, "")
    adjective_and = adjective.replace(" mono", " and mono")
    length = english_length(c.lengthscale, unit)
    if c.lengthscale > 0.5 * domain:
        summary = f"A very smooth{adjective} function"
        desc = [f"This component is a very smooth{adjective_and} function"]
        extrap = [
            "This component is assumed to continue very smoothly but is also "
            "assumed to be stationary so its distribution will eventually return "
            "to the prior"
        ]
    elif c.lengthscale < domain * 0.005:
        summary = "A rapidly varying smooth function"
        desc = [
            "This component is a rapidly varying but smooth function with a "
            f"typical lengthscale of {length}"
        ]
        extrap = [
            "This component is assumed to continue smoothly but its distribution "
            "is assumed to quickly return to the prior"
        ]
    else:
        summary = f"A smooth{adjective} function"
        desc = [
            f"This component is a smooth{adjective_and} function with a typical "
            f"lengthscale of {length}"
        ]
        extrap = [
            "This component is assumed to continue smoothly but is also assumed "
            "to be stationary so its distribution will return to the prior"
        ]
    extrap.append(
        "The prior distribution places mass on smooth functions with a marginal "
        f"mean of zero and a typical lengthscale of {length}"
    )
    return summary, desc, extrap


def translate_product(
    prod: ff.Kernel, x: np.ndarray, monotonic: int, gradient: float, unit: str
) -> Description:
    """Describe one product of base kernels.

    Parameters
    ----------
    prod : Kernel
        A product of base kernels, or one base kernel.
    x : numpy.ndarray
        Inputs.
    monotonic : int
        -1, 0 or 1.
    gradient : float
        Average slope where the component is active.
    unit : str
        Input unit.

    Returns
    -------
    tuple
        Summary, description sentences, extrapolation sentences.
    """
    prod = prod.canonical()
    c = _census(prod.operands if prod.is_operator else [prod])
    domain = float(np.ptp(x))
    desc: list[str] = []
    extrap: list[str] = []

    def parametric(quantity: str, component: str = "function") -> str:
        """Append the amplitude sentences and return the summary fragment.

        Parameters
        ----------
        quantity : str
            What the linear factors scale.
        component : str, optional
            What it belongs to.

        Returns
        -------
        str
            Summary fragment.
        """
        s, d, e = _parametric_window(
            x, unit, c.lin, c.lin_location, quantity, component
        )
        desc.append(d)
        extrap.append(e)
        return s

    if c.unknown:
        summary = f"This simple AI is not capable of describing the component {prod.structure()}"
        return summary, [summary], [summary]
    if c.noise:
        summary = "Uncorrelated noise"
        desc.append("This component models uncorrelated noise")
        extrap.append(
            "This component assumes the uncorrelated noise will continue indefinitely"
        )
        if c.lin:
            summary += " " + parametric("standard deviation", "noise")
        return summary, desc, extrap
    if c.se == c.lin == c.per == 0:
        return (
            "A constant",
            ["This component is constant"],
            ["This component is assumed to stay constant"],
        )
    if c.se and not c.per:
        summary, desc, extrap = _describe_smooth(c, domain, monotonic, unit)
        if c.lin:
            summary += " " + parametric("marginal standard deviation")
        return summary, desc, extrap
    if c.lin and not c.se and not c.per:
        if c.lin == 1:
            direction = "increasing" if gradient > 0 else "decreasing"
            verb = "increase" if gradient > 0 else "decrease"
            return (
                f"A linearly {direction} function",
                [f"This component is linearly {direction}"],
                [f"This component is assumed to continue to {verb} linearly"],
            )
        if c.lin <= len(POLY_NAMES):
            name = POLY_NAMES[c.lin - 1]
            return (
                f"A {name} polynomial",
                [f"This component is a {name} polynomial"],
                [f"This component is assumed to continue as a {name} polynomial"],
            )
        return (
            f"A polynomial of degree {c.lin}",
            [f"This component is a polynomial of degree {c.lin}"],
            [
                f"This component is assumed to continue as a polynomial of degree {c.lin}"
            ],
        )
    if c.per == 1:
        return _describe_one_periodic(c, domain, unit, desc, extrap, parametric)
    return _describe_many_periodic(c, unit, desc, extrap, parametric)


def _describe_one_periodic(
    c: _Census,
    domain: float,
    unit: str,
    desc: list[str],
    extrap: list[str],
    parametric: Callable[[str], str],
) -> Description:
    """Describe a product containing exactly one periodic kernel.

    Parameters
    ----------
    c : _Census
        The product's census.
    domain : float
        Span of the inputs.
    unit : str
        Input unit.
    desc, extrap : list of str
        Sentence lists, extended in place.
    parametric : callable
        Adds the amplitude sentences for linear factors.

    Returns
    -------
    tuple
        Summary, description sentences, extrapolation sentences.
    """
    k = c.periodic[0]
    period = english_length(math.exp(k.period), unit)
    if not c.se:
        summary = f"A periodic function with a period of {period}"
        if c.lin:
            desc.append(
                f"This component is periodic with a period of {period} but with "
                "varying amplitude"
            )
            extrap.append(
                "This component is assumed to continue periodically with a period "
                f"of {period} but with varying amplitude"
            )
            summary += " but " + parametric("amplitude")
        else:
            desc.append(f"This component is periodic with a period of {period}")
            extrap.append(
                f"This component is assumed to continue periodically with a period of {period}"
            )
        desc.append(_within_period(k, unit))
        return summary, desc, extrap

    # SE times Per: the shape drifts from one period to the next. When the SE
    # lengthscale is short against the period the upper bound on the period
    # goes negative; the original then calls the component "very approximately"
    # periodic.
    z = stats.norm.isf(0.25)
    upper = 1.0 / (math.exp(-k.period) - z / c.lengthscale)
    length = english_length(c.lengthscale, unit)
    extrap.append("This component is assumed to continue to be approximately periodic")
    if upper < 0:
        summary = f"A very approximately periodic function with a period of {period}"
        tail = " and varying marginal standard deviation" if c.lin else ""
        desc.append(
            f"This component is very approximately periodic with a period of {period}{tail}"
        )
        desc.append(
            "Across periods the shape of this function varies smoothly with a "
            f"typical lengthscale of {length}"
        )
        desc.append(
            "Since this lengthscale is small relative to the period this component "
            "may more closely resemble a non-periodic smooth function"
        )
        extrap.append(
            "The shape of the function is assumed to vary smoothly between periods "
            "but will quickly return to the prior"
        )
        if c.lin:
            summary += " and " + parametric("marginal standard deviation")
    else:
        summary = f"An approximately periodic function with a period of {period}"
        tail = " and varying amplitude" if c.lin else ""
        desc.append(
            f"This component is approximately periodic with a period of {period}{tail}"
        )
        if c.lengthscale > 0.5 * domain:
            desc.append(
                "Across periods the shape of this function varies very smoothly"
            )
            extrap.append(
                "The shape of the function is assumed to vary very smoothly between "
                "periods but will eventually return to the prior"
            )
        else:
            desc.append(
                "Across periods the shape of this function varies smoothly with a "
                f"typical lengthscale of {length}"
            )
            extrap.append(
                "The shape of the function is assumed to vary smoothly between "
                "periods but will return to the prior"
            )
        if c.lin:
            summary += " and " + parametric("amplitude")
        desc.append(_within_period(k, unit))
    extrap.append(
        "The prior is entirely uncertain about the phase of the periodic function"
    )
    extrap.append(
        "Consequently the pointwise posterior will appear to lose its periodicity, "
        "but this merely reflects the uncertainty in the shape and phase of the function"
    )
    return summary, desc, extrap


def _describe_many_periodic(
    c: _Census,
    unit: str,
    desc: list[str],
    extrap: list[str],
    parametric: Callable[[str], str],
) -> Description:
    """Describe a product of several periodic kernels.

    Parameters
    ----------
    c : _Census
        The product's census.
    unit : str
        Input unit.
    desc, extrap : list of str
        Sentence lists, extended in place.
    parametric : callable
        Adds the amplitude sentences for linear factors.

    Returns
    -------
    tuple
        Summary, description sentences, extrapolation sentences.
    """
    lead = "An approximate product of" if c.se else "A product of"
    summary = f"{lead} several periodic functions"
    desc.append("This component is a product of several periodic functions")
    extrap.append(
        "This component is assumed to continue as a product of several periodic functions"
    )
    if c.se:
        desc.append(
            "Across periods the shape of this function varies smoothly with a typical "
            f"lengthscale of {english_length(c.lengthscale, unit)}"
        )
        extrap.append(
            "Across periods the shape of this function is assumed to continue to vary "
            "smoothly but will return to the prior"
        )
        extrap.append(
            "The prior is entirely uncertain about the phase of the periodic functions"
        )
    if c.lin:
        summary += " and " + parametric("amplitude")
    for i, k in enumerate(c.periodic):
        label = ORDINALS[i] if i < len(ORDINALS) else f"{i + 1}th"
        period = english_length(math.exp(k.period), unit)
        desc.append(f"The {label} periodic function has a period of {period}")
        desc.append(_within_period(k, unit))
    return summary, desc, extrap


def _interval(interval: tuple[float, float], unit: str, x: np.ndarray) -> str:
    """Write a region of the input axis with its preposition.

    Parameters
    ----------
    interval : tuple of float
        Lower and upper bound; infinite ends are open.
    unit : str
        Input unit.
    x : numpy.ndarray
        Inputs.

    Returns
    -------
    str
        E.g. ``"from Feb 1974 onwards"``.
    """
    lo, hi = interval
    if lo == -math.inf and hi == math.inf:
        return "to the entire input domain"
    if lo == -math.inf:
        return f"until {english_point(hi, unit, x)}"
    if hi == math.inf:
        return f"from {english_point(lo, unit, x)} onwards"
    return f"from {english_point(lo, unit, x)} until {english_point(hi, unit, x)}"


def translate_component(
    k: ff.Kernel, x: np.ndarray, monotonic: int, gradient: float, unit: str
) -> Description:
    """Describe one additive component, changepoints included.

    Parameters
    ----------
    k : Kernel
        One component.
    x : numpy.ndarray
        Inputs.
    monotonic : int
        -1, 0 or 1.
    gradient : float
        Average slope where active.
    unit : str
        Input unit.

    Returns
    -------
    tuple
        Summary, description sentences, extrapolation sentences.
    """
    intervals, base = region_of_influence(k.canonical())
    summary, desc, extrap = translate_product(base, x, monotonic, gradient, unit)
    intervals = sorted(intervals)
    everywhere = intervals == [(-math.inf, math.inf)]
    if not intervals:
        where = (
            "The combination of changepoint operators is such that this simple AI "
            "cannot describe where this component acts"
        )
        summary += ". " + where
    elif len(intervals) == 1:
        where = f"This component applies {_interval(intervals[0], unit, x)}"
        if not everywhere:
            summary += f". This function applies {_interval(intervals[0], unit, x)}"
    else:
        listed = ", ".join(_interval(i, unit, x) for i in intervals[:-1])
        last = _interval(intervals[-1], unit, x)
        where = f"This component applies {listed} and {last}"
        summary += f". This function applies {listed} and {last}"
    if not any(i[1] == math.inf for i in intervals):
        extrap = [
            "This component is assumed to stop before the end of the data and will "
            "therefore be extrapolated as zero"
        ]
    if not everywhere:
        desc.append(where)
    return summary, desc, extrap


# Report


def _p(value: float) -> str:
    """Format a p-value, bold below 0.05 as the original's report did.

    Parameters
    ----------
    value : float
        A p-value.

    Returns
    -------
    str
        Markdown.
    """
    return f"**{value:.3f}**" if value < 0.05 else f"{value:.3f}"


@dataclass
class Report:
    """The decomposition, its statistics, its description and its checks.

    Parameters
    ----------
    name : str
        Dataset name.
    model : GPModel
        The simplified model described.
    parts : list of Kernel
        Its components.
    order : dict
        Output of :func:`order_by_mae`.
    stats : list of dict
        Output of :func:`component_stats`, in report order.
    texts : list of tuple
        Summary, description and extrapolation per component, in report order.
    checks : list of ComponentCheck, optional
        Model checks in report order, when computed.
    figures : list of str
        Figure file names relative to the report.
    """

    name: str
    model: GPModel
    parts: list[ff.Kernel]
    order: dict[str, Any]
    stats: list[dict[str, float]]
    texts: list[Description]
    checks: list[ComponentCheck] | None = None
    figures: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        """Render the report as Markdown.

        Returns
        -------
        str
            The document.
        """
        n = len(self.parts)
        plural = "s" if n != 1 else ""
        lines = [
            f"# An automatic report for the dataset: {self.name}",
            "",
            "*Produced by `autostat` (automatic-statistician-revisited), a Python 3 "
            "port of the Automatic Bayesian Covariance Discovery system of Lloyd et "
            "al. (AAAI 2014).*",
            "",
            "## Executive summary",
            "",
            f"The structure search selected `{self.model.kernel.structure()}` "
            f"(Bayesian information criterion {self.model.bic:.1f}; negative log "
            f"likelihood {self.model.nll:.1f} on {self.model.ndata} points). It "
            f"decomposes into {n} additive component{plural}, listed by how much "
            "each reduces extrapolation error:",
            "",
        ]
        lines += [f"{i + 1}. {t[0]}." for i, t in enumerate(self.texts)]
        lines += [
            "",
            "| # | R² cumulative (%) | Residual variance explained (%) | "
            "Cross-validated MAE | MAE reduction (%) |",
            "|---|---|---|---|---|",
            f"| data | | | {self.order['mav_data']:.3g} | |",
        ]
        for i, s in enumerate(self.stats):
            lines.append(
                f"| {i + 1} | {s['cum_var']:.1f} | {s['cum_resid_var']:.1f} | "
                f"{self.order['maes'][i]:.3g} | {self.order['mae_reductions'][i]:.1f} |"
            )
        lines += ["", "MAE: mean absolute error over ten contiguous held-out blocks."]
        if self.figures:
            lines += ["", f"![Data and full posterior]({self.figures[0]})"]
        lines += ["", "## Detailed discussion of the components", ""]
        for i, (summary, desc, extrap) in enumerate(self.texts):
            s = self.stats[i]
            lines += [f"### Component {i + 1}: {summary}", ""]
            lines += [". ".join(desc) + ".", ""]
            lines += [
                f"This component explains {s['var_explained']:.1f}% of the data's "
                f"variance on its own; with the components above it, "
                f"{s['cum_var']:.1f}%. Adding it changes the cross-validated MAE by "
                f"{-self.order['mae_reductions'][i]:.1f}%.",
                "",
                "*Extrapolation.* " + ". ".join(extrap) + ".",
                "",
            ]
            if len(self.figures) > i + 1:
                lines += [f"![Component {i + 1}]({self.figures[i + 1]})", ""]
        lines += self._checking_lines()
        return "\n".join(lines)

    def _checking_lines(self) -> list[str]:
        """Render the model-checking section.

        Returns
        -------
        list of str
            Markdown lines.
        """
        lines = ["## Model checking", ""]
        if not self.checks:
            lines += [
                "Not computed for this report (`describe(..., check=False)`).",
                "",
            ]
            return lines
        lines += [
            "Each statistic compares draws of a component's posterior with draws "
            "of its prior. A small p-value (bold below 0.05) points at a property "
            "of the data that the component's prior does not reproduce.",
            "",
            "| # | QQ max | QQ min | ACF min | ACF min lag | Periodogram max | "
            "Periodogram peak | MMD |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, c in enumerate(self.checks):
            values = " | ".join(
                _p(v)
                for v in (
                    c.qq_max,
                    c.qq_min,
                    c.acf_min,
                    c.acf_min_loc,
                    c.pxx_max,
                    c.pxx_max_loc,
                    c.mmd,
                )
            )
            lines.append(f"| {i + 1} | {values} |")
        lines += [
            "",
            "QQ: quantile-quantile gaps. ACF: autocorrelation function. MMD: "
            "maximum mean discrepancy two-sample test.",
            "",
        ]
        return lines


def describe(
    model: GPModel,
    x: Any,
    y: Any,
    name: str = "data",
    unit: str = "year",
    out_dir: str | Path | None = None,
    check: bool = True,
    seed: int = 0,
) -> Report:
    """Decompose, describe and check a fitted model; optionally write the report.

    Parameters
    ----------
    model : GPModel
        Fitted, e.g. ``search(...).best``.
    x, y : array_like
        The data it was fitted to.
    name : str, optional
        Dataset name for the title.
    unit : str, optional
        ``"year"`` (inputs in decimal years) or ``""`` (no unit).
    out_dir : str or pathlib.Path, optional
        When given, ``report.md`` and the figures are written there.
    check : bool, optional
        Compute the model-checking p-values (slower).
    seed : int, optional
        Seeds the model-checking draws.

    Returns
    -------
    Report
        Everything computed.
    """
    ys = np.asarray(y, float).ravel()
    xs = np.asarray(x, float).reshape(len(ys), -1)
    model = model.simplified()
    parts = components(model)
    order = order_by_mae(model, parts, xs, ys)
    measured = component_stats(model, parts, order["order"], xs, ys)
    texts = [
        translate_component(parts[j], xs[:, 0], s["monotonic"], s["gradient"], unit)
        for j, s in zip(order["order"], measured, strict=True)
    ]
    ordered = [parts[j] for j in order["order"]]
    checks = check_model(model, ordered, xs, ys, seed=seed) if check else None
    report = Report(name, model, parts, order, measured, texts, checks)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        report.figures = _plot(model, parts, order["order"], xs, ys, out)
        (out / "report.md").write_text(report.markdown())
    return report


def _plot(
    model: GPModel,
    parts: list[ff.Kernel],
    order: list[int],
    x: np.ndarray,
    y: np.ndarray,
    out: Path,
) -> list[str]:
    """Draw the full posterior and each component's posterior as PNG files.

    Parameters
    ----------
    model : GPModel
        Fitted model.
    parts : list of Kernel
        Its components.
    order : list of int
        Report order.
    x, y : numpy.ndarray
        Data.
    out : pathlib.Path
        Destination directory.

    Returns
    -------
    list of str
        File names, full posterior first.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = float(x[:, 0].min()), float(x[:, 0].max())
    grid = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 500)[:, None]
    inside = np.linspace(lo, hi, 500)[:, None]
    names: list[str] = []

    def panel(
        xs: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        title: str,
        fname: str,
        data: np.ndarray | None = None,
    ) -> None:
        """Draw one posterior band, optionally over data points.

        Parameters
        ----------
        xs : numpy.ndarray
            Evaluation grid, shape ``(m, 1)``.
        mean, var : numpy.ndarray
            Posterior moments on the grid.
        title : str
            Figure title.
        fname : str
            File name inside ``out``.
        data : numpy.ndarray, optional
            Values to scatter at the data inputs.
        """
        fig, ax = plt.subplots(figsize=(7, 3))
        sd = np.sqrt(var)
        ax.fill_between(
            xs[:, 0], mean - 2 * sd, mean + 2 * sd, color="#9ecae1", alpha=0.6, lw=0
        )
        ax.plot(xs[:, 0], mean, color="#08519c", lw=1.4)
        if data is not None:
            ax.plot(x[:, 0], data, ".", color="#252525", ms=2.5)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
        names.append(fname)

    mean, var = posterior(model, x, y, grid)
    panel(
        grid,
        mean,
        var,
        "Data and full model posterior with extrapolation",
        "fit.png",
        y,
    )
    for rank, j in enumerate(order):
        m, v = posterior(model, x, y, inside, component=parts[j])
        rest = [p for i, p in enumerate(parts) if i != j]
        # The data with every other component's posterior mean removed, as
        # the original plotted it.
        removed = y - posterior(model, x, y, component=_sum(rest))[0] if rest else y
        title = f"Posterior of component {rank + 1}"
        panel(inside, m, v, title, f"component_{rank + 1}.png", removed)
    return names
