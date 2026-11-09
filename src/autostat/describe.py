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
    A :class:`Report`. On request it writes the report in the format asked:
    ``report.md`` with SVG figures, compiled to ``report.pdf`` by md2star; or
    ``report.tex`` with PDF figures, compiled by ``latexmk`` as the original's
    LaTeX document was. The figures are drawn by ``sprezzature-figures``.

The sentences
    They are the original's. The authors wrote them as a rule system over the
    kernel's structure and fitted parameters; rewording them would make this
    a different tool. Two typos (``conitnue``, ``eigth``) are corrected.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>. The ported design is (c) 2013-2015
James Robert Lloyd, David Duvenaud and Roger Grosse, MIT licence, reproduced
in ``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import math
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from . import kernels as ff
from .checking import ComponentCheck, check_model
from .formula import Formula, formula
from .model import GPModel, posterior
from .stopping import SearchPath, findings

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
    ge: int = 0
    gamma: float = 2.0
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
        elif isinstance(k, ff.GammaExpKernel):
            # Counted as smooth; its correlation falls to exp(-1/2) at
            # l 2**(-1/gamma), where an SE's does at its lengthscale.
            c.se += 1
            c.ge += 1
            c.gamma = min(c.gamma, k.gamma)
            ell = k.lengthscale - math.log(2) / k.gamma
            log_ell = -0.5 * math.log(math.exp(-2 * log_ell) + math.exp(-2 * ell))
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


#: Below this exponent a gamma-exponential factor is described as rough.
ROUGH_GAMMA = 1.95


def translate_product(
    prod: ff.Kernel, x: np.ndarray, monotonic: int, gradient: float, unit: str
) -> Description:
    """Describe one product of base kernels, gamma-exponential factors included.

    A product with a gamma-exponential factor of exponent below
    :data:`ROUGH_GAMMA` is described as the same product with an SE factor,
    with "smooth" turned into "rough" and one sentence on the exponent.

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
    summary, desc, extrap = _translate_product(prod, x, monotonic, gradient, unit)
    factors = prod.canonical()
    census = _census(factors.operands if factors.is_operator else [factors])
    if not census.ge or census.gamma >= ROUGH_GAMMA:
        return summary, desc, extrap

    def rough(text: str) -> str:
        """Say "rough" where the sentence said "smooth".

        Parameters
        ----------
        text : str
            One sentence.

        Returns
        -------
        str
            The sentence for a non-differentiable component.
        """
        return text.replace("smoothly", "roughly").replace("smooth", "rough")

    desc = [rough(d) for d in desc]
    desc.append(
        "Its sample paths are continuous but not differentiable "
        f"(gamma-exponential exponent {census.gamma:.2f})"
    )
    return rough(summary), desc, [rough(e) for e in extrap]


def _translate_product(
    prod: ff.Kernel, x: np.ndarray, monotonic: int, gradient: float, unit: str
) -> Description:
    """Describe one product of base kernels, as the 2014 system did.

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


REPORT_FORMATS = ("markdown", "latex")

_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _p(value: float, latex: bool = False) -> str:
    """Format a p-value, bold below 0.05 as the original's report did.

    Parameters
    ----------
    value : float
        A p-value.
    latex : bool, optional
        Bold in LaTeX rather than in Markdown.

    Returns
    -------
    str
        Markdown, or LaTeX when ``latex``.
    """
    text = f"{value:.3f}"
    if value >= 0.05:
        return text
    return rf"\textbf{{{text}}}" if latex else f"**{text}**"


def tex(text: str) -> str:
    """Escape the characters LaTeX reads as commands.

    Parameters
    ----------
    text : str
        Plain text.

    Returns
    -------
    str
        The same text, safe in a LaTeX paragraph.
    """
    return "".join(_TEX_SPECIAL.get(c, c) for c in text)


def compile_markdown(source: Path) -> Path | None:
    """Compile a Markdown report to PDF with md2star's ``md2pdf``.

    Parameters
    ----------
    source : pathlib.Path
        The ``report.md`` to compile; its figures sit beside it.

    Returns
    -------
    pathlib.Path or None
        The PDF, or None when md2star is not installed (a warning says so).
    """
    exe = shutil.which("md2pdf")
    if exe is None:
        warnings.warn(
            "md2pdf not found (pip install md2star): the Markdown report is "
            "left uncompiled",
            stacklevel=2,
        )
        return None
    target = source.with_suffix(".pdf")
    # md2pdf drives LibreOffice, which does not convert two documents at once
    # well: concurrent reports (parallel examples, an API serving several
    # requests) timed out. One conversion at a time across processes, plus
    # one retry for a conversion that still times out.
    with _machine_lock("md2pdf"):
        try:
            _run([exe, source.name, "-o", target.name], source.parent)
        except RuntimeError:
            _run([exe, source.name, "-o", target.name], source.parent)
    return target


@contextlib.contextmanager
def _machine_lock(name: str) -> Iterator[None]:
    """Hold a lock shared by every process on this machine.

    Parameters
    ----------
    name : str
        What is being serialised; names the lock file in the temporary
        directory.

    Yields
    ------
    None
        While the lock is held. Where ``fcntl`` is missing (Windows), no lock
        is taken.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    path = Path(tempfile.gettempdir()) / f"autostat-{name}.lock"
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def compile_latex(source: Path) -> Path | None:
    """Compile a LaTeX report to PDF with ``latexmk``, then remove its by-products.

    Parameters
    ----------
    source : pathlib.Path
        The ``report.tex`` to compile; its figures sit beside it.

    Returns
    -------
    pathlib.Path or None
        The PDF, or None when ``latexmk`` is not installed (a warning says so).
    """
    exe = shutil.which("latexmk")
    if exe is None:
        warnings.warn(
            "latexmk not found (install a TeX distribution): the LaTeX report "
            "is left uncompiled",
            stacklevel=2,
        )
        return None
    _run(
        [exe, "-pdf", "-interaction=nonstopmode", "-halt-on-error", source.name],
        source.parent,
    )
    _run([exe, "-c", source.name], source.parent)
    return source.with_suffix(".pdf")


def _run(command: list[str], cwd: Path) -> None:
    """Run a compiler and raise with the end of its output when it fails.

    Parameters
    ----------
    command : list of str
        Program and arguments.
    cwd : pathlib.Path
        Working directory.

    Raises
    ------
    RuntimeError
        When the command exits with a non-zero status.
    """
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
        raise RuntimeError(f"{command[0]} failed in {cwd}:\n{tail}")


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
        Figure file names relative to the report, from the last
        :meth:`write`.
    selected_by : str
        Opening words of the executive summary, before the structure.
    x, y : numpy.ndarray, optional
        The data, kept so that :meth:`write` can draw the figures.
    unit : str
        ``"year"`` or ``""``, as :func:`describe`.
    path : SearchPath, optional
        The search's winners level by level, for the stopping study.
    data_figure, bic_figure : str, optional
        The data alone and the BIC per level, from the last :meth:`write`.
    symbolic : bool
        Look for formulas for smooth components with PySR.
    written : Formula, optional
        The formula, kept once computed: symbolic regression is slow and
        both formats show the same one.
    """

    name: str
    model: GPModel
    parts: list[ff.Kernel]
    order: dict[str, Any]
    stats: list[dict[str, float]]
    texts: list[Description]
    checks: list[ComponentCheck] | None = None
    figures: list[str] = field(default_factory=list)
    selected_by: str = "The structure search selected"
    x: np.ndarray | None = field(default=None, repr=False)
    y: np.ndarray | None = field(default=None, repr=False)
    unit: str = "year"
    path: SearchPath | None = None
    data_figure: str | None = None
    bic_figure: str | None = None
    symbolic: bool = False
    written: Formula | None = field(default=None, repr=False)

    def write(
        self,
        out_dir: str | Path,
        report_format: str = "markdown",
        pdf: bool = True,
    ) -> Path:
        """Write the report, its figures and, on request, its PDF.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Destination directory, created if needed.
        report_format : {"markdown", "latex"}, optional
            ``"markdown"`` writes ``report.md`` with SVG figures and compiles
            it with md2star; ``"latex"`` writes ``report.tex`` with PDF
            figures and compiles it with ``latexmk``.
        pdf : bool, optional
            Compile ``report.pdf``. A missing compiler is a warning, not an
            error: the source is written either way.

        Returns
        -------
        pathlib.Path
            The report's source file.

        Raises
        ------
        ValueError
            On an unknown format, or when the data were not kept.
        """
        if report_format not in REPORT_FORMATS:
            raise ValueError(
                f"report_format must be one of {REPORT_FORMATS}, not {report_format!r}"
            )
        if self.x is None or self.y is None:
            raise ValueError("the report keeps no data to draw its figures from")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        latex = report_format == "latex"
        self.figures = _plot(
            self.model,
            self.parts,
            self.order["order"],
            self.x,
            self.y,
            out,
            self.unit,
            ".pdf" if latex else ".svg",
        )
        suffix = ".pdf" if latex else ".svg"
        self.data_figure = plot_data(self.x, self.y, self.name, out, self.unit, suffix)
        # A curve needs two levels at least; one point says nothing.
        self.bic_figure = (
            plot_bic(self.path, out, suffix)
            if self.path is not None and len(self.path.steps) > 1
            else None
        )
        source = out / ("report.tex" if latex else "report.md")
        source.write_text(self.latex() if latex else self.markdown())
        if pdf:
            (compile_latex if latex else compile_markdown)(source)
        return source

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
        ]
        if self.data_figure:
            lines += [f"![The data]({self.data_figure})", ""]
        lines += [
            "## Executive summary",
            "",
            f"{self.selected_by} `{self.model.kernel.structure()}` "
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
        lines += self._formula_lines(latex=False)
        lines += self._path_lines(latex=False)
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

    def latex(self) -> str:
        """Render the report as a LaTeX document.

        The sections, sentences and tables are those of :meth:`markdown`;
        the figures are the PDF files of the last :meth:`write`.

        Returns
        -------
        str
            The document.
        """
        n = len(self.parts)
        plural = "s" if n != 1 else ""
        lines = [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}",
            r"\usepackage[margin=2.5cm]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{caption}",
            r"\usepackage{float}",
            r"\usepackage{amsmath}",
            r"\usepackage{breqn}",
            r"\usepackage[hidelinks]{hyperref}",
            "",
            rf"\title{{An automatic report for the dataset: {tex(self.name)}}}",
            r"\author{\texttt{autostat} (automatic-statistician-revisited)}",
            rf"\date{{{dt.date.today():%d %B %Y}}}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\noindent\emph{Produced by \texttt{autostat}, a Python 3 port of the "
            r"Automatic Bayesian Covariance Discovery system of Lloyd et al. "
            r"(AAAI 2014).}",
            "",
            *(
                self._latex_figure(self.data_figure, "The data")
                if self.data_figure
                else []
            ),
            "",
            r"\section*{Executive summary}",
            "",
            f"{tex(self.selected_by)} "
            rf"\texttt{{{tex(self.model.kernel.structure())}}} "
            f"(Bayesian information criterion {self.model.bic:.1f}; negative log "
            f"likelihood {self.model.nll:.1f} on {self.model.ndata} points). It "
            f"decomposes into {n} additive component{plural}, listed by how much "
            "each reduces extrapolation error:",
            r"\begin{enumerate}",
        ]
        lines += [rf"  \item {tex(t[0])}." for t in self.texts]
        lines += [
            r"\end{enumerate}",
            "",
            r"\begin{center}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\# & $R^2$ cumulative (\%) & Residual variance explained (\%) & "
            r"CV MAE & MAE reduction (\%) \\",
            r"\midrule",
            rf"data & & & {self.order['mav_data']:.3g} & \\",
        ]
        for i, s in enumerate(self.stats):
            lines.append(
                f"{i + 1} & {s['cum_var']:.1f} & {s['cum_resid_var']:.1f} & "
                f"{self.order['maes'][i]:.3g} & "
                rf"{self.order['mae_reductions'][i]:.1f} \\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "",
            "MAE: mean absolute error over ten contiguous held-out blocks; CV: "
            "cross-validated.",
        ]
        lines += self._formula_lines(latex=True)
        lines += self._path_lines(latex=True)
        if self.figures:
            lines += self._latex_figure(self.figures[0], "Data and full posterior")
        lines += ["", r"\section*{Detailed discussion of the components}", ""]
        for i, (summary, desc, extrap) in enumerate(self.texts):
            s = self.stats[i]
            lines += [
                rf"\subsection*{{Component {i + 1}: {tex(summary)}}}",
                "",
                tex(". ".join(desc)) + ".",
                "",
                f"This component explains {s['var_explained']:.1f}\\% of the data's "
                f"variance on its own; with the components above it, "
                f"{s['cum_var']:.1f}\\%. Adding it changes the cross-validated MAE "
                f"by {-self.order['mae_reductions'][i]:.1f}\\%.",
                "",
                r"\emph{Extrapolation.} " + tex(". ".join(extrap)) + ".",
            ]
            if len(self.figures) > i + 1:
                lines += self._latex_figure(self.figures[i + 1], f"Component {i + 1}")
        lines += self._latex_checking_lines()
        lines += ["", r"\end{document}", ""]
        return "\n".join(lines)

    @staticmethod
    def _latex_figure(path: str, caption: str) -> list[str]:
        """Render one full-width figure with its caption.

        Parameters
        ----------
        path : str
            Figure file, relative to the report.
        caption : str
            Caption text.

        Returns
        -------
        list of str
            LaTeX lines.
        """
        return [
            "",
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{{path}}}",
            rf"\caption*{{{tex(caption)}}}",
            r"\end{figure}",
        ]

    def _latex_checking_lines(self) -> list[str]:
        """Render the model-checking section in LaTeX.

        Returns
        -------
        list of str
            LaTeX lines.
        """
        lines = ["", r"\section*{Model checking}", ""]
        if not self.checks:
            return [
                *lines,
                r"Not computed for this report (\texttt{describe(..., check=False)}).",
            ]
        lines += [
            "Each statistic compares draws of a component's posterior with draws "
            "of its prior. A small p-value (bold below 0.05) points at a property "
            "of the data that the component's prior does not reproduce.",
            "",
            r"\begin{center}",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"\# & QQ max & QQ min & ACF min & ACF min lag & Pgram max & "
            r"Pgram peak & MMD \\",
            r"\midrule",
        ]
        for i, c in enumerate(self.checks):
            values = " & ".join(
                _p(v, latex=True)
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
            lines.append(rf"{i + 1} & {values} \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "",
            "QQ: quantile-quantile gaps. ACF: autocorrelation function. Pgram: "
            "periodogram. MMD: maximum mean discrepancy two-sample test.",
        ]
        return lines

    def formula(self) -> Formula:
        """Write the model the search kept as a formula with its fitted values.

        Returns
        -------
        Formula
            See :func:`autostat.formula.formula`.

        Raises
        ------
        ValueError
            When the report keeps no data.
        """
        if self.x is None or self.y is None:
            raise ValueError("the report keeps no data to read the formula from")
        if self.written is None:
            self.written = formula(
                self.model,
                self.parts,
                self.order["order"],
                self.x,
                self.y,
                self.unit,
                self.symbolic,
            )
        return self.written

    def _formula_lines(self, latex: bool) -> list[str]:
        """Render the formula section, in Markdown or in LaTeX.

        Parameters
        ----------
        latex : bool
            LaTeX rather than Markdown.

        Returns
        -------
        list of str
            Lines.
        """
        if self.x is None or self.y is None:
            return []
        f = self.formula()

        def show(equation: str) -> list[str]:
            """Display one equation.

            Parameters
            ----------
            equation : str
                LaTeX math.

            Returns
            -------
            list of str
                Lines.
            """
            if latex:
                return [r"\begin{dmath*}", equation, r"\end{dmath*}"]
            return ["", f"$${equation}$$"]

        unit = "years" if f.unit == "year" else "the input's unit"
        heading = r"\section*{The formula found}" if latex else "## The formula found"
        lines = [
            "",
            heading,
            "",
            "The structure search is a symbolic regression: it searches a grammar of "
            "expressions over kernels and keeps the one with the lowest Bayesian "
            "information criterion. This is the expression it kept, with every fitted "
            f"parameter in place; $t$ is in {unit}.",
        ]
        lines += show(f.model)
        lines += ["", "Each component is a Gaussian process with its own covariance:"]
        for law in f.laws:
            lines += show(law)
        if f.noise:
            lines += show(f.noise)
        if f.switches:
            lines += ["", "The changes of regime weigh the components with:"]
            for definition in f.switches:
                lines += show(definition)
        explicit = [e for e in f.explicit if e]
        if explicit:
            lines += [
                "",
                "These components have a closed form, read from their posterior mean:",
            ]
            for equation in explicit:
                lines += show(equation)
            if any(r"\approx" in equation for equation in explicit):
                lines += [
                    "",
                    "A periodic kernel is exactly a sum of cosines, so a periodic "
                    "component is a Fourier series in its period; it is written up to "
                    "the harmonic after which the rest weighs less than 0.1 % of the "
                    "component, hence the approximate sign.",
                ]
        if len(explicit) < len(f.laws):
            lines += [
                "",
                "The other components are smooth, quasi-periodic or noise without a "
                "short formula: a draw from "
                "such a process has no finite closed form; its covariance above is the "
                "formula.",
            ]
        rows = [
            (
                str(i + 1),
                self.texts[i][0],
                note or "none: the covariance is the formula",
            )
            for i, note in enumerate(f.notes)
        ]
        lines += ["", "Closed form of each component:", ""]
        lines += _table(["#", "component", "closed form"], rows, latex, wide=1)
        return lines

    def _path_lines(self, latex: bool) -> list[str]:
        """Render the search path and the stopping study.

        Parameters
        ----------
        latex : bool
            LaTeX rather than Markdown.

        Returns
        -------
        list of str
            Lines; none when the report has no search path.
        """
        if self.path is None:
            return []
        kept = self.path.kept
        heading = (
            r"\section*{Search path and stopping criterion}"
            if latex
            else "## Search path and stopping criterion"
        )
        lines = [
            "",
            heading,
            "",
            "The 2014 configuration searches a fixed number of levels and keeps the "
            "lowest BIC; it has no stopping rule. The table follows the winner of "
            "each level. A change of structure adds or removes terms; a level that "
            "keeps the structure only refits its parameters. BIC differences read "
            "on the scale of Kass and Raftery (1995): under 2 barely worth "
            "mentioning, 2 to 6 positive, 6 to 10 strong, over 10 very strong.",
        ]
        rows = []
        previous = None
        for step in self.path.steps:
            change = "" if previous is None else f"{step.bic - previous:+.1f}"
            mark = " (kept)" if step.level == kept.level else ""
            rows.append(
                (
                    f"{step.level}{mark}",
                    f"{step.bic:.1f}",
                    change,
                    step.structure,
                    str(step.components),
                    str(step.closed_forms),
                    str(step.parameters),
                )
            )
            previous = step.bic
        lines += [""]
        lines += _table(
            [
                "level",
                "BIC",
                "change",
                "structure",
                "components",
                "closed",
                "params",
            ],
            rows,
            latex,
            wide=3,
        )
        if self.bic_figure:
            if latex:
                lines += self._latex_figure(self.bic_figure, "BIC per search level")
            else:
                lines += ["", f"![BIC per search level]({self.bic_figure})"]
        elif len(self.path.steps) == 1:
            lines += ["", "One level was searched: there is no BIC curve to draw."]
        weights = self.path.plausibility()
        shown = [w for w in weights if w.weight >= 0.001]
        lines += [
            "",
            "Posterior weight of each structure met, exp(-BIC/2) normalised under "
            "a flat prior, after the Bayesian machine scientist (Guimerà et al., "
            "2020): where the search keeps one structure, the weights say how "
            "much the data prefer it to the others."
            + (
                f" The {len(weights) - len(shown)} structure(s) under 0.001 are "
                "left out."
                if len(shown) < len(weights)
                else ""
            ),
            "",
        ]
        lines += _table(
            ["structure", "lowest BIC", "weight"],
            [(w.structure, f"{w.bic:.1f}", f"{w.weight:.3f}") for w in shown],
            latex,
            wide=0,
        )
        sentences = findings(self.path, self.model.ndata)
        if latex:
            lines += ["", r"\begin{itemize}"]
            lines += [rf"\item {tex(sentence)}" for sentence in sentences]
            lines += [r"\end{itemize}"]
        else:
            lines += [""] + [f"- {sentence}" for sentence in sentences]
        return lines

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
    selected_by: str = "The structure search selected",
    report_format: str = "markdown",
    pdf: bool = True,
    path: SearchPath | None = None,
    symbolic: bool = False,
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
        When given, the report and its figures are written there.
    check : bool, optional
        Compute the model-checking p-values (slower).
    seed : int, optional
        Seeds the model-checking draws.
    selected_by : str, optional
        Opening words of the executive summary, before the structure.
    report_format : {"markdown", "latex"}, optional
        See :meth:`Report.write`.
    pdf : bool, optional
        Compile the report to ``report.pdf``; see :meth:`Report.write`.
    path : SearchPath, optional
        The search's winners level by level
        (:func:`autostat.stopping.search_path`), for the stopping study.
    symbolic : bool, optional
        Look for a formula for each smooth component with PySR (optional
        dependency, a few seconds to a minute per component).

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
    report = Report(
        name,
        model,
        parts,
        order,
        measured,
        texts,
        checks,
        selected_by=selected_by,
        x=xs,
        y=ys,
        unit=unit,
        path=path,
        symbolic=symbolic,
    )
    if out_dir is not None:
        report.write(out_dir, report_format, pdf)
    return report


def _table(
    header: list[str], rows: list[tuple[str, ...]], latex: bool, wide: int
) -> list[str]:
    """Render a table in Markdown or LaTeX.

    Parameters
    ----------
    header : list of str
        Column titles.
    rows : list of tuple of str
        Cells, as plain text.
    latex : bool
        LaTeX rather than Markdown.
    wide : int
        Index of the column that wraps (LaTeX), the one holding long text.

    Returns
    -------
    list of str
        Lines.
    """
    if not latex:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return lines
    spec = "".join(
        r"p{0.36\linewidth}" if i == wide else "l" for i in range(len(header))
    )
    lines = [
        r"\begin{center}\footnotesize",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(tex(h) for h in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return lines


def plot_data(
    x: np.ndarray, y: np.ndarray, name: str, out: Path, unit: str, suffix: str
) -> str:
    """Draw the data alone, the figure a report opens with.

    Parameters
    ----------
    x, y : numpy.ndarray
        The data.
    name : str
        Dataset name, for the subtitle.
    out : pathlib.Path
        Destination directory.
    unit : str
        ``"year"`` titles the time axis "Year".
    suffix : str
        ``".svg"`` or ``".pdf"``.

    Returns
    -------
    str
        The file name.
    """
    from sprezzature_figures import make_figure

    rows = [{"x": float(t), "y": float(v)} for t, v in zip(x[:, 0], y, strict=True)]
    fname = f"data{suffix}"
    with contextlib.redirect_stdout(io.StringIO()):
        make_figure(
            "gaussian-process",
            rows,
            out=str(out / fname),
            title="The data",
            subtitle=f"{name}: {len(rows)} observations, before any model.",
            x_label="Year" if unit == "year" else "x",
            y_label="Value",
            fit=False,
            width=900,
            height=380,
            mode="static",
            embed_fonts=False,
        )
    return fname


def plot_bic(path: SearchPath, out: Path, suffix: str) -> str:
    """Draw the BIC of each level's winner, the kept level marked.

    Parameters
    ----------
    path : SearchPath
        The search path.
    out : pathlib.Path
        Destination directory.
    suffix : str
        ``".svg"`` or ``".pdf"``.

    Returns
    -------
    str
        The file name.
    """
    from sprezzature_figures import make_figure

    settled, floor = path.settled, path.kept.bic
    beyond = sum(1 for s in path.steps if s.level > settled.level)
    # Levels within the tolerance of the lowest BIC are drawn at it, so the
    # mark falls where BIC stopped falling rather than on a later refit that
    # gained a few hundredths.
    rows = [
        {
            "epoch": s.level,
            "split": "BIC",
            "loss": floor if s.bic <= floor + path.tolerance else s.bic,
        }
        for s in path.steps
    ]
    fname = f"bic{suffix}"
    with contextlib.redirect_stdout(io.StringIO()):
        make_figure(
            "loss-curve",
            rows,
            out=str(out / fname),
            title="BIC per search level",
            subtitle=(
                "Lower is better. Shaded: levels searched after BIC stopped falling "
                f"(within {path.tolerance:g} of its lowest, drawn at it)."
            ),
            x_label="Search level",
            y_label="BIC of the level's best structure",
            best_label=f"BIC settles: level {settled.level}",
            wasted_label=f"{beyond} level(s) past it"
            if beyond
            else "no level searched past it",
            width=900,
            height=420,
            mode="static",
            embed_fonts=False,
        )
    return fname


def _plot(
    model: GPModel,
    parts: list[ff.Kernel],
    order: list[int],
    x: np.ndarray,
    y: np.ndarray,
    out: Path,
    unit: str = "year",
    suffix: str = ".svg",
) -> list[str]:
    """Draw the full posterior and each component's posterior.

    The figures are ``sprezzature-figures``' ``gaussian-process`` kind, handed
    the posterior computed here: mean and standard deviation on a grid, the
    data as points. Fonts are not embedded, so a report stays small.

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
    unit : str, optional
        ``"year"`` titles the time axis "Year".
    suffix : str, optional
        ``".svg"`` for a Markdown report, ``".pdf"`` for a LaTeX one.

    Returns
    -------
    list of str
        File names, full posterior first.
    """
    from sprezzature_figures import make_figure

    lo, hi = float(x[:, 0].min()), float(x[:, 0].max())
    grid = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 500)[:, None]
    inside = np.linspace(lo, hi, 500)[:, None]
    names: list[str] = []

    def panel(
        xs: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        title: str,
        subtitle: str,
        fname: str,
        data: np.ndarray,
    ) -> None:
        """Draw one posterior band over data points.

        Parameters
        ----------
        xs : numpy.ndarray
            Evaluation grid, shape ``(m, 1)``.
        mean, var : numpy.ndarray
            Posterior moments on the grid.
        title, subtitle : str
            Figure text.
        fname : str
            File name inside ``out``.
        data : numpy.ndarray
            Values to scatter at the data inputs.
        """
        sd = np.sqrt(np.clip(var, 0.0, None))
        rows = [
            {"x": float(t), "y": None, "mean": float(m), "sd": float(v)}
            for t, m, v in zip(xs[:, 0], mean, sd, strict=True)
        ] + [
            {"x": float(t), "y": float(v), "mean": None, "sd": None}
            for t, v in zip(x[:, 0], data, strict=True)
        ]
        # The renderer announces each file on standard output, which the
        # command line keeps for its own results.
        with contextlib.redirect_stdout(io.StringIO()):
            make_figure(
                "gaussian-process",
                rows,
                out=str(out / fname),
                title=title,
                subtitle=subtitle,
                x_label="Year" if unit == "year" else "x",
                y_label="Value",
                width=900,
                height=420,
                mode="static",
                embed_fonts=False,
            )
        names.append(fname)

    mean, var = posterior(model, x, y, grid)
    panel(
        grid,
        mean,
        var,
        "Data and full model posterior with extrapolation",
        "Posterior mean and 95% band; dots are the data.",
        f"fit{suffix}",
        y,
    )
    for rank, j in enumerate(order):
        m, v = posterior(model, x, y, inside, component=parts[j])
        rest = [p for i, p in enumerate(parts) if i != j]
        # The data with every other component's posterior mean removed, as
        # the original plotted it.
        removed = y - posterior(model, x, y, component=_sum(rest))[0] if rest else y
        panel(
            inside,
            m,
            v,
            f"Posterior of component {rank + 1}",
            "Posterior mean and 95% band; dots are the data minus the other "
            "components' posterior means.",
            f"component_{rank + 1}{suffix}",
            removed,
        )
    return names
