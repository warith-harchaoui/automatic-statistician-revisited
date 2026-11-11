"""Features of a series, read from its described model and measured against its context.

What this module does
    :func:`features` turns one :class:`~autostat.describe.Report` into a flat
    row of named values with a fixed schema, ready for a table of many
    series. Two kinds of columns:

    * generic ones, the same for every series: fit and size of the model,
      count of each base kernel and operator, share of variance carried by
      the trend, the cycles, the smooth part and the noise, slope of the
      trend, period of the main cycle, posterior weight of the kept
      structure when the search path is known;
    * columns relative to a :class:`~autostat.context.Context`: for each
      cycle the business expects, whether the model found it, at which
      period and how large; for each dated event, whether a changepoint
      sits near it; the cycles and changepoints nobody expected.

Why it exists
    The model already says what a series is made of, in the units of the
    data. Read against what the business expects, it gives features that
    mean something and say how far to trust them: "the yearly cycle is 9 %
    of the level", "no break near the promotion", "an unexpected cycle of
    2.0 years". Generic extractors such as catch22 give numbers that are
    harder to read and know nothing of the domain.

What it does not do
    It does not search: it reads a model already fitted and described. A
    missing component gives ``None`` (absent), never 0, so that "no cycle"
    and "a cycle of size 0" stay apart.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from . import kernels as ff
from .context import Context
from .model import posterior
from .plausibility import FEATURES
from .plausibility import features as kernel_counts

if TYPE_CHECKING:
    from .describe import Report

#: The four kinds of component the shares are summed over.
KINDS = ("trend", "cycle", "smooth", "noise")


def leaves(kernel: ff.Kernel) -> Iterator[ff.Kernel]:
    """Walk a kernel expression and yield its base kernels and switches.

    Parameters
    ----------
    kernel : Kernel
        Any expression.

    Yields
    ------
    Kernel
        Each base kernel; each changepoint or change window itself comes
        before its operands.
    """
    if isinstance(kernel, ff.ChangePointKernel | ff.ChangeWindowKernel):
        yield kernel
    if kernel.is_operator:
        for operand in kernel.operands:
            yield from leaves(operand)
    else:
        yield kernel


def kind(component: ff.Kernel) -> str:
    """Say what a component is, from the base kernels in it.

    Parameters
    ----------
    component : Kernel
        One additive component of a model.

    Returns
    -------
    str
        ``"cycle"`` when it holds a periodic kernel, ``"smooth"`` when it
        holds a squared-exponential, rational quadratic or gamma-exponential
        one, ``"trend"`` when it holds a linear or constant one, ``"noise"``
        otherwise. A cycle with a growing amplitude is still a cycle.
    """
    found = {type(k) for k in leaves(component)}
    if ff.PeriodicKernel in found:
        return "cycle"
    if found & {ff.SqExpKernel, ff.RQKernel, ff.GammaExpKernel}:
        return "smooth"
    if found & {ff.LinearKernel, ff.ConstKernel}:
        return "trend"
    return "noise"


def periods(component: ff.Kernel) -> list[float]:
    """List the periods of a component's periodic kernels.

    Parameters
    ----------
    component : Kernel
        One component.

    Returns
    -------
    list of float
        Periods in the time unit of the series (``period`` is stored as a
        log).
    """
    return [
        math.exp(k.period)
        for k in leaves(component)
        if isinstance(k, ff.PeriodicKernel)
    ]


def switches(component: ff.Kernel) -> list[float]:
    """List the dates where a component's changepoints and windows switch.

    Parameters
    ----------
    component : Kernel
        One component.

    Returns
    -------
    list of float
        A changepoint's location; a change window's two edges, its centre
        minus and plus half its width ``exp(width)``. An event matches an
        edge, the date something changed, not the middle of a window.
    """
    out = []
    for k in leaves(component):
        if isinstance(k, ff.ChangeWindowKernel):
            half = math.exp(k.width) / 2
            out += [float(k.location) - half, float(k.location) + half]
        elif isinstance(k, ff.ChangePointKernel):
            out.append(float(k.location))
    return out


def size(curve: np.ndarray) -> float:
    """Measure how large a component is: half the spread of its middle 90 %.

    Parameters
    ----------
    curve : numpy.ndarray
        The component's posterior mean.

    Returns
    -------
    float
        Half the distance between its 5th and 95th percentiles, in the
        units of the data: the amplitude for a sinusoid, robust to a spike.
    """
    low, high = np.percentile(curve, [5, 95])
    return float(high - low) / 2


def features(
    report: Report, context: Context | None = None, prefix: str = ""
) -> dict[str, Any]:
    """Read a described model as a row of named features.

    Parameters
    ----------
    report : Report
        Output of :func:`autostat.describe`; it keeps the data it described.
    context : Context, optional
        What the business expects; without it only the generic columns.
    prefix : str, optional
        Prepended to every column name.

    Returns
    -------
    dict
        Column name to value; ``None`` where a component is absent. Shares
        are per cent of the summed variance of the components; sizes are in
        standard deviations of the data (``_sd``) and, when the context says
        the values are positive, in per cent of their mean level
        (``_pct``); periods and gaps are in the time unit of the series.
    """
    x, y = report.x, report.y
    t = x[:, 0]
    spread = float(np.std(y)) or 1.0
    # A size in per cent of the level only means something on a ratio scale.
    level = (
        float(np.mean(np.abs(y))) if context is not None and context.positive else None
    )
    row: dict[str, Any] = {
        "n_points": len(y),
        "time_span": float(t.max() - t.min()),
        "bic": float(report.model.bic),
        "nll": float(report.model.nll),
        "n_parameters": int(report.model.kernel.effective_params),
        "n_components": len(report.parts),
        "structure": report.model.kernel.structure_key(),
    }
    counts = kernel_counts(report.model.kernel)
    row.update(
        {f"count_{name}": int(c) for name, c in zip(FEATURES, counts, strict=True)}
    )

    # One pass over the components in report order: kind, size, share, cycles.
    # One pass over the components in report order. Shares are each
    # component's variance over the sum of all of them: positive, summing to
    # 100, unlike the variance a component explains alone, which can be
    # negative when components are correlated.
    curves, kinds, gradients = [], [], []
    found_cycles: list[tuple[float, float, int]] = []  # period, size in sd, component
    found_switches: set[float] = set()
    for rank, (j, stat) in enumerate(
        zip(report.order["order"], report.stats, strict=True)
    ):
        part = report.parts[j]
        mean, _ = posterior(report.model, x, y, component=part)
        curves.append(mean)
        kinds.append(kind(part))
        gradients.append(float(stat["gradient"]))
        if kinds[-1] == "cycle":
            found_cycles += [(p, size(mean) / spread, rank) for p in periods(part)]
        # Every summand under one switch carries it: count each date once.
        found_switches |= {round(s, 6) for s in switches(part)}
    variances = np.array([np.var(c) for c in curves])
    shares = 100 * variances / max(float(variances.sum()), 1e-300)
    for name in KINDS:
        picked = [i for i, k in enumerate(kinds) if k == name]
        row[f"share_{name}"] = float(shares[picked].sum()) if picked else None
    trend = [g for g, k in zip(gradients, kinds, strict=True) if k == "trend"]
    slope = sum(trend) if trend else None
    row["trend_slope"] = slope
    row["trend_slope_sd_per_unit"] = slope / spread if slope is not None else None
    row["trend_slope_pct_per_unit"] = (
        100 * slope / level if slope is not None and level else None
    )
    main = max(found_cycles, key=lambda c: shares[c[2]], default=None)
    row["main_cycle_period"] = main[0] if main else None
    row["main_cycle_size_sd"] = main[1] if main else None
    row["n_cycles"] = len(found_cycles)
    row["n_changepoints"] = len(found_switches)
    if report.path is not None:
        weights = report.path.plausibility()
        row["weight_kept"] = weights[0].weight
        row["weight_next"] = weights[1].weight if len(weights) > 1 else 0.0
        row["level_settled"] = report.path.settled.level

    if context is not None:
        matched: set[int] = set()
        for cycle in context.cycles:
            close = [
                (i, c)
                for i, c in enumerate(found_cycles)
                if abs(c[0] - cycle.period) <= cycle.tolerance * cycle.period
            ]
            best = max(close, key=lambda ic: shares[ic[1][2]], default=None)
            key = f"cycle_{_slug(cycle.name)}"
            row[f"{key}_present"] = best is not None
            row[f"{key}_period"] = best[1][0] if best else None
            row[f"{key}_size_sd"] = best[1][1] if best else None
            row[f"{key}_size_pct"] = (
                100 * best[1][1] * spread / level if best and level else None
            )
            row[f"{key}_share"] = float(shares[best[1][2]]) if best else None
            matched |= {i for i, _ in close}
        unexpected = [c for i, c in enumerate(found_cycles) if i not in matched]
        row["unexpected_cycles"] = len(unexpected)
        top = max(unexpected, key=lambda c: shares[c[2]], default=None)
        row["unexpected_main_period"] = top[0] if top else None
        explained: set[int] = set()
        dates = sorted(found_switches)
        for event in context.events:
            gaps = [(i, abs(s - event.at)) for i, s in enumerate(dates)]
            nearest = min(gaps, key=lambda ig: ig[1], default=None)
            key = f"event_{_slug(event.name)}"
            hit = nearest is not None and nearest[1] <= event.tolerance
            row[f"{key}_changepoint"] = hit
            row[f"{key}_gap"] = nearest[1] if nearest else None
            if hit:
                explained.add(nearest[0])
        row["unexpected_changepoints"] = len(dates) - len(explained)
        row["positive_violated"] = bool(context.positive and float(np.min(y)) < 0)
    return {f"{prefix}{k}": v for k, v in row.items()}


def _slug(name: str) -> str:
    """Turn a business name into a column-name fragment.

    Parameters
    ----------
    name : str
        E.g. ``"Covid lockdown"``.

    Returns
    -------
    str
        E.g. ``"covid_lockdown"``.
    """
    return "_".join("".join(ch if ch.isalnum() else " " for ch in name.lower()).split())
