"""Where should a changepoint detector look? A pilot on the synthetic families.

What this script does
    It runs PELT (``ruptures``) on the two synthetic families of
    ``doc/research/context_search``, five seeds each: the jump family (a
    trend, a yearly cycle and a level jump of 8 in mid-2014) and the
    quarterly family (a trend and a three-month cycle, no jump). Each
    series is read six ways:

    * ``raw``: the values, piecewise-constant cost;
    * ``stl_trend``, ``stl_resid``: the trend and remainder of a robust STL
      decomposition, piecewise-constant cost;
    * ``global_resid``: the remainder of one least-squares fit of a line
      and two harmonics over the whole span, piecewise-constant cost;
    * ``in_cost``: the values with a line and two harmonics inside the
      cost, each segment its own coefficients;
    * ``stl_linear``: the values minus the STL seasonal part, a line inside
      the cost.

    The period of the harmonics and of STL is not given: it is the highest
    Lomb-Scargle peak of the linearly detrended values, as
    ``Experiment.spectral_starts`` finds it. Every penalty is BIC-like,
    ``(p + 1) log n`` times a robust noise variance read from differences.

    It prints, per family and view, the breaks found and how many fall
    within 0.1 year of mid-2014.

Usage
    ``python doc/research/changepoints/pilot.py``. Needs ``ruptures`` and
    ``statsmodels``, which autostat does not depend on. Runs in seconds.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import ruptures as rpt
from statsmodels.tsa.seasonal import STL

from autostat.search import spectral_periods

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "context_search", HERE.parent / "context_search" / "run.py"
)
families = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(families)

#: The date of the jump the jump family carries.
JUMP = 2014.5


def noise_sd(y: np.ndarray) -> float:
    """Estimate the noise spread from second differences, robustly.

    Second differences remove a level and a slope; the median absolute
    deviation ignores the few large values a jump leaves.

    Parameters
    ----------
    y : numpy.ndarray
        Values ``(n,)``.

    Returns
    -------
    float
        The standard deviation of white noise that would give them.
    """
    return float(np.median(np.abs(np.diff(y, 2))) / 0.6745 / np.sqrt(6))


def design(t: np.ndarray, period: float, harmonics: int = 2) -> np.ndarray:
    """Build the regressors of a line and a few harmonics of one period.

    Parameters
    ----------
    t : numpy.ndarray
        Times ``(n,)``.
    period : float
        The cycle's period, in the units of ``t``.
    harmonics : int
        Sine and cosine pairs.

    Returns
    -------
    numpy.ndarray
        Regressors ``(n, 2 + 2 * harmonics)``.
    """
    columns = [np.ones_like(t), t - t.mean()]
    for j in range(1, harmonics + 1):
        columns += [
            np.sin(2 * np.pi * j * t / period),
            np.cos(2 * np.pi * j * t / period),
        ]
    return np.column_stack(columns)


def dates(t: np.ndarray, ends: list[int]) -> list[float]:
    """Turn segment ends into dates halfway between two samples.

    Parameters
    ----------
    t : numpy.ndarray
        Times ``(n,)``.
    ends : list of int
        Segment ends as ``ruptures`` returns them, the last one ``n``.

    Returns
    -------
    list of float
        One date per break, rounded to the hundredth of a year.
    """
    return [round(float((t[i - 1] + t[i]) / 2), 2) for i in ends[:-1]]


def constant(t: np.ndarray, signal: np.ndarray, sd: float) -> list[float]:
    """Find the breaks of a piecewise-constant mean.

    Parameters
    ----------
    t : numpy.ndarray
        Times ``(n,)``.
    signal : numpy.ndarray
        Values ``(n,)``.
    sd : float
        Noise spread that scales the penalty.

    Returns
    -------
    list of float
        Dates of the breaks.
    """
    algo = rpt.Pelt(model="l2", min_size=3).fit(signal.reshape(-1, 1))
    return dates(t, algo.predict(pen=2 * np.log(t.size) * sd**2))


def regression(t: np.ndarray, y: np.ndarray, x: np.ndarray, sd: float) -> list[float]:
    """Find the breaks of a piecewise linear regression on given regressors.

    Parameters
    ----------
    t : numpy.ndarray
        Times ``(n,)``.
    y : numpy.ndarray
        Values ``(n,)``.
    x : numpy.ndarray
        Regressors ``(n, p)``; each segment gets its own coefficients.
    sd : float
        Noise spread that scales the penalty.

    Returns
    -------
    list of float
        Dates of the breaks.
    """
    p = x.shape[1]
    algo = rpt.Pelt(model="linear", min_size=p + 2).fit(np.column_stack([y, x]))
    return dates(t, algo.predict(pen=(p + 1) * np.log(t.size) * sd**2))


def views(t: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    """Detect breaks in the six readings of one series.

    Parameters
    ----------
    t : numpy.ndarray
        Times ``(n,)``, in years.
    y : numpy.ndarray
        Values ``(n,)``.

    Returns
    -------
    dict
        Breaks found, by reading.
    """
    line = np.column_stack([np.ones_like(t), t - t.mean()])
    detrended = y - line @ np.linalg.lstsq(line, y, rcond=None)[0]
    step = float(np.median(np.diff(t)))
    period = spectral_periods(t, detrended, 2 * step, (t[-1] - t[0]) / 5, 1)[0]
    sd = noise_sd(y)
    stl = STL(y, period=max(2, round(period / step)), robust=True).fit()
    harmonic = design(t, period)
    remainder = y - harmonic @ np.linalg.lstsq(harmonic, y, rcond=None)[0]
    return {
        "raw": constant(t, y, sd),
        "stl_trend": constant(t, np.asarray(stl.trend), sd),
        "stl_resid": constant(t, np.asarray(stl.resid), sd),
        "global_resid": constant(t, remainder, sd),
        "in_cost": regression(t, y, harmonic, sd),
        "stl_linear": regression(t, y - np.asarray(stl.seasonal), line, sd),
    }


def main() -> int:
    """Print the breaks each reading finds on both families.

    Returns
    -------
    int
        0.
    """
    warnings.filterwarnings("ignore")
    for name, make in (("jump", families.jump), ("quarterly", families.quarterly)):
        found: dict[str, list[list[float]]] = {}
        for seed in range(5):
            t, y, _ = make(seed)
            for view, breaks in views(t, y).items():
                found.setdefault(view, []).append(breaks)
        for view, runs in found.items():
            near = sum(any(abs(b - JUMP) < 0.1 for b in r) for r in runs)
            total = sum(map(len, runs))
            print(
                f"{name:9s} {view:12s} breaks {total:3d}  near mid-2014 {near}/5  {runs}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
