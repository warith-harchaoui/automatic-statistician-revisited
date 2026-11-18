"""Compare the detector's breaks with the switches of the 2014 models.

What this script does
    For each of the thirteen series of the 2014 reports it runs the break
    detector the search uses with ``break_starts=3`` (trend and strongest
    periodogram cycle inside the cost) and lists the published model's
    switches: the location of each changepoint, both ends of each change
    window. For every break it prints the distance to the nearest switch
    date, as a share of the series' span.

Usage
    ``python doc/research/changepoints/real.py``. Runs in seconds.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from autostat import Experiment, benchmarks
from autostat import kernels as ff
from autostat.search import data_shape


def walk(kernel: ff.Kernel) -> Iterator[ff.Kernel]:
    """Yield a kernel and every kernel inside it.

    Parameters
    ----------
    kernel : Kernel
        Any kernel.

    Yields
    ------
    Kernel
        The kernel, then its operands, depth first.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    yield kernel
    for inner in getattr(kernel, "operands", None) or []:
        yield from walk(inner)


def switch_dates(kernel: ff.Kernel) -> list[float]:
    """List where a published model switches.

    Parameters
    ----------
    kernel : Kernel
        A fitted kernel.

    Returns
    -------
    list of float
        Changepoint locations and both ends of change windows.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    out = []
    for k in walk(kernel):
        if isinstance(k, ff.ChangeWindowKernel):
            half = math.exp(k.width) / 2
            out += [k.location - half, k.location + half]
        elif isinstance(k, ff.ChangePointKernel):
            out.append(k.location)
    return out


def main() -> int:
    """Print the breaks, the published switches and their distances.

    Returns
    -------
    int
        0.

    Author
    ------
    Warith Harchaoui <warithmetics@deraison.ai>
    """
    for name in benchmarks.names():
        x, y = benchmarks.load(name)
        x = np.asarray(x, float).reshape(-1, 1)
        y = np.asarray(y, float).ravel()
        shape = data_shape(x, y, Experiment.abcd_2014(break_starts=3))
        span = float(np.ptp(x))
        published = switch_dates(benchmarks.published(name)[-1].kernel)
        distances = [
            min((abs(b - s) / span for s in published), default=math.nan)
            for b in shape["break_hints"]
        ]
        period = shape["break_period"]
        print(
            f"{name:18s} period {period:.3g}"
            f" | breaks {', '.join(f'{b:.2f}' for b in shape['break_hints'])}"
            f" | published {', '.join(f'{s:.2f}' for s in published) or '-'}"
            f" | distance/span {', '.join(f'{d:.3f}' for d in distances)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
