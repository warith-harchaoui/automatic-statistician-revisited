"""The thirteen benchmark time series of the 2014 paper and their published results.

What this module does
    It loads the series the AAAI 2014 paper reported on (airline passengers,
    solar irradiance, Mauna Loa carbon dioxide and ten others) and the
    original search's result file for each.

Why it exists as a separate module
    These files are the yardstick of the port: the tests re-score every
    published model; the replication script reruns the search on them.
    Shipping them inside the package makes both work from an installed copy.

What it consumes
    ``autostat/datasets/*.mat`` (MATLAB files with ``X`` and ``y``) and
    ``autostat/reference/*_result.txt``, copied unchanged from
    ``github.com/jamesrobertlloyd/gpss-research`` (``data/tsdlr-renamed`` and
    ``results/2014-01-14-GPSS-add``). Most series come from Rob Hyndman's
    Time Series Data Library, as the original repository states.

What it produces
    Arrays and parsed models.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import scipy.io

from .legacy import read_results
from .model import GPModel


def names() -> list[str]:
    """List the benchmark series, sorted.

    Returns
    -------
    list of str
        E.g. ``["01-airline", "02-solar", ...]``.
    """
    folder = resources.files("autostat") / "datasets"
    return sorted(p.name[:-4] for p in folder.iterdir() if p.name.endswith(".mat"))


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one benchmark series.

    Parameters
    ----------
    name : str
        One of :func:`names`.

    Returns
    -------
    x : numpy.ndarray
        Inputs in decimal years, shape ``(n, 1)``.
    y : numpy.ndarray
        Values, shape ``(n,)``.

    Raises
    ------
    KeyError
        On an unknown name.
    """
    if name not in names():
        raise KeyError(f"unknown benchmark {name!r}; known: {names()}")
    path = resources.files("autostat") / "datasets" / f"{name}.mat"
    with resources.as_file(path) as local:
        data = scipy.io.loadmat(local)
    return np.asarray(data["X"], float), np.asarray(data["y"], float).ravel()


def published(name: str) -> list[GPModel]:
    """Return the winner of each level of the published 2014 search.

    Parameters
    ----------
    name : str
        One of :func:`names`.

    Returns
    -------
    list of GPModel
        Ten models, level 1 first, with the published negative log likelihood.
    """
    path = resources.files("autostat") / "reference" / f"{name}_result.txt"
    with resources.as_file(path) as local:
        return [level[0] for level in read_results(local) if level]
