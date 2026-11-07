"""Every model the 2014 search published reproduces its likelihood here.

The strongest test of the port. It needs no tolerance argument beyond
rounding: each level winner of the published run is re-scored at its printed
hyperparameters and must give back the published negative log likelihood.
That pins the kernels, their parameterisation and the marginal likelihood to
GPML's all at once.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import pytest

from autostat import benchmarks
from conftest import nll_at_published

CASES = [
    (name, level)
    for name in benchmarks.names()
    for level in range(len(benchmarks.published(name)))
]


def test_the_package_ships_thirteen_benchmarks():
    """The 2014 paper reported thirteen series."""
    assert len(benchmarks.names()) == 13


def test_every_benchmark_has_ten_published_levels():
    """The published run searched ten levels on every series."""
    assert len(CASES) == 130


@pytest.mark.parametrize(("name", "level"), CASES)
def test_a_published_model_reproduces_its_likelihood(name, level):
    """The port re-scores a published model to its published NLL.

    Solar's covariances reach a condition number of 1e14, where the nine
    printed digits of the hyperparameters limit agreement to about 5e-3
    nats; every other series agrees to 5e-5 or better.
    """
    x, y = benchmarks.load(name)
    model = benchmarks.published(name)[level]
    assert nll_at_published(model, x, y) == pytest.approx(model.nll, abs=1e-2)
