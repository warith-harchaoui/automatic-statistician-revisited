"""Model checking returns valid p-values and its MMD test discriminates.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest

from autostat.checking import check_model, mmd_test
from autostat.describe import components


def test_mmd_does_not_reject_two_samples_from_one_distribution():
    """Same distribution: the permutation p-value stays large."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((80, 2)), rng.standard_normal((80, 2))
    assert mmd_test(a, b, rng, shuffles=200) > 0.05


def test_mmd_rejects_two_shifted_samples():
    """A shift of two standard deviations gives a tiny p-value."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((80, 2)), rng.standard_normal((80, 2)) + 2.0
    assert mmd_test(a, b, rng, shuffles=200) < 0.01


@pytest.mark.slow
def test_every_airline_check_is_a_probability(airline, airline_final):
    """All seven statistics of every component lie in [0, 1].

    Parameters
    ----------
    airline : tuple of numpy.ndarray
        Fixture: the airline inputs and passenger counts.
    airline_final : GPModel
        Fixture: the final published airline model.
    """
    x, y = airline
    model = airline_final.simplified()
    checks = check_model(model, components(model), x, y, samples=200)
    values = np.array([list(c.as_dict().values()) for c in checks])
    assert ((values >= 0) & (values <= 1)).all()
