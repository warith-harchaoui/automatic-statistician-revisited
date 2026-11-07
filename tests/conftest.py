"""Shared fixtures for the autostat test suite.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autostat import benchmarks
from autostat.model import GPModel, negative_log_likelihood


def nll_at_published(model: GPModel, x: np.ndarray, y: np.ndarray) -> float:
    """Re-score a model at its stored parameters.

    Parameters
    ----------
    model : GPModel
        Model with every parameter set.
    x, y : numpy.ndarray
        Data.

    Returns
    -------
    float
        Negative log marginal likelihood.
    """
    theta = torch.as_tensor(model.theta_vector())
    with torch.no_grad():
        return float(
            negative_log_likelihood(
                model, torch.as_tensor(x), torch.as_tensor(y), theta
            )
        )


@pytest.fixture(scope="session")
def airline() -> tuple[np.ndarray, np.ndarray]:
    """Return the airline passenger series.

    Returns
    -------
    tuple of numpy.ndarray
        Inputs in years and passenger counts.
    """
    return benchmarks.load("01-airline")


@pytest.fixture(scope="session")
def airline_final() -> GPModel:
    """Return the final published airline model.

    Returns
    -------
    GPModel
        ``SE + Lin + WN x Lin + SE x Per x Lin`` with its 2014 parameters.
    """
    return benchmarks.published("01-airline")[-1]
