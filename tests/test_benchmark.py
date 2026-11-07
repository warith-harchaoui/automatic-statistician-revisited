"""The extrapolation benchmark reproduces the 2014 paper's table.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("score", ROOT / "benchmark" / "score.py")
score = importlib.util.module_from_spec(_spec)
# Registered before execution: dataclasses and pickling look modules up by name.
sys.modules["score"] = score
_spec.loader.exec_module(score)


def test_the_paper_table_is_reproduced_from_the_published_predictions():
    """Standardised RMSE recomputed from the 2014 predictions equals the paper's."""
    assert score.paper_check() < 1e-6


@pytest.mark.parametrize("name", sorted(score.test_values()["dataset"].unique()))
def test_the_published_actuals_are_the_held_out_values(name):
    """The 2014 prediction files and the test CSV files hold the same targets."""
    published = score.published()
    actual = published[(published.dataset == name) & (published.method == "SE")][
        "actual"
    ]
    expected = score.test_values().query("dataset == @name")["actual"]
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())


def test_the_benchmark_has_thirteen_series():
    """The 2014 paper scored thirteen series."""
    assert score.test_values()["dataset"].nunique() == 13


def test_nine_published_methods_cover_every_series():
    """Every 2014 method has a forecast for every series."""
    counts = score.published().groupby("method")["dataset"].nunique()
    assert (len(counts), set(counts)) == (9, {13})


def test_rmse_of_a_perfect_forecast_is_zero():
    """The RMSE helper returns zero when prediction equals actual."""
    frame = pd.DataFrame(
        {
            "dataset": ["a"] * 3,
            "method": ["m"] * 3,
            "step": [0, 1, 2],
            "actual": [1.0, 2.0, 3.0],
            "prediction": [1.0, 2.0, 3.0],
        }
    )
    assert score.rmse_table(frame).loc["a", "m"] == 0.0
