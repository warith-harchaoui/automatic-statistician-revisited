"""Example 4, step 1: eight years of monthly sales, trend + yearly cycle + noise.

Writes sales.csv next to this script.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
months = pd.date_range("2016-01-01", periods=96, freq="MS")
k = np.arange(96)
units = 100 + 0.5 * k + 8 * np.sin(2 * np.pi * k / 12) + rng.normal(0, 2, 96)
pd.DataFrame({"month": months.strftime("%Y-%m-%d"), "units": units.round(2)}).to_csv(
    Path(__file__).parent / "sales.csv", index=False
)
