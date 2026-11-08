"""Example 1: re-score a model the 2014 system published (no report).

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

import torch

from autostat import benchmarks
from autostat.model import negative_log_likelihood

print(benchmarks.names()[:3])

x, y = benchmarks.load("01-airline")
model = benchmarks.published("01-airline")[-1]  # the final 2014 winner
print(model.kernel.structure())
print(round(model.nll, 3))  # as published in 2014

theta = torch.as_tensor(model.theta_vector())
with torch.no_grad():
    nll = negative_log_likelihood(model, torch.as_tensor(x), torch.as_tensor(y), theta)
print(round(float(nll), 3))
