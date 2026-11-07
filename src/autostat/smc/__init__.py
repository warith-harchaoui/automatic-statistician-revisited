"""AutoGP in Python: a weighted population of kernel structures, fitted by SMC.

What this package does
    It ports AutoGP.jl (Saad, Patton, Hoffman, Saurous and Mansinghka,
    "Sequential Monte Carlo Learning for Time Series Structure Discovery",
    ICML 2023) to Python and ``torch``. Where :func:`autostat.search` keeps
    the one structure with the best Bayesian information criterion, this
    keeps a weighted sample from the posterior over structures and averages
    their forecasts. Observations can be added one batch at a time.

    >>> from autostat.smc import AutoGP, linear_schedule
    >>> model = AutoGP(t, y, n_particles=8, seed=0)
    >>> model.fit_smc(linear_schedule(len(t), 0.10), n_mcmc=75, n_hmc=10)
    >>> model.structures()[:3]

Why it sits beside the 2013 and 2014 searches
    The three are different algorithms over overlapping grammars, kept
    side by side so that each can be compared with the others on the same
    data. Nothing here changes :mod:`autostat.search`.

What it exposes
    :class:`AutoGP`, :class:`Config`, :class:`HMCConfig`, :class:`Tree`,
    :func:`linear_schedule`, :func:`logarithmic_schedule`.

Author
------
Warith Harchaoui <warith@deraison.ai>. The ported design is (c) 2023 Google
LLC, Apache-2.0, reproduced in ``LICENSE-AUTOGP``.
"""

from .inference import (
    AutoGP,
    HMCConfig,
    Particle,
    linear_schedule,
    logarithmic_schedule,
)
from .tree import Config, Node, Scaling, Tree

__all__ = [
    "AutoGP",
    "Config",
    "HMCConfig",
    "Node",
    "Particle",
    "Scaling",
    "Tree",
    "linear_schedule",
    "logarithmic_schedule",
]
