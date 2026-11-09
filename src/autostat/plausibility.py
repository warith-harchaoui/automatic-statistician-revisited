"""How plausible each structure is, in the manner of the Bayesian machine scientist.

What this module does
    The Bayesian machine scientist (Guimerà et al., Science Advances 2020)
    scores a symbolic expression by its description length: the BIC
    approximation of its marginal likelihood plus minus the log of a prior
    over expressions, then turns those scores into a posterior over
    expressions. Its prior is learned from a corpus of known equations, so
    that expressions resemble those scientists have written. Two of its ideas
    are rebuilt here for kernel expressions, without its code:

    * :func:`plausibility` weighs the structures a search scored by
      ``exp(-BIC / 2)`` times a prior, normalised: a posterior over the
      structures searched, where the search itself keeps only the best;
    * :class:`ExpressionPrior` learns that prior from a corpus, as the
      machine scientist does, by maximum entropy: the prior matches the
      corpus's average count of each base kernel and operator.

What is not used, with why
    The reports use the flat prior. A prior learned by maximum entropy only
    means something when its pool spans the sizes of the corpus. The
    expressions the grammar reaches in two steps carry about one sum each,
    the published winners three; fitted on that pool, the prior learns that
    more terms are better and its posterior prefers more complex structures
    than BIC on solar and gas production. A pool spanning the corpus needs
    the machine scientist's Monte Carlo over expressions, which is not
    ported. :class:`ExpressionPrior` is kept for that use and is
    experimental.

Why it matters
    BIC picks one structure. When the next is close, the report should say
    that the data barely prefer it; a posterior weight says so in one number.

What it consumes
    Fitted models with their BIC, for instance a search's level winners; a
    corpus of kernels for the prior, for instance the 130 level winners the
    2014 system published (:func:`published_corpus`).

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from . import kernels as ff
from .model import GPModel

#: What a kernel expression is counted by: base kernels and operators.
FEATURES = ("SE", "Per", "Lin", "C", "WN", "RQ", "GE", "+", "x", "CP", "CW")

_OPERATORS = {
    ff.SumKernel: "+",
    ff.ProductKernel: "x",
    ff.ChangePointKernel: "CP",
    ff.ChangeWindowKernel: "CW",
}
_BASES = {
    ff.SqExpKernel: "SE",
    ff.PeriodicKernel: "Per",
    ff.LinearKernel: "Lin",
    ff.ConstKernel: "C",
    ff.NoiseKernel: "WN",
    ff.RQKernel: "RQ",
    ff.GammaExpKernel: "GE",
}


def features(kernel: ff.Kernel) -> np.ndarray:
    """Count each base kernel and each operator in an expression.

    An n-ary sum or product counts once per ``+`` or ``x`` written, so
    ``A + B + C`` counts two sums, as a formula would.

    Parameters
    ----------
    kernel : Kernel
        A kernel expression.

    Returns
    -------
    numpy.ndarray
        One count per entry of :data:`FEATURES`.
    """
    counts: Counter[str] = Counter()

    def walk(k: ff.Kernel) -> None:
        """Add the counts of one subexpression.

        Parameters
        ----------
        k : Kernel
            The subexpression.
        """
        for kind, name in _OPERATORS.items():
            if isinstance(k, kind):
                operands = [o for o in k.operands if not isinstance(o, ff.ZeroKernel)]
                written = len(operands) - 1 if name in ("+", "x") else 1
                counts[name] += max(written, 0)
                for operand in operands:
                    walk(operand)
                return
        for kind, name in _BASES.items():
            if isinstance(k, kind):
                counts[name] += 1
                return

    walk(kernel)
    return np.array([counts[name] for name in FEATURES], dtype=float)


@dataclass
class ExpressionPrior:
    """A prior over kernel expressions, ``log p(e) = theta . f(e) - log Z``.

    Parameters
    ----------
    theta : numpy.ndarray
        One weight per entry of :data:`FEATURES`; a negative weight makes
        expressions using that feature less likely a priori.
    """

    theta: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES)))

    def log_prior(self, kernel: ff.Kernel) -> float:
        """Return the unnormalised log prior of an expression.

        The normalising constant is the same for every expression, so it
        cancels in :func:`plausibility`.

        Parameters
        ----------
        kernel : Kernel
            An expression.

        Returns
        -------
        float
            ``theta . f(kernel)``.
        """
        return float(self.theta @ features(kernel))

    @classmethod
    def fit(
        cls, corpus: list[ff.Kernel], pool: list[ff.Kernel], ridge: float = 0.1
    ) -> ExpressionPrior:
        """Learn the prior from a corpus, by maximum entropy over a pool.

        The weights are those of the distribution over ``pool`` whose
        expected feature counts equal the corpus's average counts, with the
        most entropy: the machine scientist's construction, on kernels. A
        small ridge keeps the weights finite for a feature the corpus never
        uses.

        Parameters
        ----------
        corpus : list of Kernel
            Expressions considered typical, e.g. :func:`published_corpus`.
        pool : list of Kernel
            The expressions the prior spreads over, e.g. every structure a
            search scored; it must vary the features the corpus uses.
        ridge : float, optional
            L2 penalty on the weights.

        Returns
        -------
        ExpressionPrior
            The fitted prior.
        """
        target = np.mean([features(k) for k in corpus], axis=0)
        f = np.array([features(k) for k in pool])

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            """Negative log likelihood of the corpus means under the pool family.

            Parameters
            ----------
            theta : numpy.ndarray
                Feature weights.

            Returns
            -------
            float, numpy.ndarray
                The objective and its gradient.
            """
            scores = f @ theta
            top = float(np.max(scores))
            weights = np.exp(scores - top)
            z = float(np.sum(weights))
            expected = (weights @ f) / z
            value = top + math.log(z) - theta @ target + ridge * float(theta @ theta)
            return value, expected - target + 2 * ridge * theta

        found = optimize.minimize(
            objective, np.zeros(len(FEATURES)), jac=True, method="L-BFGS-B"
        )
        return cls(found.x)


@dataclass
class Plausible:
    """One structure's plausibility.

    Parameters
    ----------
    structure : str
        The expression.
    bic : float
        Its best BIC among the models given.
    log_prior : float
        Its log prior, 0 under the flat prior.
    description_length : float
        ``BIC / 2 - log prior``, in nats: lower is more plausible.
    weight : float
        Its posterior weight among the structures given.
    """

    structure: str
    bic: float
    log_prior: float
    description_length: float
    weight: float


def plausibility(
    models: list[GPModel], prior: ExpressionPrior | None = None
) -> list[Plausible]:
    """Weigh the distinct structures among some models by their posterior.

    Parameters
    ----------
    models : list of GPModel
        Scored models; a structure met several times keeps its lowest BIC.
    prior : ExpressionPrior, optional
        A prior over expressions; flat when omitted, so the weights are
        ``exp(-BIC / 2)`` normalised.

    Returns
    -------
    list of Plausible
        One per structure, most plausible first.
    """
    best: dict[str, GPModel] = {}
    for model in models:
        key = model.kernel.simplified().structure()
        if key not in best or model.bic < best[key].bic:
            best[key] = model
    return weigh(
        [
            (
                key,
                float(model.bic),
                0.0 if prior is None else prior.log_prior(model.kernel.simplified()),
            )
            for key, model in best.items()
        ]
    )


def weigh(scored: list[tuple[str, float, float]]) -> list[Plausible]:
    """Turn structures scored by BIC and log prior into posterior weights.

    Parameters
    ----------
    scored : list of (str, float, float)
        Structure, BIC and log prior; a structure met several times keeps
        its lowest description length.

    Returns
    -------
    list of Plausible
        One per structure, most plausible first.
    """
    best: dict[str, tuple[float, float, float]] = {}
    for key, bic, log_prior in scored:
        length = bic / 2 - log_prior
        if key not in best or length < best[key][2]:
            best[key] = (bic, log_prior, length)
    lowest = min(length for _, _, length in best.values())
    weights = np.exp(-(np.array([r[2] for r in best.values()]) - lowest))
    weights /= weights.sum()
    out = [
        Plausible(key, bic, log_prior, length, float(w))
        for (key, (bic, log_prior, length)), w in zip(
            best.items(), weights, strict=True
        )
    ]
    return sorted(out, key=lambda p: p.description_length)


def grammar_pool(depth: int = 2, limit: int = 3000) -> list[ff.Kernel]:
    """List the expressions the 2014 grammar reaches from noise in a few steps.

    This is the space a prior spreads over: fitted on it, the prior says
    which of the expressions the search could write look like the corpus.

    Parameters
    ----------
    depth : int, optional
        Grammar steps from the starting white noise.
    limit : int, optional
        Expressions kept at most, in the order they are reached.

    Returns
    -------
    list of Kernel
        Distinct expressions, in canonical form.
    """
    from . import grammar
    from .search import Experiment

    exp = Experiment.abcd_2014()
    frontier = [ff.NoiseKernel()]
    seen: dict[str, ff.Kernel] = {}
    for _ in range(depth):
        frontier = grammar.expand_kernels(1, frontier, exp.base_kernels, exp.rules)
        for kernel in frontier:
            # Unfitted kernels have no parameters to simplify with; their
            # canonical form is enough to tell structures apart.
            canonical = kernel.canonical()
            seen.setdefault(canonical.structure(), canonical)
        frontier = frontier[:limit]
        if len(seen) >= limit:
            break
    return list(seen.values())[:limit]


def published_corpus() -> list[ff.Kernel]:
    """Return the 130 level winners the 2014 system published, as a corpus.

    Returns
    -------
    list of Kernel
        Ten simplified kernels for each of the thirteen benchmark series.
    """
    from . import benchmarks

    return [
        model.kernel.simplified()
        for name in benchmarks.names()
        for model in benchmarks.published(name)
    ]
