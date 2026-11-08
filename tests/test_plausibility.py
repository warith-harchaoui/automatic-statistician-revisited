"""Plausibility of structures, after the Bayesian machine scientist.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autostat import benchmarks
from autostat import kernels as ff
from autostat.plausibility import (
    FEATURES,
    ExpressionPrior,
    features,
    plausibility,
    published_corpus,
)


def _model(structure_kernel, bic):
    """A stand-in model with a given kernel and BIC.

    Parameters
    ----------
    structure_kernel : Kernel
        Its kernel, parameters set.
    bic : float
        Its BIC.

    Returns
    -------
    object
        Something :func:`plausibility` reads like a model.
    """

    class Stub:
        """Holds a kernel and a BIC."""

    stub = Stub()
    stub.kernel, stub.bic = structure_kernel, bic
    return stub


def test_features_count_operators_as_written():
    """A + B + C counts two sums; base kernels count once each."""
    se = ff.SqExpKernel(dimension=0, lengthscale=0.0, sf=0.0)
    lin = ff.LinearKernel(dimension=0, sf=0.0, location=0.0)
    noise = ff.NoiseKernel(sf=0.0)
    counts = dict(zip(FEATURES, features(ff.SumKernel([se, lin, noise])), strict=True))
    assert (counts["+"], counts["SE"], counts["Lin"], counts["WN"]) == (2, 1, 1, 1)


def test_two_bic_apart_is_a_ratio_of_e():
    """Under the flat prior, a BIC 2 higher is e times less plausible."""
    a = ff.SqExpKernel(dimension=0, lengthscale=0.0, sf=0.0)
    b = ff.NoiseKernel(sf=0.0)
    rows = plausibility([_model(a, 100.0), _model(b, 102.0)])
    assert rows[0].weight / rows[1].weight == pytest.approx(math.e)
    assert sum(r.weight for r in rows) == pytest.approx(1.0)


def test_a_structure_met_twice_keeps_its_lowest_bic():
    """Refits of one structure count once, at their best."""
    a = ff.SqExpKernel(dimension=0, lengthscale=0.0, sf=0.0)
    rows = plausibility([_model(a, 103.0), _model(a, 100.0)])
    assert [(r.bic, r.weight) for r in rows] == [(100.0, 1.0)]


def test_call_centre_has_a_close_rival():
    """The published call-centre search keeps a structure the data barely prefer."""
    rows = plausibility(benchmarks.published("07-call-centre"))
    assert rows[0].weight == pytest.approx(0.713, abs=0.01)
    assert rows[1].weight == pytest.approx(0.281, abs=0.01)


def test_a_prior_fitted_on_its_own_corpus_is_flat():
    """When pool and corpus coincide, maximum entropy adds nothing."""
    corpus = published_corpus()
    prior = ExpressionPrior.fit(corpus, corpus)
    assert np.allclose(prior.theta, 0.0, atol=1e-4)
