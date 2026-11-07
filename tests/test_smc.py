"""The AutoGP port's pure-Python checks: prior, moves, schedules, a small fit.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

import math

import numpy as np
import pytest
import torch

from autostat.smc import (
    AutoGP,
    Config,
    Tree,
    linear_schedule,
    logarithmic_schedule,
    moves,
)
from autostat.smc.tree import (
    CHANGEPOINT,
    LINEAR,
    PERIODIC,
    PLUS,
    Node,
    log_likelihood,
    log_prior,
    sample_tree,
)


def _tree(*pairs: tuple[int, Node]) -> Tree:
    """A tree from ``(index, node)`` pairs."""
    return Tree(dict(pairs))


LIN = Node(LINEAR, (0.1, -0.2, 0.3))
PER = Node(PERIODIC, (0.0, 0.5, -0.5))


def test_the_prior_of_a_leaf_is_its_type_probability_times_normal_densities():
    """One linear leaf: log(1/3 of the non-operator mass) plus three normals."""
    config = Config()
    tree = _tree((1, LIN))
    expected = math.log(6 / 28) + sum(
        -0.5 * (z * z + math.log(2 * math.pi)) for z in LIN.z
    )
    assert log_prior(tree, config) == pytest.approx(expected)


def test_a_changepoint_under_a_sum_is_impossible():
    """Sums and products forbid changepoints below them."""
    tree = _tree(
        (1, Node(PLUS)),
        (2, Node(CHANGEPOINT, (0.0,))),
        (3, LIN),
        (4, LIN),
        (5, PER),
    )
    assert log_prior(tree, Config()) == -math.inf


def test_moving_a_subtree_keeps_its_shape():
    """The subtree at 3 moved to 1 keeps left and right children."""
    tree = _tree((1, Node(PLUS)), (2, LIN), (3, Node(PLUS)), (6, PER), (7, LIN))
    assert tree.moved(3, 1) == {1: Node(PLUS), 2: PER, 3: LIN}
    assert tree.moved(1, 2)[4] == LIN


def test_detach_then_attach_is_the_identity():
    """Detaching b to a, then attaching the dropped part back, restores the tree."""
    old = _tree((1, Node(PLUS)), (2, LIN), (3, Node(PLUS)), (6, PER), (7, LIN))
    new = old.replaced(1, old.moved(3, 1))
    dropped = {i: old[i] for i in set(old.subtree(1)) - set(old.subtree(3))}
    grown = dict(dropped)
    grown.update(new.moved(1, 3))
    assert new.replaced(1, grown) == old


def test_the_acceptance_ratio_is_antisymmetric():
    """Detach and attach have opposite log acceptance ratios."""
    config = Config()
    t = torch.linspace(0, 1, 20, dtype=torch.float64)
    y = torch.sin(6 * t)
    old = _tree((1, Node(PLUS)), (2, LIN), (3, Node(PLUS)), (6, PER), (7, LIN))
    new = old.replaced(1, old.moved(3, 1))

    def score(tree: Tree) -> float:
        return log_prior(tree, config) + float(log_likelihood(tree, config, t, y, 0.0))

    forward = moves.detach_densities(old, new, 1, 3, config)
    backward = moves.attach_densities(new, old, 1, 3, config)
    a = moves.log_acceptance(forward, score, score(old))
    b = moves.log_acceptance(backward, score, score(new))
    assert a == pytest.approx(-b)


def test_proposals_from_prior_trees_are_valid_trees():
    """Every proposed tree has a finite proposal density and well-formed nodes."""
    rng = np.random.default_rng(0)
    config = Config()
    for _ in range(300):
        tree = sample_tree(rng, config)
        move = moves.propose(rng, tree, config)
        assert math.isfinite(move.log_q_forward)
        for i in move.tree:
            if move.tree[i].kind in (PLUS, CHANGEPOINT, 7):
                assert 2 * i in move.tree and 2 * i + 1 in move.tree
        assert 1 in move.tree


@pytest.mark.parametrize(
    ("n", "percent", "expected"),
    [(100, 0.1, list(range(10, 101, 10))), (129, 0.1, [*range(13, 118, 13), 129])],
)
def test_linear_schedule(n, percent, expected):
    """Same checkpoints as ``Schedule.linear_schedule``."""
    assert linear_schedule(n, percent) == expected


def test_logarithmic_schedule_ends_at_n():
    """The last checkpoint is always ``n``."""
    assert logarithmic_schedule(100, 2, 1)[-1] == 100


@pytest.mark.slow
def test_a_small_fit_finds_a_periodic_structure_and_covers():
    """On a noisy sine plus trend, most weight goes to periodic structures.

    Four particles and twenty moves per step are far below AutoGP's usual
    settings, so the test asks for structure and calibrated intervals, not
    for an accurate mean.
    """
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 120)
    y = 0.3 * t + np.sin(2 * np.pi * t) + 0.1 * rng.standard_normal(len(t))
    model = AutoGP(t[:100], y[:100], n_particles=4, seed=0, n_jobs=1)
    model.fit_smc(linear_schedule(100, 0.25), n_mcmc=20, n_hmc=5)
    weight = sum(w for w, f in model.structures() if "Per" in f)
    assert weight > 0.5
    mean, bounds = model.predict(t[100:], quantiles=(0.05, 0.95))
    inside = (y[100:] >= bounds[:, 0]) & (y[100:] <= bounds[:, 1])
    assert inside.mean() > 0.6
    assert np.isfinite(mean).all()
