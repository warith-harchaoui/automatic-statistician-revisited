"""AutoGP's structure moves: subtree replace and detach-attach, with their acceptance.

What this module does
    It proposes a new kernel structure from the current one and returns the
    log Metropolis-Hastings acceptance ratio of that proposal. Two moves are
    used, each half of the time:

    * **subtree replace**: pick a node uniformly, redraw the subtree below it
      from the prior;
    * **detach-attach**: either *detach* (pick a node ``a``, then a node
      ``b`` inside its subtree; move ``b``'s subtree up to ``a``, dropping
      what lay between) or its mirror *attach* (pick ``a``, draw a
      random path down from it to a hole ``b``, grow a new tree around the
      path and hang the old subtree of ``a`` in the hole).

    Detach and attach undo each other, which is what makes the pair an
    involution and the acceptance ratio exact.

Why the ratio is written out in full
    AutoGP.jl lets Gen derive the ratio from the proposal program and the
    involution. Here every proposal density is computed explicitly:
    ``log alpha = log p(new) - log p(old) + log q(reverse) - log q(forward)``.
    The test suite compares that number with the one Gen computes on the same
    moves, through ``juliacall``.

What it consumes
    A :class:`~autostat.smc.tree.Tree` and a scorer of whole structures.

What it produces
    :class:`Move` records holding the proposed tree and its log ratio.

Faithfulness
    Port of ``inference_rejuv_tree*.jl`` and ``inference_utils.jl`` from
    AutoGP.jl v0.1.19 (Apache-2.0, (c) 2023 Google LLC), for the unbiased
    node choice that ``fit_smc!`` uses by default.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .tree import (
    CHANGEPOINT,
    FIELDS,
    LEAVES,
    PLUS,
    TIMES,
    Config,
    Node,
    Tree,
    children,
    depth_of,
    node_log_prior,
    sample_subtree,
)

_LOG_HALF = math.log(0.5)
_LOG_2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class Move:
    """A proposed structure change.

    Parameters
    ----------
    kind : str
        ``"replace"``, ``"detach"`` or ``"attach"``.
    a : int
        The node the move is anchored at.
    b : int
        For detach, the node moved up; for attach, the hole. Equal to ``a``
        for a replace.
    tree : Tree
        The proposed structure.
    log_q_forward : float
        Log density of proposing it.
    log_q_reverse : float
        Log density of proposing the way back from it.
    """

    kind: str
    a: int
    b: int
    tree: Tree
    log_q_forward: float
    log_q_reverse: float


def _log_p_detach(tree: Tree) -> float:
    """Log probability of choosing detach rather than attach."""
    return -math.inf if len(tree) == 1 else _LOG_HALF


def _log_p_attach(tree: Tree) -> float:
    """Log probability of choosing attach rather than detach."""
    return 0.0 if len(tree) == 1 else _LOG_HALF


def _path(a: int, b: int) -> list[int]:
    """Nodes from ``a`` down to ``b``, both included.

    Parameters
    ----------
    a, b : int
        ``b`` lies in the subtree of ``a``.

    Returns
    -------
    list of int
        ``[a, ..., b]``.
    """
    out = [b]
    while out[-1] != a:
        out.append(out[-1] // 2)
    return out[::-1]


def _aux_max_depth(config: Config, height: int) -> int:
    """Depth limit of the attach path, ``max_depth - (height - 1)``."""
    return -1 if config.max_depth == -1 else config.max_depth - (height - 1)


def path_log_prob(a: int, b: int, max_depth: int) -> float:
    """Log density of ``generate_random_path`` reaching ``b`` from ``a``.

    Parameters
    ----------
    a, b : int
        Start and end of the path.
    max_depth : int
        Depth at which the path must stop; -1 for none.

    Returns
    -------
    float
        Each step down costs ``log(1/2)`` for not stopping and ``log(1/2)``
        for the side; stopping costs ``log(1/2)``, or nothing at the limit.
    """
    total = 0.0
    for node in _path(a, b)[:-1]:
        if depth_of(node) == max_depth:
            return -math.inf
        total += 2 * _LOG_HALF
    return total + (0.0 if depth_of(b) == max_depth else _LOG_HALF)


def _sample_path(rng: np.random.Generator, a: int, max_depth: int) -> int:
    """Draw the hole of an attach, as ``generate_random_path`` does.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness.
    a : int
        Start of the path.
    max_depth : int
        Depth at which the path must stop; -1 for none.

    Returns
    -------
    int
        The hole's index.
    """
    node = a
    while not (depth_of(node) == max_depth or rng.random() < 0.5):
        node = children(node)[int(rng.random() >= 0.5)]
    return node


def _aux_dist(
    config: Config, idx: int, on_path: bool, force_cp: bool, changepoints: bool
) -> np.ndarray:
    """Node type probabilities in the attach proposal, ``get_node_dist_attach_detach``.

    Parameters
    ----------
    config : Config
        The prior.
    idx : int
        Node index.
    on_path : bool
        Whether ``idx`` lies on the path above the hole.
    force_cp : bool
        Whether path nodes must be changepoints.
    changepoints : bool
        Whether a changepoint may sit at ``idx``.

    Returns
    -------
    numpy.ndarray
        Probabilities of types ``1 ..``.
    """
    dist = np.array(config.node_dist(idx, changepoints), dtype=float)
    if not on_path:
        return dist
    if force_cp:
        dist = np.zeros_like(dist)
        dist[CHANGEPOINT - 1] = 1.0
        return dist
    dist[: len(LEAVES)] = 0.0
    return dist / dist.sum()


def aux_log_prob(
    nodes: dict[int, Node], a: int, b: int, force_cp: bool, config: Config
) -> float:
    """Log density of the tree grown around the attach path.

    Parameters
    ----------
    nodes : dict
        The grown nodes, without the hole's subtree.
    a, b : int
        Anchor and hole.
    force_cp : bool
        Whether path nodes must be changepoints.
    config : Config
        The prior.

    Returns
    -------
    float
        ``covariance_proposal_attach_detach`` scored on ``nodes``.
    """
    path = set(_path(a, b))
    total = 0.0
    stack = [(a, config.changepoints)]
    while stack:
        idx, cp = stack.pop()
        if idx == b:
            continue
        node = nodes[idx]
        dist = _aux_dist(config, idx, idx in path, force_cp, cp)
        p = dist[node.kind - 1] if node.kind <= len(dist) else 0.0
        if p <= 0.0:
            return -math.inf
        total += math.log(p) + sum(-0.5 * (z * z + _LOG_2PI) for z in node.z)
        if node.kind in (PLUS, TIMES):
            stack += [(c, False) for c in children(idx)]
        elif node.kind == CHANGEPOINT:
            stack += [(c, cp) for c in children(idx)]
    return total


def _sample_aux(
    rng: np.random.Generator, a: int, b: int, force_cp: bool, config: Config
) -> dict[int, Node]:
    """Grow the tree around the attach path.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness.
    a, b : int
        Anchor and hole.
    force_cp : bool
        Whether path nodes must be changepoints.
    config : Config
        The prior.

    Returns
    -------
    dict
        New nodes; nothing at ``b`` or below.
    """
    path = set(_path(a, b))
    out: dict[int, Node] = {}
    stack = [(a, config.changepoints)]
    while stack:
        idx, cp = stack.pop()
        if idx == b:
            continue
        if idx not in path:
            out.update(sample_subtree(rng, config, idx, cp))
            continue
        dist = _aux_dist(config, idx, True, force_cp, cp)
        kind = int(rng.choice(len(dist), p=dist)) + 1
        z = tuple(float(v) for v in rng.standard_normal(len(FIELDS[kind])))
        out[idx] = Node(kind, z)
        child_cp = cp if kind == CHANGEPOINT else False
        stack += [(c, child_cp) for c in reversed(children(idx))]
    return out


def _changepoint_allowed(tree: Tree, idx: int, config: Config) -> bool:
    """Whether subtree replace may put a changepoint at ``idx``."""
    if not config.changepoints:
        return False
    return idx == 1 or tree[idx // 2].kind == CHANGEPOINT


def replace_move(rng: np.random.Generator, tree: Tree, config: Config) -> Move:
    """Propose a subtree replace.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness.
    tree : Tree
        Current structure.
    config : Config
        The prior.

    Returns
    -------
    Move
        The proposal and its densities.
    """
    indices = list(tree)
    a = indices[int(rng.integers(len(indices)))]
    cp = _changepoint_allowed(tree, a, config)
    new = tree.replaced(a, sample_subtree(rng, config, a, cp))
    return replace_densities(tree, new, a, config)


def replace_densities(old: Tree, new: Tree, a: int, config: Config) -> Move:
    """Forward and reverse densities of a subtree replace at ``a``.

    Parameters
    ----------
    old, new : Tree
        Structures before and after.
    a : int
        The replaced node.
    config : Config
        The prior.

    Returns
    -------
    Move
        With both densities.
    """
    cp = _changepoint_allowed(old, a, config)
    forward = -math.log(len(old)) + node_log_prior(new, config, a, cp)
    reverse = -math.log(len(new)) + node_log_prior(old, config, a, cp)
    return Move("replace", a, a, new, forward, reverse)


def detach_densities(old: Tree, new: Tree, a: int, b: int, config: Config) -> Move:
    """Forward and reverse densities of detaching ``b`` up to ``a``.

    Parameters
    ----------
    old, new : Tree
        Structures before and after.
    a, b : int
        Anchor and the node moved up.
    config : Config
        The prior.

    Returns
    -------
    Move
        With both densities.
    """
    forward = _log_p_detach(old) - math.log(len(old)) - math.log(len(old.subtree(a)))
    dropped = set(old.subtree(a)) - set(old.subtree(b))
    aux = {i: old[i] for i in dropped}
    reverse = (
        _log_p_attach(new)
        - math.log(len(new))
        + path_log_prob(a, b, _aux_max_depth(config, new.height(a)))
        + aux_log_prob(aux, a, b, new[a].kind == CHANGEPOINT, config)
    )
    return Move("detach", a, b, new, forward, reverse)


def attach_densities(old: Tree, new: Tree, a: int, b: int, config: Config) -> Move:
    """Forward and reverse densities of attaching the subtree at ``a`` into hole ``b``.

    Parameters
    ----------
    old, new : Tree
        Structures before and after.
    a, b : int
        Anchor and hole.
    config : Config
        The prior.

    Returns
    -------
    Move
        With both densities.
    """
    dropped = set(new.subtree(a)) - set(new.subtree(b))
    aux = {i: new[i] for i in dropped}
    forward = (
        _log_p_attach(old)
        - math.log(len(old))
        + path_log_prob(a, b, _aux_max_depth(config, old.height(a)))
        + aux_log_prob(aux, a, b, old[a].kind == CHANGEPOINT, config)
    )
    reverse = _log_p_detach(new) - math.log(len(new)) - math.log(len(new.subtree(a)))
    return Move("attach", a, b, new, forward, reverse)


def detach_attach_move(rng: np.random.Generator, tree: Tree, config: Config) -> Move:
    """Propose a detach or an attach.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness.
    tree : Tree
        Current structure.
    config : Config
        The prior.

    Returns
    -------
    Move
        The proposal and its densities.
    """
    if len(tree) == 1 and tree.height() == config.max_depth:
        raise ValueError("detach-attach needs max_depth above 1")
    indices = list(tree)
    a = indices[int(rng.integers(len(indices)))]
    if math.isfinite(_log_p_detach(tree)) and rng.random() < 0.5:
        inside = tree.subtree(a)
        b = inside[int(rng.integers(len(inside)))]
        new = tree.replaced(a, tree.moved(b, a))
        return detach_densities(tree, new, a, b, config)
    b = _sample_path(rng, a, _aux_max_depth(config, tree.height(a)))
    force_cp = tree[a].kind == CHANGEPOINT
    grown = _sample_aux(rng, a, b, force_cp, config)
    grown.update(tree.moved(a, b))
    new = tree.replaced(a, grown)
    return attach_densities(tree, new, a, b, config)


def propose(rng: np.random.Generator, tree: Tree, config: Config) -> Move:
    """One structure proposal, ``tree_rejuvenation_proposal``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness.
    tree : Tree
        Current structure.
    config : Config
        The prior.

    Returns
    -------
    Move
        Subtree replace or detach-attach, half of the time each (detach-attach
        is off when ``max_depth`` is 1).
    """
    p_detach_attach = 0.5 if config.max_depth != 1 else 0.0
    if rng.random() < p_detach_attach:
        return detach_attach_move(rng, tree, config)
    return replace_move(rng, tree, config)


def log_acceptance(
    move: Move, score: Callable[[Tree], float], old_score: float
) -> float:
    """Log Metropolis-Hastings ratio of a move.

    Parameters
    ----------
    move : Move
        The proposal.
    score : callable
        Log prior plus log likelihood of a structure, at fixed noise.
    old_score : float
        That score for the current structure.

    Returns
    -------
    float
        ``score(new) - old_score + log q_reverse - log q_forward``.
    """
    if not math.isfinite(move.log_q_reverse):
        return -math.inf
    return score(move.tree) - old_score + move.log_q_reverse - move.log_q_forward
