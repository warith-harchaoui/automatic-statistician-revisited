"""AutoGP's kernel trees: the grammar, the prior, the covariance and the likelihood.

What this module does
    It holds a covariance structure as AutoGP.jl holds it in a Gen trace: a
    binary tree whose nodes are numbered like a heap (the root is 1, the
    children of node ``i`` are ``2i`` and ``2i + 1``). Each node stores its
    type and, for a leaf or a changepoint, its parameters on the unconstrained
    scale on which the prior is a standard normal. From that tree it computes
    the log prior, the covariance matrix and the Gaussian log likelihood,
    in ``torch`` so that Hamiltonian Monte Carlo has gradients.

Why the representation copies Gen's
    Every quantity the sampler needs (prior, proposal densities, the moves
    that cut and graft subtrees) is defined in AutoGP.jl on these addresses.
    Keeping the same addresses makes the port checkable line by line and lets
    a tree travel to and from Julia unchanged.

What it consumes
    Times rescaled to ``[0, 1]`` and values centred and divided by their
    range, as ``AutoGP.GPModel`` does; :class:`Scaling` does both.

What it produces
    :class:`Tree` objects, their scores, covariance matrices and a readable
    formula.

Faithfulness
    Port of ``GP.jl`` and ``Model.jl`` from AutoGP.jl v0.1.19 (Apache-2.0,
    (c) 2023 Google LLC, Feras Saad and co-authors). The base kernels,
    the node probabilities, the parameter transforms and the observation
    noise floor are theirs, value for value.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

# Node type codes, as in ``GP.GPConfig``.
CONSTANT, LINEAR, SQUARED_EXPONENTIAL, GAMMA_EXPONENTIAL, PERIODIC = 1, 2, 3, 4, 5
PLUS, TIMES, CHANGEPOINT = 6, 7, 8

LEAVES = (CONSTANT, LINEAR, SQUARED_EXPONENTIAL, GAMMA_EXPONENTIAL, PERIODIC)
OPERATORS = (PLUS, TIMES, CHANGEPOINT)
NAMES = {
    CONSTANT: "Const",
    LINEAR: "Lin",
    SQUARED_EXPONENTIAL: "SE",
    GAMMA_EXPONENTIAL: "GE",
    PERIODIC: "Per",
    PLUS: "+",
    TIMES: "*",
    CHANGEPOINT: "CP",
}

#: Parameters of each node type, in the order of the Julia struct fields.
FIELDS: dict[int, tuple[str, ...]] = {
    CONSTANT: ("value",),
    LINEAR: ("intercept", "bias", "amplitude"),
    SQUARED_EXPONENTIAL: ("lengthscale", "amplitude"),
    GAMMA_EXPONENTIAL: ("lengthscale", "gamma", "amplitude"),
    PERIODIC: ("lengthscale", "period", "amplitude"),
    PLUS: (),
    TIMES: (),
    CHANGEPOINT: ("location",),
}

#: Width of the changepoint's sigmoid, fixed in ``Model.covariance_prior``.
CHANGEPOINT_SCALE = 0.001

#: Added to the observation noise variance, ``Model.JITTER``.
NOISE_FLOOR = 1e-5

_LOG_2PI = math.log(2 * math.pi)


def _normalise(weights: list[float]) -> np.ndarray:
    """Divide by the sum.

    Parameters
    ----------
    weights : list of float
        Non-negative weights.

    Returns
    -------
    numpy.ndarray
        Probabilities.
    """
    array = np.asarray(weights, dtype=float)
    return array / array.sum()


@dataclass(frozen=True)
class Config:
    """The prior over structures and parameters, ``GP.GPConfig``.

    Parameters
    ----------
    node_dist_leaf : numpy.ndarray
        Node type probabilities at the maximum depth (types 1 to 5).
    node_dist_nocp : numpy.ndarray
        Probabilities where no changepoint may appear (types 1 to 7).
    node_dist_cp : numpy.ndarray
        Probabilities where a changepoint may appear (types 1 to 8).
    max_depth : int
        Deepest level of the tree; -1 leaves it unbounded.
    changepoints : bool
        Whether the root may be a changepoint.
    noise : float or None
        A fixed observation noise variance; None infers it.
    gamma_scale, gamma_mu, gamma_sigma : float
        The logit-normal prior of the gamma-exponential exponent.
    period_mu, period_sigma : float
        The log-normal prior of periods.
    wildcard_mu, wildcard_sigma : float
        The log-normal prior of every other positive parameter.
    """

    node_dist_leaf: np.ndarray = field(
        default_factory=lambda: _normalise([0.0, 1, 0, 1, 1])
    )
    node_dist_nocp: np.ndarray = field(
        default_factory=lambda: _normalise([0.0, 6, 0, 6, 6, 5, 5])
    )
    node_dist_cp: np.ndarray = field(
        default_factory=lambda: _normalise([0.0, 6, 0, 6, 6, 4, 4, 2])
    )
    max_depth: int = -1
    changepoints: bool = True
    noise: float | None = None
    gamma_scale: float = 2.0
    gamma_mu: float = 0.0
    gamma_sigma: float = 1.0
    period_mu: float = -1.5
    period_sigma: float = 1.0
    wildcard_mu: float = -1.5
    wildcard_sigma: float = 1.0

    def node_dist(self, idx: int, changepoints: bool) -> np.ndarray:
        """Node type probabilities at a node, ``Model.get_node_dist``.

        Parameters
        ----------
        idx : int
            Heap index of the node.
        changepoints : bool
            Whether a changepoint is allowed there.

        Returns
        -------
        numpy.ndarray
            Probabilities of types ``1 .. len``; index 0 is type 1.
        """
        if depth_of(idx) == self.max_depth:
            return self.node_dist_leaf
        return self.node_dist_cp if changepoints else self.node_dist_nocp

    def transform(self, name: str, z: torch.Tensor | float) -> torch.Tensor:
        """Map an unconstrained parameter to its value, ``Model.transform_param``.

        Parameters
        ----------
        name : str
            Field name.
        z : torch.Tensor or float
            Value on the standard normal scale.

        Returns
        -------
        torch.Tensor
            The parameter.
        """
        z = torch.as_tensor(z, dtype=torch.float64)
        if name == "gamma":
            return self.gamma_scale * torch.sigmoid(
                self.gamma_mu + self.gamma_sigma * z
            )
        if name == "period":
            return torch.exp(self.period_mu + self.period_sigma * z)
        return torch.exp(self.wildcard_mu + self.wildcard_sigma * z)

    def untransform(self, name: str, value: float) -> float:
        """Inverse of :meth:`transform`.

        Parameters
        ----------
        name : str
            Field name.
        value : float
            The parameter.

        Returns
        -------
        float
            Value on the standard normal scale.
        """
        if name == "gamma":
            logit = math.log(value / (self.gamma_scale - value))
            return (logit - self.gamma_mu) / self.gamma_sigma
        if name == "period":
            return (math.log(value) - self.period_mu) / self.period_sigma
        return (math.log(value) - self.wildcard_mu) / self.wildcard_sigma


def depth_of(idx: int) -> int:
    """Depth of a heap index, the root being at depth 1.

    Parameters
    ----------
    idx : int
        Heap index.

    Returns
    -------
    int
        ``1 + floor(log2(idx))``.
    """
    return idx.bit_length()


def children(idx: int) -> tuple[int, int]:
    """Heap indices of the two children.

    Parameters
    ----------
    idx : int
        Heap index.

    Returns
    -------
    tuple of int
        ``(2 idx, 2 idx + 1)``.
    """
    return 2 * idx, 2 * idx + 1


@dataclass(frozen=True)
class Node:
    """One node: its type and its parameters on the standard normal scale.

    Parameters
    ----------
    kind : int
        Type code.
    z : tuple of float
        One value per entry of ``FIELDS[kind]``.
    """

    kind: int
    z: tuple[float, ...] = ()


class Tree(Mapping[int, Node]):
    """A covariance structure, as a map from heap index to :class:`Node`.

    Parameters
    ----------
    nodes : mapping of int to Node
        Every node, the root at index 1.
    """

    def __init__(self, nodes: Mapping[int, Node]) -> None:
        """Store a copy of the nodes.

        Parameters
        ----------
        nodes : mapping of int to Node
            Every node, the root at index 1.
        """
        self._nodes = dict(nodes)

    def __getitem__(self, idx: int) -> Node:
        """The node at ``idx``.

        Parameters
        ----------
        idx : int
            Heap index.

        Returns
        -------
        Node
            The node stored there.
        """
        return self._nodes[idx]

    def __iter__(self) -> Iterator[int]:
        """Heap indices in increasing order.

        Returns
        -------
        iterator of int
            The sorted indices.
        """
        return iter(sorted(self._nodes))

    def __len__(self) -> int:
        """Number of nodes, AutoGP's ``size``.

        Returns
        -------
        int
            The node count.
        """
        return len(self._nodes)

    def __eq__(self, other: object) -> bool:
        """Same nodes at the same indices.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            Whether ``other`` is a :class:`Tree` with the same nodes.
        """
        return isinstance(other, Tree) and self._nodes == other._nodes

    def __hash__(self) -> int:
        """Hash of the node set.

        Returns
        -------
        int
            Hash of the ``(index, node)`` pairs.
        """
        return hash(frozenset(self._nodes.items()))

    def __repr__(self) -> str:
        """The formula, with parameters.

        Returns
        -------
        str
            ``Tree(<formula>)``.
        """
        return f"Tree({self.formula()})"

    def subtree(self, idx: int) -> list[int]:
        """Indices of the subtree rooted at ``idx``, preorder.

        Parameters
        ----------
        idx : int
            Root of the subtree.

        Returns
        -------
        list of int
            Indices present in the tree.
        """
        out, stack = [], [idx]
        while stack:
            i = stack.pop()
            if i in self._nodes:
                out.append(i)
                if self._nodes[i].kind in OPERATORS:
                    left, right = children(i)
                    stack += [right, left]
        return out

    def height(self, idx: int = 1) -> int:
        """Height of the subtree at ``idx``, a leaf having height 1.

        Parameters
        ----------
        idx : int, optional
            Root of the subtree.

        Returns
        -------
        int
            AutoGP's ``depth`` of that node.
        """
        return max(depth_of(i) for i in self.subtree(idx)) - depth_of(idx) + 1

    def moved(self, source: int, target: int) -> dict[int, Node]:
        """The subtree at ``source``, re-indexed to hang at ``target``.

        Parameters
        ----------
        source : int
            Root of the subtree.
        target : int
            Its new root index.

        Returns
        -------
        dict
            Nodes at their new indices, ``extract_subtree_choices``.
        """
        out = {}
        for i in self.subtree(source):
            # A node ``level`` steps below ``source`` sits at
            # ``source * 2**level + offset``; it keeps its offset.
            level = depth_of(i) - depth_of(source)
            offset = i - (source << level)
            out[(target << level) + offset] = self._nodes[i]
        return out

    def replaced(self, idx: int, nodes: Mapping[int, Node]) -> Tree:
        """A copy with the subtree at ``idx`` replaced by ``nodes``.

        Parameters
        ----------
        idx : int
            Root of the subtree to drop.
        nodes : mapping
            New nodes, rooted at ``idx``.

        Returns
        -------
        Tree
            ``replace_subtree_choices``.
        """
        drop = set(self.subtree(idx))
        kept = {i: n for i, n in self._nodes.items() if i not in drop}
        kept.update(nodes)
        return Tree(kept)

    def formula(self, idx: int = 1, config: Config | None = None) -> str:
        """A readable formula; with ``config``, parameter values too.

        Parameters
        ----------
        idx : int, optional
            Root of the subtree.
        config : Config, optional
            Used to show transformed parameter values.

        Returns
        -------
        str
            For example ``(Lin + (Per * GE))``.
        """
        node = self._nodes[idx]
        if node.kind in LEAVES:
            if config is None:
                return NAMES[node.kind]
            values = ", ".join(
                f"{float(config.transform(f, z)):.3g}"
                for f, z in zip(FIELDS[node.kind], node.z, strict=True)
            )
            return f"{NAMES[node.kind]}({values})"
        left, right = (self.formula(c, config) for c in children(idx))
        if node.kind == CHANGEPOINT:
            where = (
                ""
                if config is None
                else f", {float(config.transform('location', node.z[0])):.3g}"
            )
            return f"CP({left}, {right}{where})"
        return f"({left} {NAMES[node.kind]} {right})"


def node_log_prior(
    tree: Tree, config: Config, idx: int, changepoints: bool, stop: int | None = None
) -> float:
    """Log prior of the subtree at ``idx``, ``Model.covariance_prior``.

    Parameters
    ----------
    tree : Tree
        The structure.
    config : Config
        The prior.
    idx : int
        Root of the subtree.
    changepoints : bool
        Whether a changepoint may sit at ``idx``.
    stop : int, optional
        An index whose subtree is left out of the sum.

    Returns
    -------
    float
        Log probability of the node types plus the standard normal log
        densities of the parameters; ``-inf`` for an impossible tree.
    """
    total = 0.0
    stack = [(idx, changepoints)]
    while stack:
        i, cp = stack.pop()
        if i == stop:
            continue
        node = tree[i]
        dist = config.node_dist(i, cp)
        p = dist[node.kind - 1] if node.kind <= len(dist) else 0.0
        if p <= 0.0:
            return -math.inf
        total += math.log(p) + sum(-0.5 * (z * z + _LOG_2PI) for z in node.z)
        if node.kind in (PLUS, TIMES):
            stack += [(c, False) for c in children(i)]
        elif node.kind == CHANGEPOINT:
            stack += [(c, cp) for c in children(i)]
    return total


def log_prior(tree: Tree, config: Config) -> float:
    """Log prior of a whole structure with its parameters.

    Parameters
    ----------
    tree : Tree
        The structure.
    config : Config
        The prior.

    Returns
    -------
    float
        Log density; ``-inf`` when the structure is impossible.
    """
    return node_log_prior(tree, config, 1, config.changepoints)


def sample_subtree(
    rng: np.random.Generator, config: Config, idx: int, changepoints: bool
) -> dict[int, Node]:
    """Draw a subtree at ``idx`` from the prior.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness.
    config : Config
        The prior.
    idx : int
        Index of the subtree's root.
    changepoints : bool
        Whether a changepoint may sit at ``idx``.

    Returns
    -------
    dict
        Nodes by heap index.
    """
    out: dict[int, Node] = {}
    stack = [(idx, changepoints)]
    while stack:
        i, cp = stack.pop()
        dist = config.node_dist(i, cp)
        kind = int(rng.choice(len(dist), p=dist)) + 1
        z = tuple(float(v) for v in rng.standard_normal(len(FIELDS[kind])))
        out[i] = Node(kind, z)
        if kind in (PLUS, TIMES):
            stack += [(c, False) for c in reversed(children(i))]
        elif kind == CHANGEPOINT:
            stack += [(c, cp) for c in reversed(children(i))]
    return out


def sample_tree(rng: np.random.Generator, config: Config) -> Tree:
    """Draw a whole structure from the prior.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness.
    config : Config
        The prior.

    Returns
    -------
    Tree
        A structure with parameters.
    """
    return Tree(sample_subtree(rng, config, 1, config.changepoints))


def parameter_slots(tree: Tree) -> list[tuple[int, int]]:
    """Every ``(index, field position)`` holding a parameter, in a fixed order.

    Parameters
    ----------
    tree : Tree
        The structure.

    Returns
    -------
    list of tuple
        Slots in heap order, then field order.
    """
    return [(i, k) for i in tree for k in range(len(tree[i].z))]


def with_parameters(tree: Tree, values: np.ndarray) -> Tree:
    """A copy with new parameter values, in :func:`parameter_slots` order.

    Parameters
    ----------
    tree : Tree
        The structure.
    values : numpy.ndarray
        One value per slot.

    Returns
    -------
    Tree
        Same structure, new parameters.
    """
    nodes = dict(tree.items())
    position = 0
    for i in tree:
        width = len(tree[i].z)
        if width:
            z = tuple(float(v) for v in values[position : position + width])
            nodes[i] = Node(tree[i].kind, z)
            position += width
    return Tree(nodes)


def covariance(
    tree: Tree,
    config: Config,
    t: torch.Tensor,
    s: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    idx: int = 1,
) -> torch.Tensor:
    """Covariance between two sets of times, ``GP.eval_cov``.

    Parameters
    ----------
    tree : Tree
        The structure.
    config : Config
        Holds the parameter transforms.
    t : torch.Tensor
        ``(n,)`` times.
    s : torch.Tensor, optional
        ``(m,)`` times; ``t`` when omitted.
    z : torch.Tensor, optional
        Parameters in :func:`parameter_slots` order, for gradients. The
        tree's own values are used when omitted.
    idx : int, optional
        Root of the subtree to evaluate.

    Returns
    -------
    torch.Tensor
        ``(n, m)`` matrix.
    """
    if s is None:
        s = t
    if z is None:
        z = torch.tensor(
            [tree[i].z[k] for i, k in parameter_slots(tree)], dtype=torch.float64
        )
    offsets: dict[int, int] = {}
    position = 0
    for i in tree:
        offsets[i] = position
        position += len(tree[i].z)

    def param(i: int, name: str) -> torch.Tensor:
        """One parameter of a node, on its constrained scale.

        Parameters
        ----------
        i : int
            Heap index of the node.
        name : str
            Parameter name, a field of the node's kind.

        Returns
        -------
        torch.Tensor
            The transformed value of that entry of ``z``.
        """
        k = FIELDS[tree[i].kind].index(name)
        return config.transform(name, z[offsets[i] + k])

    def evaluate(i: int) -> torch.Tensor:
        """Covariance matrix of the subtree rooted at ``i``, recursively.

        Parameters
        ----------
        i : int
            Heap index of the subtree's root.

        Returns
        -------
        torch.Tensor
            ``(n, m)`` matrix between ``t`` and ``s``.
        """
        kind = tree[i].kind
        dt = t[:, None] - s[None, :]
        if kind == CONSTANT:
            return param(i, "value") * torch.ones_like(dt)
        if kind == LINEAR:
            c = param(i, "intercept")
            return param(i, "bias") + param(i, "amplitude") * (
                (t[:, None] - c) * (s[None, :] - c)
            )
        if kind == SQUARED_EXPONENTIAL:
            ell = param(i, "lengthscale")
            return param(i, "amplitude") * torch.exp(-0.5 * dt * dt / ell**2)
        if kind == GAMMA_EXPONENTIAL:
            ell, gamma = param(i, "lengthscale"), param(i, "gamma")
            return param(i, "amplitude") * torch.exp(-((dt.abs() / ell) ** gamma))
        if kind == PERIODIC:
            ell, period = param(i, "lengthscale"), param(i, "period")
            angle = torch.sin(math.pi / period * dt.abs())
            return param(i, "amplitude") * torch.exp(-2.0 / ell**2 * angle**2)
        left, right = (evaluate(c) for c in children(i))
        if kind == PLUS:
            return left + right
        if kind == TIMES:
            return left * right
        location = param(i, "location")
        sig_t = 0.5 * (1 + torch.tanh((location - t) / CHANGEPOINT_SCALE))
        sig_s = 0.5 * (1 + torch.tanh((location - s) / CHANGEPOINT_SCALE))
        return (
            sig_t[:, None] * left * sig_s[None, :]
            + (1 - sig_t)[:, None] * right * (1 - sig_s)[None, :]
        )

    return evaluate(idx)


def noise_variance(config: Config, z_noise: torch.Tensor | float) -> torch.Tensor:
    """Observation noise variance from its unconstrained value.

    Parameters
    ----------
    config : Config
        The prior.
    z_noise : torch.Tensor or float
        Value on the standard normal scale.

    Returns
    -------
    torch.Tensor
        ``transform(z) + NOISE_FLOOR``.
    """
    return config.transform("noise", z_noise) + NOISE_FLOOR


def log_likelihood(
    tree: Tree,
    config: Config,
    t: torch.Tensor,
    y: torch.Tensor,
    z_noise: torch.Tensor | float,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian log likelihood of the data under the zero-mean GP.

    Parameters
    ----------
    tree : Tree
        The structure.
    config : Config
        The prior (parameter transforms).
    t, y : torch.Tensor
        ``(n,)`` scaled times and values.
    z_noise : torch.Tensor or float
        Noise on the standard normal scale.
    z : torch.Tensor, optional
        Kernel parameters, for gradients.

    Returns
    -------
    torch.Tensor
        Scalar; ``-inf`` when the covariance is not positive definite.
    """
    n = t.shape[0]
    if n == 0:
        return torch.zeros((), dtype=torch.float64)
    k = covariance(tree, config, t, z=z)
    k = 0.5 * (k + k.T) + noise_variance(config, z_noise) * torch.eye(
        n, dtype=torch.float64
    )
    chol, info = torch.linalg.cholesky_ex(k)
    if int(info) != 0 or not torch.isfinite(chol).all():
        return torch.tensor(-math.inf, dtype=torch.float64)
    alpha = torch.cholesky_solve(y[:, None], chol)[:, 0]
    return (
        -0.5 * (y @ alpha) - torch.log(torch.diagonal(chol)).sum() - 0.5 * n * _LOG_2PI
    )


@dataclass(frozen=True)
class Scaling:
    """The affine maps AutoGP applies to times and values before inference.

    Times go to ``[0, 1]``; values are centred on their mean and divided by
    their range (``Transforms.LinearTransform(y, 1)``).

    Parameters
    ----------
    t_slope, t_intercept : float
        ``t' = t_slope t + t_intercept``.
    y_slope, y_intercept : float
        ``y' = y_slope y + y_intercept``.
    """

    t_slope: float
    t_intercept: float
    y_slope: float
    y_intercept: float

    @classmethod
    def fit(cls, t: np.ndarray, y: np.ndarray) -> Scaling:
        """The maps for one series.

        Parameters
        ----------
        t, y : numpy.ndarray
            Raw times and values.

        Returns
        -------
        Scaling
            As ``AutoGP.GPModel`` builds them.
        """
        t_slope = 1.0 / (t.max() - t.min())
        y_slope = 1.0 / (y.max() - y.min())
        return cls(t_slope, -t_slope * t.min(), y_slope, -y_slope * y.mean())

    def t(self, t: np.ndarray) -> np.ndarray:
        """Scaled times.

        Parameters
        ----------
        t : numpy.ndarray
            Raw times.

        Returns
        -------
        numpy.ndarray
            ``t_slope * t + t_intercept``.
        """
        return self.t_slope * np.asarray(t, dtype=float) + self.t_intercept

    def y(self, y: np.ndarray) -> np.ndarray:
        """Scaled values.

        Parameters
        ----------
        y : numpy.ndarray
            Raw values.

        Returns
        -------
        numpy.ndarray
            ``y_slope * y + y_intercept``.
        """
        return self.y_slope * np.asarray(y, dtype=float) + self.y_intercept

    def y_back(
        self, mean: np.ndarray, var: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance back on the original scale.

        Parameters
        ----------
        mean, var : numpy.ndarray
            On the scaled axis.

        Returns
        -------
        tuple of numpy.ndarray
            ``unapply_mean_var``.
        """
        return (mean - self.y_intercept) / self.y_slope, var / self.y_slope**2
