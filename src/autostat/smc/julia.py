"""A bridge to AutoGP.jl through ``juliacall``, the reference the port is checked against.

What this module does
    It loads the Julia package AutoGP.jl in the running Python process and
    moves kernel structures both ways: a Gen trace becomes a
    :class:`~autostat.smc.tree.Tree` and back. On top of that it exposes the
    numbers the Python port must reproduce: trace scores, covariance
    matrices, gradients, the acceptance ratio of a structure move, the
    predictive distribution and a full ``fit_smc!`` run.

Why it exists
    Only the original can say whether the port is right. The tests in
    ``tests/test_smc_julia.py`` and ``scripts/compare_autogp.py`` use it;
    nothing in :mod:`autostat.smc` needs it.

What it needs
    ``pip install juliacall``. The first import downloads Julia and
    AutoGP.jl v0.1.19 through ``juliapkg`` (a few minutes, once). Set
    ``PYTHON_JULIACALL_THREADS`` before the import to run particles on
    several threads.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from .tree import FIELDS, Node, Tree

AUTOGP_UUID = "6eb593e7-dfb4-4e48-b98e-d7222cdf0053"
AUTOGP_VERSION = "0.1.19"

_HELPERS = r"""
module AutostatBridge
using AutoGP, Gen
import Random
const Inference = AutoGP.Inference
const Model = AutoGP.Model
const GP = AutoGP.GP

function tree_entries(choices)
    sub = Gen.get_submap(choices, :tree)
    out = Tuple{Int,String,Float64}[]
    for (addr, value) in Gen.get_values_shallow(sub)
        push!(out, (addr[1], String(addr[2]), Float64(value)))
    end
    out
end

function make_choicemap(idx::Vector{Int}, field::Vector{String}, value::Vector{Float64})
    c = Gen.choicemap()
    for (i, f, v) in zip(idx, field, value)
        c[(i, Symbol(f))] = f == "node_type" ? Int(v) : v
    end
    c
end

function make_trace(idx, field, value, z_noise, ts, xs)
    constraints = Gen.choicemap()
    Gen.set_submap!(constraints, :tree, make_choicemap(idx, field, value))
    constraints[:noise] = z_noise
    constraints[:xs] = xs
    trace, = Gen.generate(Model.model, (ts, GP.GPConfig()), constraints)
    trace
end

function simulate_trace(seed, ts, xs)
    AutoGP.seed!(seed)
    trace = Gen.simulate(Model.model, (Float64[], GP.GPConfig()))
    constraints = Gen.choicemap()
    Gen.set_submap!(constraints, :tree, Gen.get_submap(Gen.get_choices(trace), :tree))
    constraints[:noise] = trace[:noise]
    constraints[:xs] = xs
    trace, = Gen.generate(Model.model, (ts, GP.GPConfig()), constraints)
    trace
end

function covariance(trace)
    ts = Gen.get_args(trace)[1]
    GP.compute_cov_matrix_vectorized(trace[], 0.0, ts)
end

function parameter_gradient(trace)
    addrs = Inference.get_gp_parameter_addresses(trace)
    leaf = [a for a in addrs if a != :noise]
    (_, values, grads) = Gen.choice_gradients(trace, Gen.select(leaf...), nothing)
    out = Tuple{Int,String,Float64,Float64}[]
    for (addr, g) in Gen.get_values_shallow(Gen.get_submap(grads, :tree))
        push!(out, (addr[1], String(addr[2]), values[:tree => addr], g))
    end
    (_, nvalues, ngrads) = Gen.choice_gradients(trace, Gen.select(:noise), nothing)
    (out, ngrads[:noise])
end

function propose_move(seed, trace)
    AutoGP.seed!(seed)
    args = (false,)
    (fwd_choices, fwd_score, retval) = Gen.propose(Inference.tree_rejuvenation_proposal, (trace, args...))
    (new_trace, bwd_choices, weight) = Inference.tree_rejuvenation_involution(trace, fwd_choices, retval, args)
    (bwd_score, _) = Gen.assess(Inference.tree_rejuvenation_proposal, (new_trace, args...), bwd_choices)
    move_type = fwd_choices[:move_type]
    detach = move_type == 1 ? fwd_choices[:detach_move] : false
    if move_type == 0
        a = retval[1]; b = retval[1]
    elseif detach
        a = retval[1][2]; b = retval[2][2]
    else
        a = retval[1]; b = retval[2]
    end
    (new_trace, Int(move_type), Bool(detach), Int(a), Int(b), fwd_score, bwd_score, weight)
end

function predict_mvn(trace, ts_new)
    d = Inference.predict_mvn(trace, ts_new)
    (Vector(d.μ), Matrix(d.Σ))
end

function fit(seed, ds, y, n_particles, percent, n_mcmc, n_hmc)
    AutoGP.seed!(seed)
    model = AutoGP.GPModel(ds, y; n_particles=n_particles)
    schedule = AutoGP.Schedule.linear_schedule(length(ds), percent)
    elapsed = @elapsed AutoGP.fit_smc!(model; schedule=schedule, n_mcmc=n_mcmc, n_hmc=n_hmc)
    (model, elapsed)
end

end
"""


@functools.cache
def julia() -> Any:
    """The Julia runtime with AutoGP.jl loaded, imported once.

    Returns
    -------
    module
        ``juliacall.Main`` with the bridge module defined.
    """
    import juliapkg

    juliapkg.add("AutoGP", AUTOGP_UUID, version=AUTOGP_VERSION)
    juliapkg.add("Gen", "ea4f424c-a589-11e8-07c0-fd5c91b9da4a")
    juliapkg.resolve()
    from juliacall import Main as jl

    jl.seval("using AutoGP, Gen")
    jl.seval(_HELPERS)
    return jl


def _bridge() -> Any:
    """The helper module."""
    return julia().AutostatBridge


def tree_from_trace(trace: Any) -> tuple[Tree, float]:
    """Read a Gen trace's structure and noise.

    Parameters
    ----------
    trace : juliacall.AnyValue
        A trace of ``AutoGP.Model.model``.

    Returns
    -------
    tuple
        The :class:`Tree` and the noise on the standard normal scale.
    """
    entries = _bridge().tree_entries(julia().Gen.get_choices(trace))
    kinds: dict[int, int] = {}
    params: dict[int, dict[str, float]] = {}
    for idx, field, value in entries:
        if field == "node_type":
            kinds[int(idx)] = int(value)
        else:
            params.setdefault(int(idx), {})[str(field)] = float(value)
    nodes = {
        i: Node(k, tuple(params.get(i, {})[f] for f in FIELDS[k]))
        for i, k in kinds.items()
    }
    z_noise = float(julia().seval("t -> t[:noise]")(trace))
    return Tree(nodes), z_noise


def trace_from_tree(tree: Tree, z_noise: float, t: np.ndarray, y: np.ndarray) -> Any:
    """Build the Gen trace of a structure on data.

    Parameters
    ----------
    tree : Tree
        Structure and parameters.
    z_noise : float
        Noise on the standard normal scale.
    t, y : numpy.ndarray
        Scaled data.

    Returns
    -------
    juliacall.AnyValue
        The trace.
    """
    idx, field, value = [], [], []
    for i in tree:
        node = tree[i]
        idx.append(i)
        field.append("node_type")
        value.append(float(node.kind))
        for name, z in zip(FIELDS[node.kind], node.z, strict=True):
            idx.append(i)
            field.append(name)
            value.append(z)
    jl = julia()
    return _bridge().make_trace(
        jl.Vector[jl.Int](idx),
        jl.Vector[jl.String](field),
        jl.Vector[jl.Float64](value),
        float(z_noise),
        jl.Vector[jl.Float64](np.asarray(t, dtype=float)),
        jl.Vector[jl.Float64](np.asarray(y, dtype=float)),
    )


def simulate(seed: int, t: np.ndarray, y: np.ndarray) -> Any:
    """A trace whose structure is drawn from AutoGP's prior, scored on data.

    Parameters
    ----------
    seed : int
        Julia seed.
    t, y : numpy.ndarray
        Scaled data.

    Returns
    -------
    juliacall.AnyValue
        The trace.
    """
    jl = julia()
    return _bridge().simulate_trace(
        seed,
        jl.Vector[jl.Float64](np.asarray(t, dtype=float)),
        jl.Vector[jl.Float64](np.asarray(y, dtype=float)),
    )


def score(trace: Any) -> float:
    """Gen's log joint density of a trace."""
    return float(julia().Gen.get_score(trace))


def covariance(trace: Any) -> np.ndarray:
    """The kernel matrix on the trace's times, without noise."""
    return np.asarray(_bridge().covariance(trace))


def gradients(trace: Any) -> tuple[dict[tuple[int, str], tuple[float, float]], float]:
    """Gen's gradients of the log joint with respect to every parameter.

    Parameters
    ----------
    trace : juliacall.AnyValue
        The trace.

    Returns
    -------
    tuple
        ``{(index, field): (value, gradient)}`` for kernel parameters;
        the gradient with respect to the noise.
    """
    leaves, noise = _bridge().parameter_gradient(trace)
    out = {(int(i), str(f)): (float(v), float(g)) for i, f, v, g in leaves}
    return out, float(noise)


def propose(seed: int, trace: Any) -> dict[str, Any]:
    """Draw one structure move in Julia and return everything about it.

    Parameters
    ----------
    seed : int
        Julia seed.
    trace : juliacall.AnyValue
        Current trace.

    Returns
    -------
    dict
        ``new_trace``, ``kind`` (``"replace"``, ``"detach"`` or
        ``"attach"``), ``a``, ``b``, ``forward`` and ``reverse`` (Gen's
        proposal log densities, which include the 1/2 for the move type)
        and ``weight`` (the model's log density ratio).
    """
    new, move_type, detach, a, b, fwd, bwd, weight = _bridge().propose_move(seed, trace)
    kind = "replace" if move_type == 0 else ("detach" if detach else "attach")
    return {
        "new_trace": new,
        "kind": kind,
        "a": int(a),
        "b": int(b),
        "forward": float(fwd),
        "reverse": float(bwd),
        "weight": float(weight),
    }


def predict_mvn(trace: Any, t_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predictive mean and covariance of one trace, on the scaled axis."""
    jl = julia()
    mean, cov = _bridge().predict_mvn(
        trace, jl.Vector[jl.Float64](np.asarray(t_new, dtype=float))
    )
    return np.asarray(mean), np.asarray(cov)


def fit(
    seed: int,
    t: np.ndarray,
    y: np.ndarray,
    n_particles: int,
    percent: float = 0.10,
    n_mcmc: int = 75,
    n_hmc: int = 10,
) -> tuple[Any, float]:
    """Run ``AutoGP.fit_smc!`` with a linear schedule.

    Parameters
    ----------
    seed : int
        Julia seed.
    t, y : numpy.ndarray
        Raw times and values.
    n_particles : int
        Population size.
    percent : float, optional
        Fraction of the data added per step.
    n_mcmc, n_hmc : int, optional
        Structure moves and HMC rounds per step.

    Returns
    -------
    tuple
        The fitted ``AutoGP.GPModel`` and its run time in seconds.
    """
    jl = julia()
    model, elapsed = _bridge().fit(
        seed,
        jl.Vector[jl.Float64](np.asarray(t, dtype=float)),
        jl.Vector[jl.Float64](np.asarray(y, dtype=float)),
        n_particles,
        percent,
        n_mcmc,
        n_hmc,
    )
    return model, float(elapsed)


def particles(model: Any) -> tuple[list[tuple[float, Tree, float]], float]:
    """Read a fitted ``AutoGP.GPModel``: its particles and its evidence estimate.

    Parameters
    ----------
    model : juliacall.AnyValue
        A model returned by :func:`fit`.

    Returns
    -------
    tuple
        ``[(weight, tree, z_noise), ...]`` and
        ``AutoGP.log_marginal_likelihood_estimate(model)``.
    """
    jl = julia()
    weights = np.asarray(jl.AutoGP.particle_weights(model), dtype=float)
    traces = jl.seval("m -> m.pf_state.traces")(model)
    out = []
    for w, trace in zip(weights, traces, strict=True):
        tree, z_noise = tree_from_trace(trace)
        out.append((float(w), tree, z_noise))
    return out, float(jl.AutoGP.log_marginal_likelihood_estimate(model))
