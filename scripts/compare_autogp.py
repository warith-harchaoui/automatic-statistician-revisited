#!/usr/bin/env python3
"""Run AutoGP.jl and its Python port on the same series and compare what they find.

    PYTHON_JULIACALL_THREADS=8 .venv-bench/bin/python scripts/compare_autogp.py \
        --series 01-airline 02-solar --seeds 0 1 2

What this script does
    For each series of the 2014 extrapolation benchmark (first 90 % to
    fit, last 10 % to forecast) and each seed, it fits three samplers with
    the same settings (particles, linear schedule, structure moves, HMC
    rounds):

    * ``julia``: AutoGP.jl v0.1.19 itself, through ``juliacall``;
    * ``python``: the port with AutoGP.jl's HMC (prior gradient only);
    * ``python-full``: the port with the likelihood's gradient in HMC.

    It then scores every fitted population the same way, with the port's
    prediction code (checked against Julia's to 3e-12): RMSE of the mixture
    mean, coverage of the 90 % interval, held-out log density, the weight
    the posterior puts on each base kernel, the evidence estimate and the
    run time.

Why the comparison is statistical
    The samplers draw different random numbers, so they cannot agree draw
    for draw. A faithful port must agree in distribution: across seeds, the
    two sets of scores should overlap. ``tests/test_smc_julia.py`` checks
    the deterministic pieces exactly.

What it produces
    ``results/autogp_comparison.parquet``, one row per series, seed and
    sampler; a summary on stderr.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import juliacall  # noqa: F401  (must load before torch, see juliacall's warning)
import numpy as np
import pandas as pd

from autostat.smc import AutoGP, HMCConfig, Particle, linear_schedule
from autostat.smc import julia as J

LOGGER = logging.getLogger("compare_autogp")
ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "benchmark" / "data" / "series.parquet"
OUT = ROOT / "results" / "autogp_comparison.parquet"
BASE_KERNELS = ("Lin", "GE", "Per", "CP")


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train and test arrays of one benchmark series.

    Parameters
    ----------
    name : str
        Dataset name in ``series.parquet``.

    Returns
    -------
    tuple of numpy.ndarray
        ``x_train, y_train, x_test, y_test``.

    Raises
    ------
    ValueError
        When ``name`` is not a series of the comparison.
    """
    frame = pd.read_parquet(SERIES)
    frame = frame[frame["dataset"] == name].sort_values("step")
    train, test = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
    if train.empty:
        raise ValueError(f"unknown series {name}")
    return (
        train["x"].to_numpy(),
        train["y"].to_numpy(),
        test["x"].to_numpy(),
        test["y"].to_numpy(),
    )


def score(model: AutoGP, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Forecast scores and posterior kernel weights of a fitted population.

    Parameters
    ----------
    model : AutoGP
        Fitted, or loaded with Julia's particles.
    x_test, y_test : numpy.ndarray
        Held-out data.

    Returns
    -------
    dict
        ``rmse``, ``cover90``, ``log_density``, ``ess`` and one
        ``w_<kernel>`` per base kernel: the posterior weight of structures
        that contain it.
    """
    mean, bounds = model.predict(x_test, quantiles=(0.05, 0.95))
    inside = (y_test >= bounds[:, 0]) & (y_test <= bounds[:, 1])
    out = {
        "rmse": float(np.sqrt(np.mean((mean - y_test) ** 2))),
        "cover90": float(inside.mean()),
        "log_density": model.predictive_log_density(x_test, y_test),
        "ess": model.effective_sample_size(),
    }
    structures = model.structures()
    for kernel in BASE_KERNELS:
        out[f"w_{kernel}"] = float(sum(w for w, f in structures if kernel in f))
    out["top_structure"] = structures[0][1]
    out["top_weight"] = structures[0][0]
    return out


def run_julia(name, seed, args, data) -> dict[str, float]:
    """Fit AutoGP.jl and score its particles with the port's predictor.

    Parameters
    ----------
    name : str
        Series name; unused here, kept so both runners share a signature.
    seed : int
        Julia seed.
    args : argparse.Namespace
        Command-line settings: particles, percent, n_mcmc, n_hmc.
    data : tuple of numpy.ndarray
        ``(x, y, x_test, y_test)`` from :func:`load`.

    Returns
    -------
    dict
        ``seconds``, ``log_ml`` and the scores of :func:`score`.
    """
    x, y, x_test, y_test = data
    fitted, seconds = J.fit(
        seed, x, y, args.particles, args.percent, args.n_mcmc, args.n_hmc
    )
    parts, log_ml = J.particles(fitted)
    model = AutoGP(x, y, n_particles=len(parts), seed=seed)
    model.particles = [Particle(tree, z) for _, tree, z in parts]
    model.log_weights = np.log(np.maximum([w for w, _, _ in parts], 1e-300))
    return {"seconds": seconds, "log_ml": log_ml, **score(model, x_test, y_test)}


def run_python(name, seed, args, data, gradient: str) -> dict[str, float]:
    """Fit the port and score it.

    Parameters
    ----------
    name : str
        Series name; unused here, kept so both runners share a signature.
    seed : int
        Seed of the population.
    args : argparse.Namespace
        Command-line settings: particles, percent, n_mcmc, n_hmc, jobs.
    data : tuple of numpy.ndarray
        ``(x, y, x_test, y_test)`` from :func:`load`.
    gradient : str
        HMC gradient mode, ``"prior"`` or ``"full"``.

    Returns
    -------
    dict
        ``seconds``, ``log_ml`` and the scores of :func:`score`.
    """
    x, y, x_test, y_test = data
    start = time.perf_counter()
    model = AutoGP(x, y, n_particles=args.particles, seed=seed, n_jobs=args.jobs)
    model.fit_smc(
        linear_schedule(len(x), args.percent),
        n_mcmc=args.n_mcmc,
        n_hmc=args.n_hmc,
        hmc=HMCConfig(n_exit=1, gradient=gradient),
    )
    seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "log_ml": model.log_marginal_likelihood_estimate(),
        **score(model, x_test, y_test),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the comparison.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    int
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--series", nargs="+", default=["01-airline"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--samplers", nargs="+", default=["julia", "python", "python-full"]
    )
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--percent", type=float, default=0.10)
    parser.add_argument("--n-mcmc", type=int, default=75)
    parser.add_argument("--n-hmc", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    done = pd.read_parquet(args.out) if args.out.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for name in args.series:
        data = load(name)
        for seed in args.seeds:
            for sampler in args.samplers:
                if (
                    not done.empty
                    and (
                        (done["series"] == name)
                        & (done["seed"] == seed)
                        & (done["sampler"] == sampler)
                    ).any()
                ):
                    continue
                if sampler == "julia":
                    result = run_julia(name, seed, args, data)
                else:
                    gradient = "full" if sampler == "python-full" else "prior"
                    result = run_python(name, seed, args, data, gradient)
                row = {"series": name, "seed": seed, "sampler": sampler, **result}
                LOGGER.info(
                    "%s seed %d %-11s rmse %.4g cover90 %.2f logdens %.1f  %.0f s  top %s",
                    name,
                    seed,
                    sampler,
                    row["rmse"],
                    row["cover90"],
                    row["log_density"],
                    row["seconds"],
                    row["top_structure"],
                )
                rows.append(row)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    table = pd.DataFrame(rows)
    summary = table.groupby(["series", "sampler"])[
        [
            "rmse",
            "cover90",
            "log_density",
            "log_ml",
            "seconds",
            "w_Per",
            "w_Lin",
            "w_GE",
            "w_CP",
        ]
    ].median()
    LOGGER.info("\n%s", summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
