"""Why are the port's 90 % intervals too narrow? What widens them honestly.

What this script does
    On the extrapolation benchmark the port's 90 % intervals hold 72 % of the
    held-out tails. This script reads the depth-5 searches already kept in
    ``benchmark/results/searches/abcd_2014-depth5`` (nothing is searched
    again) and, on each of the thirteen series:

    * **Diagnoses** the misses from the standardised errors
      ``z = (y - mean) / sd`` of the tail: their mean says whether the
      forecast is off-centre (bias), their spread whether the band is too
      narrow (scale). A calibrated Gaussian forecast has mean 0, spread 1.
    * **Recalibrates on the training part only.** The kept structure is
      refitted on the first 90 % of the training part, from its own
      parameters, then forecasts the last 10 %, as the benchmark does with
      the whole series. The 90 % quantile of ``|z|`` on that inner tail,
      divided by 1.645 and at least 1, widens the band on the real tail. No
      held-out value is read.
    * **Tempers the mixture.** The BIC weights ``exp(-BIC / 2)`` are raised
      to ``1 / T`` (T = 1 is the mixture of the benchmark), which spreads the
      weight over more structures. T is not chosen on the tail; every value
      is reported, as a sensitivity.

    Each forecast is scored by its RMSE, its coverage and the interval score
    of Gneiting and Raftery (2007) for a 90 % band, ``width + 20 x miss``,
    divided by the mean absolute one-step change of the training part so
    series of different scales can be averaged. Lower is better.

Usage
    ``python doc/research/calibration/run.py`` writes ``results.json`` here.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from autostat import SearchResult, posterior
from autostat.model import bic_weights, fit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import run_2026  # noqa: E402

Z90 = 1.645
TEMPERATURES = (1, 2, 5, 10)
#: Structures a tempered mixture averages, heaviest first.
MAX_STRUCTURES = 20


def score(mean, sd, s) -> dict:
    """Score a Gaussian forecast of the held-out tail.

    Parameters
    ----------
    mean, sd : numpy.ndarray
        Predictive mean and standard deviation.
    s : run_2026.Series
        One benchmark series.

    Returns
    -------
    dict
        RMSE, coverage of the 90 % band, its scaled interval score and the
        mean and spread of the standardised errors.
    """
    lo, hi = mean - Z90 * sd, mean + Z90 * sd
    y = s.y_test
    miss = np.where(y < lo, lo - y, 0.0) + np.where(y > hi, y - hi, 0.0)
    scale = np.mean(np.abs(np.diff(s.y_train)))
    z = (y - mean) / sd
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "interval_score": float(np.mean(hi - lo + 2 / 0.1 * miss) / scale),
        "z_mean": float(np.mean(z)),
        "z_sd": float(np.std(z)),
    }


def inner_factor(model, s) -> float:
    """Widening factor learnt on the training part alone.

    Parameters
    ----------
    model : autostat.GPModel
        The kept model.
    s : run_2026.Series
        One benchmark series.

    Returns
    -------
    float
        The 90 % quantile of ``|z|`` on the last 10 % of the training part,
        forecast from the first 90 %, over 1.645; at least 1.
    """
    cut = round(0.9 * len(s.y_train))
    x_in, y_in = s.x_train[:cut], s.y_train[:cut]
    x_out, y_out = s.x_train[cut:], s.y_train[cut:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit = fit(model, x_in, y_in)
    if not np.isfinite(refit.nll):
        refit = model
    mean, var = posterior(refit, x_in, y_in, x_out, noise=True)
    z = np.abs(y_out - mean) / np.sqrt(var)
    return float(max(1.0, np.quantile(z, 0.9) / Z90))


def tempered(models, s, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Mixture forecast with the BIC weights raised to ``1 / temperature``.

    Parameters
    ----------
    models : list of autostat.GPModel
        Every model the search scored.
    s : run_2026.Series
        One benchmark series.
    temperature : float
        1 gives the benchmark's mixture.

    Returns
    -------
    mean, sd : numpy.ndarray
        Mixture mean and standard deviation (law of total variance).
    """
    chosen, _ = bic_weights(models)
    bics = np.array([m.bic for m in chosen])
    w = np.exp(-(bics - bics[0]) / (2 * temperature))[:MAX_STRUCTURES]
    w /= w.sum()
    first = second = 0.0
    for model, weight in zip(chosen[:MAX_STRUCTURES], w, strict=True):
        if weight < 1e-6:
            continue
        m, v = posterior(model, s.x_train, s.y_train, s.x_test, noise=True)
        first = first + weight * m
        second = second + weight * (v + m**2)
    mean = first / w[w >= 1e-6].sum()
    var = second / w[w >= 1e-6].sum() - mean**2
    return mean, np.sqrt(np.maximum(var, 1e-12))


def main() -> None:
    """Diagnose and recalibrate every series; write the table."""
    rows = []
    for s in run_2026.load_series():
        result = SearchResult.load(
            ROOT / "benchmark/results/searches/abcd_2014-depth5" / f"{s.name}.json"
        )
        models = [m for level in result.levels for m in level]
        mean, var = posterior(result.best, s.x_train, s.y_train, s.x_test, noise=True)
        sd = np.sqrt(var)
        factor = inner_factor(result.best, s)
        _, weights = bic_weights(models)
        row = {
            "dataset": s.name,
            "top_weight": float(weights[0]),
            "factor": factor,
            "best": score(mean, sd, s),
            "recalibrated": score(mean, factor * sd, s),
        }
        for t in TEMPERATURES:
            row[f"mixture_T{t}"] = score(*tempered(models, s, t), s)
        rows.append(row)
        b, r = row["best"], row["recalibrated"]
        print(
            f"{s.name[:24]:24s} z mean {b['z_mean']:6.2f} sd {b['z_sd']:5.2f} | "
            f"cover {b['coverage']:.2f} -> x{factor:.2f} {r['coverage']:.2f} | "
            f"IS {b['interval_score']:7.2f} -> {r['interval_score']:7.2f} | top w "
            f"{row['top_weight']:.2f} | T10 cover {row['mixture_T10']['coverage']:.2f}",
            flush=True,
        )
    (HERE / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
