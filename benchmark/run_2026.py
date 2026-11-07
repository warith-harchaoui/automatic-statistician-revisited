#!/usr/bin/env python3
"""Run 2026 forecasting methods on the 2014 extrapolation benchmark.

    python benchmark/run_2026.py                       # every method
    python benchmark/run_2026.py --methods auto_ets chronos2
    python benchmark/run_2026.py --methods abcd_2014 --n-jobs 6

What this script does
    The AAAI 2014 paper scored its methods on thirteen series split into a
    training part (the first 90 %) and a held-out tail (the last 10 %). This
    script trains each method on the training part, forecasts the tail and
    writes one CSV of predictions per method, with a 90 % interval when the
    method gives one.

The families
    * ``classical``: naive, seasonal naive, automatic exponential smoothing
      (ETS), automatic ARIMA (autoregressive integrated moving average) and
      automatic Theta, from ``statsforecast``.
    * ``machine learning``: gradient-boosted trees (LightGBM) on lags,
      forecasting recursively.
    * ``deep learning``: N-HiTS, N-BEATS and PatchTST from ``neuralforecast``,
      each trained on the one series.
    * ``foundation``: Chronos-2, Chronos-Bolt (base), TimesFM 2.5 and
      TabPFN-TS used zero-shot; NVIDIA's KumoTabular in context on lags.
    * ``structure search``: this repository's ports of the 2014 and 2013
      Automatic Statistician.

    Three 2026 structure-search systems do not run here, for reasons recorded
    in ``benchmark/README.md``: AutoGP.jl needs a Julia installation, GPy-ABCD
    needs GPy (which does not build on Python 3.13) and AutoBNN no longer
    imports with current JAX.

What it produces
    ``benchmark/results/<method>.csv`` with columns ``dataset, step,
    prediction, lower, upper``; also ``benchmark/results/<method>.json`` with
    the family, the time taken and any failure.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("benchmark")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

#: Seasonal period, in samples, of each series. Monthly series repeat every
#: 12 samples; the temperature series is sampled every 3 days (122 per year)
#: and the births series every 4 days (91 per year); internet traffic is
#: hourly (24 per day); the annual series have no season.
SEASONS = {
    "01-airline": 12,
    "02-solar": 1,
    "03-mauna2003": 12,
    "beveridge-wheat-price-index-1500": 1,
    "daily-minimum-temperatures-in-me": 122,
    "internet-traffic-data-in-bits-fr-2": 24,
    "monthly-average-daily-calls-to-d": 12,
    "monthly-critical-radio-frequenci": 12,
    "monthly-production-of-gas-in-aus": 12,
    "monthly-production-of-sulphuric-": 12,
    "monthly-us-male-1619-years-unemp": 12,
    "number-of-daily-births-in-quebec": 91,
    "real-daily-wages-in-pounds-engla": 1,
}


@dataclass
class Series:
    """One benchmark series, split as in 2014.

    Parameters
    ----------
    name : str
        Dataset name.
    x_train, y_train : numpy.ndarray
        Training inputs (decimal years) and values.
    x_test, y_test : numpy.ndarray
        Held-out inputs and values.
    season : int
        Seasonal period in samples.
    """

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    season: int

    @property
    def horizon(self) -> int:
        """Number of held-out points.

        Returns
        -------
        int
            Length of the test part.
        """
        return len(self.y_test)


Forecast = tuple[np.ndarray, np.ndarray | None, np.ndarray | None]


def load_series() -> list[Series]:
    """Read the thirteen train and test CSV files.

    Returns
    -------
    list of Series
        In name order.
    """
    out = []
    for name, season in sorted(SEASONS.items()):
        train = np.loadtxt(DATA / f"{name}-train.csv", delimiter=",", ndmin=2)
        test = np.loadtxt(DATA / f"{name}-test.csv", delimiter=",", ndmin=2)
        out.append(
            Series(name, train[:, 0], train[:, 1], test[:, 0], test[:, 1], season)
        )
    return out


# Classical


def _statsforecast(
    model_factory: Callable[[int], object],
) -> Callable[[Series], Forecast]:
    """Wrap a statsforecast model as a benchmark method.

    Parameters
    ----------
    model_factory : callable
        Builds the model from the seasonal period.

    Returns
    -------
    callable
        Method returning mean and 90 % interval.
    """

    def method(s: Series) -> Forecast:
        """Fit on the training part and forecast the held-out tail.

        Parameters
        ----------
        s : Series
            One benchmark series.

        Returns
        -------
        tuple
            Mean, lower and upper 90 % bounds.
        """
        model = model_factory(s.season)
        model.fit(s.y_train.astype(float))
        out = model.predict(h=s.horizon, level=[90])
        return out["mean"], out.get("lo-90"), out.get("hi-90")

    return method


def _classical() -> dict[str, Callable[[Series], Forecast]]:
    """Build the classical methods.

    Returns
    -------
    dict
        Name to method.
    """
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive

    return {
        "naive": _statsforecast(lambda m: Naive()),
        "seasonal_naive": _statsforecast(
            lambda m: SeasonalNaive(season_length=max(m, 1))
        ),
        "auto_ets": _statsforecast(
            lambda m: AutoETS(season_length=m if m <= 24 else 1)
        ),
        "auto_arima": _statsforecast(
            lambda m: AutoARIMA(season_length=m if m <= 24 else 1)
        ),
        "auto_theta": _statsforecast(
            lambda m: AutoTheta(season_length=m if m <= 24 else 1)
        ),
    }


# Machine learning


def _lag_matrix(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a design of the previous ``lags`` values for each target.

    Parameters
    ----------
    y : numpy.ndarray
        Series.
    lags : int
        Number of lags.

    Returns
    -------
    tuple of numpy.ndarray
        Design ``(n - lags, lags)``, most recent lag first; then the targets.
    """
    rows = [y[i - lags : i][::-1] for i in range(lags, len(y))]
    return np.asarray(rows), y[lags:]


def _recursive(
    predict_one: Callable[[np.ndarray], float], y: np.ndarray, lags: int, h: int
) -> np.ndarray:
    """Forecast ``h`` steps by feeding each prediction back as a lag.

    Parameters
    ----------
    predict_one : callable
        Maps a lag vector to the next value.
    y : numpy.ndarray
        History.
    lags : int
        Lag count.
    h : int
        Steps.

    Returns
    -------
    numpy.ndarray
        Forecasts.
    """
    history = list(y)
    out = []
    for _ in range(h):
        nxt = float(predict_one(np.asarray(history[-lags:][::-1])))
        out.append(nxt)
        history.append(nxt)
    return np.asarray(out)


def _lags_for(s: Series) -> int:
    """Choose the lag count: two seasons, bounded to keep enough rows.

    Parameters
    ----------
    s : Series
        One benchmark series.

    Returns
    -------
    int
        Between 4 and a third of the training length.
    """
    return int(np.clip(2 * max(s.season, 6), 4, len(s.y_train) // 3))


def lightgbm_lags(s: Series) -> Forecast:
    """Gradient-boosted trees on lags, forecasting recursively.

    Trees do not extrapolate a trend, so the series is differenced first and
    the forecasts are integrated back.

    Parameters
    ----------
    s : Series
        One benchmark series.

    Returns
    -------
    tuple
        Mean only.
    """
    import lightgbm as lgb

    diff = np.diff(s.y_train)
    lags = _lags_for(s)
    x, y = _lag_matrix(diff, lags)
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0
    )
    model.fit(x, y)
    steps = _recursive(lambda v: model.predict(v[None, :])[0], diff, lags, s.horizon)
    return s.y_train[-1] + np.cumsum(steps), None, None


# Deep learning


def _neuralforecast(model_name: str) -> Callable[[Series], Forecast]:
    """Wrap a neuralforecast model trained on the one series.

    Parameters
    ----------
    model_name : str
        ``"NHITS"``, ``"NBEATS"`` or ``"PatchTST"``.

    Returns
    -------
    callable
        Method returning the mean forecast.
    """

    def method(s: Series) -> Forecast:
        """Train on the training part and forecast the tail.

        Parameters
        ----------
        s : Series
            One benchmark series.

        Returns
        -------
        tuple
            Mean only.
        """
        import neuralforecast.models as models
        from neuralforecast import NeuralForecast

        h = s.horizon
        input_size = int(min(max(2 * h, 2 * s.season, 24), len(s.y_train) // 2))
        cls = getattr(models, model_name)
        model = cls(
            h=h,
            input_size=input_size,
            max_steps=500,
            random_seed=0,
            scaler_type="standard",
            enable_progress_bar=False,
            logger=False,
        )
        frame = pd.DataFrame(
            {"unique_id": "s", "ds": np.arange(len(s.y_train)), "y": s.y_train}
        )
        nf = NeuralForecast(models=[model], freq=1)
        nf.fit(frame)
        out = nf.predict()
        return out[model_name].to_numpy(), None, None

    return method


# Foundation models


_CACHE: dict[str, object] = {}


def _chronos(checkpoint: str) -> Callable[[Series], Forecast]:
    """Wrap a Chronos checkpoint, zero-shot.

    Parameters
    ----------
    checkpoint : str
        Hugging Face model id.

    Returns
    -------
    callable
        Method returning the median and the 5 and 95 % quantiles.
    """

    def method(s: Series) -> Forecast:
        """Forecast the tail from the training part.

        Parameters
        ----------
        s : Series
            One benchmark series.

        Returns
        -------
        tuple
            Median, 5 % and 95 % quantiles.
        """
        import torch
        from chronos import BaseChronosPipeline

        if checkpoint not in _CACHE:
            _CACHE[checkpoint] = BaseChronosPipeline.from_pretrained(
                checkpoint, device_map="cpu", torch_dtype=torch.float32
            )
        pipeline = _CACHE[checkpoint]
        quantiles, _ = pipeline.predict_quantiles(
            inputs=[torch.tensor(s.y_train, dtype=torch.float32)],
            prediction_length=s.horizon,
            quantile_levels=[0.05, 0.5, 0.95],
        )
        q = np.asarray(quantiles[0] if isinstance(quantiles, list) else quantiles)
        q = q.reshape(-1, s.horizon, 3)[0]
        return q[:, 1], q[:, 0], q[:, 2]

    return method


def timesfm_2p5(s: Series) -> Forecast:
    """TimesFM 2.5 (200 million parameters), zero-shot.

    Parameters
    ----------
    s : Series
        One benchmark series.

    Returns
    -------
    tuple
        Point forecast, 10 % and 90 % quantiles (the model's outer deciles).
    """
    import timesfm

    if "timesfm" not in _CACHE:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
        )
        _CACHE["timesfm"] = model
    point, quantiles = _CACHE["timesfm"].forecast(
        horizon=s.horizon, inputs=[s.y_train.astype(np.float32)]
    )
    q = np.asarray(quantiles)[0]
    return np.asarray(point)[0], q[:, 1], q[:, -1]


def tabpfn_ts(s: Series) -> Forecast:
    """TabPFN-TS: TabPFN on calendar and index features, zero-shot.

    Parameters
    ----------
    s : Series
        One benchmark series.

    Returns
    -------
    tuple
        Median, 5 % and 95 % quantiles.
    """
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    start = pd.Timestamp("2000-01-01")
    train = pd.DataFrame(
        {
            "item_id": "s",
            "timestamp": start + pd.to_timedelta(np.arange(len(s.y_train)), unit="D"),
            "target": s.y_train,
        }
    )
    pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
    out = pipeline.predict_df(
        context_df=train, prediction_length=s.horizon, quantiles=[0.05, 0.5, 0.95]
    )
    return out["0.5"].to_numpy(), out["0.05"].to_numpy(), out["0.95"].to_numpy()


def kumo_lags(s: Series) -> Forecast:
    """NVIDIA KumoTabular in context on lagged differences, recursively.

    Parameters
    ----------
    s : Series
        One benchmark series.

    Returns
    -------
    tuple
        Median only (recursive quantiles do not compose).
    """
    import sdm
    import torch

    diff = np.diff(s.y_train)
    lags = _lags_for(s)
    x, y = _lag_matrix(diff, lags)
    names = [f"lag{i}" for i in range(lags)]

    def table(values: np.ndarray, columns: list[str]):
        """Wrap an array as an ``sdm.TableTensor``.

        Parameters
        ----------
        values : numpy.ndarray
            Rows.
        columns : list of str
            Names.

        Returns
        -------
        sdm.TableTensor
            The table.
        """
        frame = pd.DataFrame(values, columns=columns)
        return sdm.TableTensor.from_pandas(
            df=frame, stypes=sdm.infer_stypes(frame), device="cpu"
        )

    model = sdm.models.KumoTabular(task=sdm.Task.regression, device="cpu", size="small")
    model.fit(
        x=table(x[-2048:], names),
        y=table(y[-2048:, None], ["target"])[:, "target"],
        generator=torch.Generator().manual_seed(0),
    )

    def one(v: np.ndarray) -> float:
        """Predict the next difference.

        Parameters
        ----------
        v : numpy.ndarray
            Lag vector.

        Returns
        -------
        float
            Median.
        """
        return float(
            model.predict(table(v[None, :], names)).to_pandas()["q500"].iloc[0]
        )

    steps = _recursive(one, diff, lags, s.horizon)
    return s.y_train[-1] + np.cumsum(steps), None, None


# Structure search (this repository)


def _autostat(preset: str, n_jobs: int, max_depth: int) -> Callable[[Series], Forecast]:
    """Wrap this repository's structure search.

    Parameters
    ----------
    preset : str
        ``"abcd_2014"`` or ``"cks_2013"``.
    n_jobs : int
        Worker processes for the search.
    max_depth : int
        Search levels.

    Returns
    -------
    callable
        Method returning the posterior mean and a 90 % interval.
    """

    def method(s: Series) -> Forecast:
        """Search on the training part and extrapolate the posterior.

        Parameters
        ----------
        s : Series
            One benchmark series.

        Returns
        -------
        tuple
            Posterior mean and 90 % predictive interval.
        """
        from autostat import Experiment, posterior, search

        exp = getattr(Experiment, preset)(max_depth=max_depth, n_jobs=n_jobs)
        result = search(s.x_train, s.y_train, exp)
        mean, var = posterior(result.best, s.x_train, s.y_train, s.x_test, noise=True)
        # A white-noise kernel is part of the covariance, not of the
        # likelihood, so its variance comes from the kernel's own diagonal.
        sd = np.sqrt(var)
        return mean, mean - 1.645 * sd, mean + 1.645 * sd

    return method


@dataclass
class Method:
    """A registered method and its family.

    Parameters
    ----------
    family : str
        One of the families listed in the module docstring.
    run : callable
        Maps a series to a forecast.
    """

    family: str
    run: Callable[[Series], Forecast]


def registry(n_jobs: int, max_depth: int) -> dict[str, Method]:
    """Build every method lazily, importing each library only when used.

    Parameters
    ----------
    n_jobs : int
        Workers for the structure search.
    max_depth : int
        Levels for the structure search.

    Returns
    -------
    dict
        Name to method.
    """
    methods: dict[str, Method] = {}
    for name in ("naive", "seasonal_naive", "auto_ets", "auto_arima", "auto_theta"):
        methods[name] = Method("classical", lambda s, n=name: _classical()[n](s))
    methods["lightgbm_lags"] = Method("machine learning", lightgbm_lags)
    for name in ("NHITS", "NBEATS", "PatchTST"):
        methods[name.lower()] = Method("deep learning", _neuralforecast(name))
    methods["chronos2"] = Method("foundation", _chronos("amazon/chronos-2"))
    methods["chronos_bolt_base"] = Method(
        "foundation", _chronos("amazon/chronos-bolt-base")
    )
    methods["timesfm_2p5"] = Method("foundation", timesfm_2p5)
    methods["tabpfn_ts"] = Method("foundation", tabpfn_ts)
    methods["kumo_lags"] = Method("foundation", kumo_lags)
    methods["abcd_2014_port"] = Method(
        "structure search", _autostat("abcd_2014", n_jobs, max_depth)
    )
    methods["cks_2013_port"] = Method(
        "structure search", _autostat("cks_2013", n_jobs, max_depth)
    )
    return methods


def run_method(name: str, method: Method, series: list[Series]) -> None:
    """Run one method on every series and write its predictions.

    Parameters
    ----------
    name : str
        Method name.
    method : Method
        The method.
    series : list of Series
        The benchmark.
    """
    rows, failures = [], {}
    started = time.perf_counter()
    for s in series:
        tic = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, lower, upper = method.run(s)
            mean = np.asarray(mean, float).ravel()[: s.horizon]
            lo = (
                np.full(s.horizon, np.nan)
                if lower is None
                else np.asarray(lower, float).ravel()[: s.horizon]
            )
            hi = (
                np.full(s.horizon, np.nan)
                if upper is None
                else np.asarray(upper, float).ravel()[: s.horizon]
            )
            for step in range(s.horizon):
                rows.append(
                    {
                        "dataset": s.name,
                        "step": step,
                        "prediction": mean[step],
                        "lower": lo[step],
                        "upper": hi[step],
                    }
                )
            LOGGER.info("%s %s: %.1f s", name, s.name, time.perf_counter() - tic)
        except Exception as error:
            failures[s.name] = f"{type(error).__name__}: {error}"[:300]
            LOGGER.warning("%s %s failed: %s", name, s.name, failures[s.name])
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"{name}.csv", index=False)
    meta = {
        "family": method.family,
        "seconds": time.perf_counter() - started,
        "failures": failures,
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(meta, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested methods.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 on completion.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="*")
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="structure-search workers"
    )
    parser.add_argument(
        "--max-depth", type=int, default=10, help="structure-search levels"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    series = load_series()
    methods = registry(args.n_jobs, args.max_depth)
    for name in args.methods or list(methods):
        run_method(name, methods[name], series)
    return 0


if __name__ == "__main__":
    sys.exit(main())
