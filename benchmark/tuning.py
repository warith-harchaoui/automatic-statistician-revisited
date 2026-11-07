"""Cross-validated hyper-parameters for the benchmark methods that have them.

What this module does
    LightGBM and the three neuralforecast models (N-HiTS, N-BEATS, PatchTST)
    have hyper-parameters that no internal criterion chooses. Here each one is
    chosen on the training part of each series only, by rolling-origin
    cross-validation of the actual task: from an origin, forecast the next
    ``h`` points, where ``h`` is the length of the series' held-out tail.

How
    Each method is a scikit-learn estimator whose ``fit`` receives the time
    index and the values of a training prefix. Its ``predict`` receives
    future time indices and forecasts them from the end of that prefix, the
    way the benchmark does. The estimator is the tail of a skrub DataOps
    graph whose knobs are ``skrub.choose_from`` choices. The grid search
    uses ``TimeSeriesSplit(n_splits=3, test_size=h)``: each fold trains on
    every point before its origin and scores the RMSE of the ``h`` points
    after it. No gap is needed, since the model forecasts the whole block
    from the origin and never sees a value inside it. The chosen learner is
    then scored by ``skore.evaluate`` with the same splitter. The fold
    RMSEs are kept with the forecast.

    The held-out tail never enters this: ``tune`` sees ``y_train`` only.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

#: Rolling origins per series.
FOLDS = 3


def lag_matrix(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
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


class _Forecaster(BaseEstimator, RegressorMixin):
    """Common part: ``X`` is a one-column frame of integer time indices."""

    def _remember(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Store the training values and the last training index.

        Parameters
        ----------
        X : pandas.DataFrame
            Column ``t``, consecutive integers.
        y : pandas.Series
            Values.

        Returns
        -------
        numpy.ndarray
            The values, in time order.
        """
        t = np.asarray(X["t"])
        if np.any(np.diff(t) != 1):
            raise ValueError("training indices must be consecutive")
        self.last_t_ = int(t[-1])
        self.first_t_ = int(t[0])
        self.values_ = np.asarray(y, dtype=float)
        return self.values_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast the requested future indices.

        A training index gets its observed value back: skrub previews each
        step on the data it was fitted on; the scores only ever ask for
        indices after the training end.

        Parameters
        ----------
        X : pandas.DataFrame
            Column ``t``.

        Returns
        -------
        numpy.ndarray
            One forecast per row.
        """
        t = np.asarray(X["t"])
        steps = t - self.last_t_
        out = np.empty(len(t))
        past = steps < 1
        out[past] = self.values_[t[past] - self.first_t_]
        if np.any(~past):
            out[~past] = self.forecast(int(steps.max()))[steps[~past] - 1]
        return out


class LightGBMForecaster(_Forecaster):
    """LightGBM on lagged differences, forecasting recursively.

    Trees do not extrapolate a trend, so the series is differenced first and
    the forecasts are integrated back.

    Parameters
    ----------
    lags : int
        Lagged differences used as features.
    n_estimators, learning_rate, num_leaves : int, float, int
        LightGBM's own.
    """

    def __init__(
        self,
        lags: int = 24,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
    ):
        self.lags = lags
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMForecaster:
        """Fit on the training prefix.

        Parameters
        ----------
        X : pandas.DataFrame
            Column ``t``.
        y : pandas.Series
            Values.

        Returns
        -------
        LightGBMForecaster
            Fitted.
        """
        import lightgbm as lgb

        values = self._remember(X, y)
        self.diff_ = np.diff(values)
        self.level_ = values[-1]
        self.lags_ = int(min(self.lags, len(self.diff_) // 3))
        design, target = lag_matrix(self.diff_, self.lags_)
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            verbose=-1,
            random_state=0,
            n_jobs=1,
        ).fit(design, target)
        return self

    def forecast(self, h: int) -> np.ndarray:
        """Forecast ``h`` steps after the training end.

        Parameters
        ----------
        h : int
            Steps.

        Returns
        -------
        numpy.ndarray
            Forecasts of the level.
        """
        history = list(self.diff_)
        steps = []
        for _ in range(h):
            lags = np.asarray(history[-self.lags_ :][::-1])[None, :]
            nxt = float(self.model_.predict(lags)[0])
            steps.append(nxt)
            history.append(nxt)
        return self.level_ + np.cumsum(steps)


class NeuralForecaster(_Forecaster):
    """A neuralforecast model trained on the one series, on the CPU.

    Parameters
    ----------
    model_name : str
        ``"NHITS"``, ``"NBEATS"`` or ``"PatchTST"``.
    h : int
        Horizon the network is built for.
    input_multiple : int
        Input window, in multiples of ``h``.
    max_steps : int
        Training steps.
    learning_rate : float
        Adam's step size.
    """

    def __init__(
        self,
        model_name: str = "NHITS",
        h: int = 12,
        input_multiple: int = 2,
        max_steps: int = 500,
        learning_rate: float = 1e-3,
    ):
        self.model_name = model_name
        self.h = h
        self.input_multiple = input_multiple
        self.max_steps = max_steps
        self.learning_rate = learning_rate

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NeuralForecaster:
        """Train on the training prefix.

        Parameters
        ----------
        X : pandas.DataFrame
            Column ``t``.
        y : pandas.Series
            Values.

        Returns
        -------
        NeuralForecaster
            Fitted.
        """
        import neuralforecast.models as models
        import torch
        from neuralforecast import NeuralForecast

        # One thread per fit: parallelism comes from the grid search workers.
        torch.set_num_threads(1)
        values = self._remember(X, y)
        # Leave room for at least as many training windows as the input is long.
        input_size = int(min(self.input_multiple * self.h, len(values) // 2))
        model = getattr(models, self.model_name)(
            h=self.h,
            input_size=input_size,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            random_seed=0,
            scaler_type="standard",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="cpu",
        )
        frame = pd.DataFrame(
            {"unique_id": "s", "ds": np.arange(len(values)), "y": values}
        )
        self.nf_ = NeuralForecast(models=[model], freq=1)
        self.nf_.fit(frame)
        return self

    def forecast(self, h: int) -> np.ndarray:
        """Forecast ``h`` steps after the training end.

        Parameters
        ----------
        h : int
            Steps; at most the horizon the network was built for.

        Returns
        -------
        numpy.ndarray
            Forecasts.
        """
        if h > self.h:
            raise ValueError(f"built for {self.h} steps, asked {h}")
        return self.nf_.predict()[self.model_name].to_numpy()[:h]


def lightgbm_grid(season: int, n_train: int) -> dict:
    """The LightGBM search space for one series.

    Parameters
    ----------
    season : int
        Seasonal period in samples.
    n_train : int
        Training length.

    Returns
    -------
    dict
        Parameter name to candidate values.
    """
    base = max(season, 6)
    lags = sorted({int(np.clip(k * base, 4, n_train // 4)) for k in (1, 2, 4)})
    return {
        "lags": lags,
        "n_estimators": [100, 300, 1000],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [7, 15, 31],
    }


def neural_grid() -> dict:
    """The neuralforecast search space, shared by the three models.

    Returns
    -------
    dict
        Parameter name to candidate values.
    """
    return {
        "input_multiple": [2, 4],
        "max_steps": [250, 1000],
        "learning_rate": [1e-3, 1e-4],
    }


def tune(
    estimator: _Forecaster,
    grid: dict,
    y_train: np.ndarray,
    horizon: int,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict]:
    """Choose the hyper-parameters by rolling-origin CV, refit, forecast.

    Parameters
    ----------
    estimator : _Forecaster
        Unfitted, with the fixed parameters set.
    grid : dict
        Parameter name to candidate values.
    y_train : numpy.ndarray
        The training part; nothing after it is used.
    horizon : int
        Steps to forecast, also the length of each validation block.
    n_jobs : int, optional
        Parallel fits in the search.

    Returns
    -------
    forecast : numpy.ndarray
        ``horizon`` values after the end of ``y_train``.
    info : dict
        Chosen parameters, their CV RMSE and the per-fold RMSE that
        ``skore.evaluate`` reports for the chosen learner.
    """
    import skore
    import skrub
    from sklearn.model_selection import TimeSeriesSplit

    frame = pd.DataFrame({"t": np.arange(len(y_train)), "y": y_train})
    splitter = TimeSeriesSplit(n_splits=FOLDS, test_size=horizon)
    choices = {
        name: skrub.choose_from(values, name=name) for name, values in grid.items()
    }
    series = skrub.var("series", frame)
    X = series[["t"]].skb.mark_as_X()
    y = series["y"].skb.mark_as_y()
    pred = X.skb.apply(estimator.set_params(**choices), y=y)
    search = pred.skb.make_grid_search(
        fitted=True,
        cv=splitter,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
    )
    results = search.results_.sort_values("mean_test_score", ascending=False)
    best = results.iloc[0]
    chosen = {name: _plain(best[name]) for name in grid}

    learner = X.skb.apply(estimator.set_params(**chosen), y=y).skb.make_learner()
    report = skore.evaluate(learner, data={"series": frame}, splitter=splitter)
    folds = [float(fold.metrics.rmse()) for fold in report.reports_]

    future = pd.DataFrame(
        {"t": np.arange(len(y_train), len(y_train) + horizon), "y": np.nan}
    )
    forecast = np.asarray(search.predict({"series": future}), dtype=float)
    info = {
        "chosen": chosen,
        "cv_rmse": float(-best["mean_test_score"]),
        "skore_fold_rmse": folds,
        "candidates": len(results),
    }
    return forecast, info


def _plain(value: object) -> object:
    """Turn a NumPy scalar into a JSON-friendly Python value.

    Parameters
    ----------
    value : object
        A grid value.

    Returns
    -------
    object
        ``int``, ``float`` or the value unchanged.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
