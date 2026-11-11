"""Greedy search over kernel structures, scored by an information criterion.

What this module does
    It runs the Automatic Statistician's structure search: expand the best
    structures by every grammar rule, optimise each candidate from several
    random starts, score it, keep the best and repeat.

Why it exists as a separate module
    The search loop is the part of the system that the 2013 and 2014 versions
    share. The two differ in their settings (grammar, base kernels, restarts,
    noise model), which :class:`Experiment` names as two presets.

What it consumes
    Inputs ``x`` and targets ``y`` as arrays, plus an :class:`Experiment`.

What it produces
    A :class:`SearchResult`: the best model, every level's ranked models,
    the settings and the time taken.

The loop, step for step as the original
    1. expand the current best structure(s) by every grammar rule;
    2. rewrite into additive form and drop duplicates;
    3. give every unset parameter ``n_rand`` random starts, simplify,
       deduplicate and jitter;
    4. add the incumbent and jittered copies of it, in case it only needed
       more optimisation;
    5. optimise and score everything, discarding out-of-bounds and failed
       fits;
    6. keep the best ``k`` and repeat, up to ``max_depth`` levels.

    The original shipped each batch to a Sun Grid Engine cluster or to a local
    MATLAB through ``cblparallel``, polling for ``.mat`` files. A ``joblib``
    process pool calls the optimiser directly here.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>. The ported design is (c) 2013-2015
David Duvenaud, James Robert Lloyd and Roger Grosse, MIT licence, reproduced
in ``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from . import grammar
from . import kernels as ff
from .model import GPModel, fit

LOGGER = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Search settings. The defaults reproduce the published 2014 runs.

    Parameters
    ----------
    max_depth : int
        Levels of expansion.
    k : int
        Structures kept per level; ``1`` is greedy.
    n_rand : int
        Random starts per new structure.
    sd : float
        Spread of the random starts, on the log scale.
    jitter_sd : float
        Noise added to every parameter before optimising.
    iters : int
        Optimiser iterations per fit (on the subset when ``subset`` is on).
    base_kernels : str
        Base kernels the grammar introduces.
    additive_form : bool
        Rewrite every candidate as a sum of products.
    score : str
        ``"bic"``, ``"aic"``, ``"pl2"`` or ``"nll"``.
    random_seed : int
        Seeds every random choice of the search.
    period_heuristic : float
        Minimum number of data points per period, roughly.
    max_period_heuristic : float
        Minimum number of periods the data covers for a period to count.
    period_heuristic_type : str
        ``"min"``, ``"average"``, ``"both"`` or ``"none"``.
    subset : bool
        Optimise on a random subset first.
    subset_size : int
        Size of that subset.
    full_iters : int
        Iterations on all the data after the subset stage.
    stopping_criteria : list of str
        ``["no_improvement"]`` stops once the best score has not fallen for
        ``patience`` levels in a row. Empty (the published configurations)
        searches all ``max_depth`` levels.
    improvement_tolerance : float
        The fall that counts as an improvement.
    patience : int
        Levels searched past the last improvement before stopping, for the
        report to show what the search would have found by going on. The
        model kept is the best of all levels either way. ``1`` stops at the
        first level that does not improve.
    n_jobs : int
        Worker processes; ``1`` runs in the calling process.
    rules : list, optional
        Grammar rules; the 2014 set when omitted.
    start : {"noise", "empty"}
        ``"noise"`` expands white noise first (2014); ``"empty"`` starts
        from the base kernels themselves (2013).
    likelihood_noise : bool
        Model observation noise in the likelihood (2013) instead of leaving
        the search to discover a white-noise kernel (2014).
    period_hints : list of float
        Periods the business expects, in the time unit of the inputs (see
        :meth:`autostat.context.Context.search_hints`). Half the random
        starts of a periodic kernel then begin near one of them. They only
        say where to start: BIC still decides what is kept. Empty (the
        published configurations) changes nothing.
    location_hints : list of float
        Dates of known events; half the random starts of a changepoint or a
        change window then switch near them. Empty changes nothing.
    """

    max_depth: int = 10
    k: int = 1
    n_rand: int = 9
    sd: float = 2.0
    jitter_sd: float = 0.1
    iters: int = 250
    base_kernels: str = "SE,Per,Lin,Const,Noise"
    additive_form: bool = True
    score: str = "bic"
    random_seed: int = 1
    period_heuristic: float = 3.0
    max_period_heuristic: float = 5.0
    period_heuristic_type: str = "min"
    subset: bool = True
    subset_size: int = 250
    full_iters: int = 10
    stopping_criteria: list[str] = field(default_factory=list)
    improvement_tolerance: float = 0.1
    patience: int = 1
    n_jobs: int = -1
    rules: list[grammar.Rule] | None = None
    start: str = "noise"
    likelihood_noise: bool = False
    period_hints: list[float] = field(default_factory=list)
    location_hints: list[float] = field(default_factory=list)

    @classmethod
    def abcd_2014(cls, **overrides: Any) -> Experiment:
        """Return the Automatic Bayesian Covariance Discovery settings (AAAI 2014).

        These are the settings of ``gpss-research``'s ``2014-01-14-GPSS-add``,
        the run that produced the published reports. They equal the defaults.

        Parameters
        ----------
        **overrides
            Any field to change, e.g. ``n_jobs=4``.

        Returns
        -------
        Experiment
            The preset.
        """
        return cls(**overrides)

    @classmethod
    def cks_2013(cls, **overrides: Any) -> Experiment:
        """Return the compositional kernel search settings (ICML 2013).

        Three rules (add, multiply, replace a base kernel), the rational
        quadratic among the base kernels, Gaussian likelihood noise instead
        of a white-noise kernel, no changepoints, no additive form, two random
        starts with spread 4 and 100 iterations on all the data. They are
        transcribed from ``gp-structure-search``'s ``Experiment`` defaults.
        The parameter initialisation stays the 2014 one, so this preset
        approximates the 2013 system rather than reproducing it. Unlike the
        2014 preset it is not checked against published numbers: the 2013
        result files use kernel classes this package does not read.

        Parameters
        ----------
        **overrides
            Any field to change.

        Returns
        -------
        Experiment
            The preset.
        """
        settings: dict[str, Any] = {
            "n_rand": 2,
            "sd": 4.0,
            "jitter_sd": 0.0,
            "iters": 100,
            "base_kernels": "SE,RQ,Per,Lin,Const",
            "additive_form": False,
            "random_seed": 0,
            "period_heuristic": 10.0,
            "max_period_heuristic": 1.0,
            "subset": False,
            "full_iters": 0,
            "rules": grammar.RULES_2013,
            "start": "empty",
            "likelihood_noise": True,
        }
        settings.update(overrides)
        return cls(**settings)


@dataclass
class SearchResult:
    """Everything a search produced.

    Parameters
    ----------
    best : GPModel
        The best model found at any level.
    levels : list of list of GPModel
        Scored models per level, best first.
    experiment : Experiment
        The settings used.
    seconds : float
        Wall-clock time.
    """

    best: GPModel
    levels: list[list[GPModel]]
    experiment: Experiment
    seconds: float

    def best_per_level(self) -> list[GPModel]:
        """List the winner of each level.

        Returns
        -------
        list of GPModel
            One per completed level.
        """
        return [level[0] for level in self.levels if level]

    def to_dict(self) -> dict[str, Any]:
        """Summarise the search as JSON-serialisable data.

        Returns
        -------
        dict
            Best model, the top five of each level, settings and time.
        """
        return {
            "best": self.best.to_dict(),
            "levels": [[m.to_dict() for m in level[:5]] for level in self.levels],
            "experiment": asdict(self.experiment),
            "seconds": self.seconds,
        }

    @classmethod
    def load(cls, path: str | Path) -> SearchResult:
        """Read back what :meth:`save` wrote, to re-describe a search without rerunning it.

        Only the top five models of each level were saved, which is all a
        report reads.

        Parameters
        ----------
        path : str or pathlib.Path
            A file :meth:`save` wrote.

        Returns
        -------
        SearchResult
            The search.
        """
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(Experiment)}
        settings = {k: v for k, v in data["experiment"].items() if k in known}
        if settings.get("rules") is not None:
            settings["rules"] = [_as_tuples(rule) for rule in settings["rules"]]
        return cls(
            best=GPModel.from_dict(data["best"]),
            levels=[[GPModel.from_dict(m) for m in level] for level in data["levels"]],
            experiment=Experiment(**settings),
            seconds=float(data["seconds"]),
        )

    def save(self, path: str | Path) -> None:
        """Write :meth:`to_dict` as indented JSON.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=float))


def _min_abs_diff(values: np.ndarray) -> float:
    """Return the smallest gap between distinct sorted values.

    Parameters
    ----------
    values : numpy.ndarray
        One input dimension.

    Returns
    -------
    float
        The gap, or ``1.0`` when there is a single distinct value.
    """
    diffs = np.diff(np.unique(values))
    return float(diffs.min()) if diffs.size else 1.0


def data_shape(x: np.ndarray, y: np.ndarray, exp: Experiment) -> dict[str, Any]:
    """Summarise location and scale to seed initialisation and bounds.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs ``(n, D)``.
    y : numpy.ndarray
        Targets ``(n,)``.
    exp : Experiment
        Supplies the period heuristics.

    Returns
    -------
    dict
        The original's ``data_shape`` dictionary, key for key.

    Raises
    ------
    ValueError
        On an unknown ``period_heuristic_type``.
    """
    dims = range(x.shape[1])
    shape: dict[str, Any] = {
        "x_mean": [float(np.mean(x[:, d])) for d in dims],
        "y_mean": float(np.mean(y)),
        "x_sd": [float(np.log(np.std(x[:, d]))) for d in dims],
        "y_sd": float(np.log(np.std(y))),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "x_min": [float(np.min(x[:, d])) for d in dims],
        "x_max": [float(np.max(x[:, d])) for d in dims],
        "x_min_abs_diff": [math.log(_min_abs_diff(x[:, d])) for d in dims],
    }
    # The shortest admissible period: a few samples per cycle, measured either
    # on the smallest spacing or on the average spacing.
    h = exp.period_heuristic
    mins = [h * _min_abs_diff(x[:, d]) for d in dims]
    averages = [h * float(np.ptp(x[:, d])) / x.shape[0] for d in dims]
    kind = exp.period_heuristic_type
    if kind == "none":
        shape["min_period"] = None
    elif kind == "min":
        shape["min_period"] = [math.log(v) for v in mins]
    elif kind == "average":
        shape["min_period"] = [math.log(v) for v in averages]
    elif kind == "both":
        pairs = zip(mins, averages, strict=True)
        shape["min_period"] = [math.log(max(a, b)) for a, b in pairs]
    else:
        raise ValueError(f"unknown period_heuristic_type {kind!r}")
    # The longest admissible period: the data spans at least a few cycles.
    shape["max_period"] = [
        math.log((shape["x_max"][d] - shape["x_min"][d]) / exp.max_period_heuristic)
        for d in dims
    ]
    # What the business expects: read by the kernels' random starts only.
    shape["period_hints"] = [math.log(p) for p in exp.period_hints if p > 0]
    shape["location_hints"] = [float(t) for t in exp.location_hints]
    return shape


def _restarts(
    models: list[GPModel],
    exp: Experiment,
    shape: dict[str, Any],
    rng: np.random.Generator,
) -> list[GPModel]:
    """Give each model ``n_rand`` copies with fresh draws for unset parameters.

    Parameters
    ----------
    models : list of GPModel
        Candidates; fitted parameters stay as they are.
    exp : Experiment
        Supplies ``n_rand`` and ``sd``.
    shape : dict
        Data summary.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    list of GPModel
        ``n_rand`` copies per input model.
    """
    out = []
    for model in models:
        for _ in range(exp.n_rand):
            m = model.copy()
            m.kernel.initialise_params(rng, exp.sd, shape)
            out.append(m)
    return out


def _jitter(
    models: list[GPModel], sd: float, rng: np.random.Generator
) -> list[GPModel]:
    """Perturb every kernel parameter, which the authors found helps L-BFGS escape.

    Parameters
    ----------
    models : list of GPModel
        Modified in place.
    sd : float
        Standard deviation of the perturbation.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    list of GPModel
        The same list.
    """
    if sd <= 0:
        return models
    for m in models:
        vector = m.kernel.param_vector
        m.kernel.load_param_vector(vector + rng.normal(0.0, sd, vector.size))
    return models


def _expand(models: list[GPModel], ndim: int, exp: Experiment) -> list[GPModel]:
    """Expand each model's kernel by the grammar.

    Parameters
    ----------
    models : list of GPModel
        Models to expand.
    ndim : int
        Input dimensions.
    exp : Experiment
        Supplies the grammar settings.

    Returns
    -------
    list of GPModel
        Unscored neighbours, in additive form when requested.
    """
    out = []
    for model in models:
        neighbours = grammar.expand_kernels(
            ndim, [model.kernel], exp.base_kernels, exp.rules
        )
        for k in neighbours:
            m = model.copy()
            m.kernel = k
            m.nll = math.nan
            out.append(m)
    if exp.additive_form:
        out = ff.remove_duplicates([m.additive_form() for m in out])
    return out


def _fit_one(
    model: GPModel, x: np.ndarray, y: np.ndarray, exp: Experiment, seed: int
) -> GPModel:
    """Fit one candidate under the experiment's optimiser settings.

    Parameters
    ----------
    model : GPModel
        Candidate with every parameter set.
    x, y : numpy.ndarray
        Data.
    exp : Experiment
        Optimiser settings.
    seed : int
        Seeds the subset draw.

    Returns
    -------
    GPModel
        Fitted copy.
    """
    return fit(
        model,
        x,
        y,
        iters=exp.iters,
        subset=exp.subset,
        subset_size=exp.subset_size,
        full_iters=exp.full_iters,
        seed=seed,
    )


def _evaluate(
    models: list[GPModel], x: np.ndarray, y: np.ndarray, exp: Experiment, seed: int
) -> list[GPModel]:
    """Fit every candidate, in parallel unless ``n_jobs`` is one.

    Parameters
    ----------
    models : list of GPModel
        Candidates.
    x, y : numpy.ndarray
        Data.
    exp : Experiment
        Settings.
    seed : int
        Seeds the subset draw.

    Returns
    -------
    list of GPModel
        Fitted models, in input order.
    """
    if exp.n_jobs == 1:
        return [_fit_one(m, x, y, exp, seed) for m in models]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=exp.n_jobs, backend="loky")(
        delayed(_fit_one)(m, x, y, exp, seed) for m in models
    )


def search(
    x: Any,
    y: Any,
    exp: Experiment | None = None,
    start: ff.Kernel | None = None,
) -> SearchResult:
    """Search for the kernel structure that best explains ``y`` given ``x``.

    Parameters
    ----------
    x : array_like
        Inputs, ``(n,)`` or ``(n, D)``. For a time series, time in the unit
        the report speaks (decimal years, by the original's convention).
    y : array_like
        Targets, ``(n,)``.
    exp : Experiment, optional
        Settings; the 2014 preset when omitted.
    start : Kernel, optional
        Structure to expand first; set by ``exp.start`` when omitted.

    Returns
    -------
    SearchResult
        Best model, per-level results, settings and time.

    Raises
    ------
    ValueError
        When ``x`` and ``y`` disagree in length or contain non-finite values.
    RuntimeError
        When no candidate of the first level could be fitted.
    """
    exp = exp or Experiment()
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    y = np.asarray(y, float).ravel()
    if x.shape[0] != y.size:
        raise ValueError(f"x has {x.shape[0]} rows but y has {y.size} values")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must be finite; drop or impute missing values")
    ndim = x.shape[1]
    rng = np.random.default_rng(exp.random_seed)
    shape = data_shape(x, y, exp)
    started = time.perf_counter()

    if start is None:
        start = ff.NoneKernel() if exp.start == "empty" else ff.NoiseKernel()
    lik_sf = shape["y_sd"] - math.log(10) if exp.likelihood_noise else -math.inf
    current = _expand([GPModel(start, lik_sf=lik_sf, ndata=y.size)], ndim, exp)
    best_models: list[GPModel] | None = None
    levels: list[list[GPModel]] = []
    best_score = math.inf
    stalled_levels = 0

    for depth in range(exp.max_depth):
        current = _restarts(current, exp, shape, rng)
        current = ff.remove_duplicates([m.simplified() for m in current])
        current = _jitter(current, exp.jitter_sd, rng)
        if best_models is not None:
            for incumbent in best_models:
                copies = [incumbent.copy() for _ in range(exp.n_rand)]
                jittered = _jitter(copies, exp.jitter_sd, rng)
                current = [*current, incumbent.copy(), *jittered]
        # Shuffling spreads expensive kernels over the worker pool.
        rng.shuffle(current)

        tic = time.perf_counter()
        scored = _evaluate(current, x, y, exp, seed=exp.random_seed + depth)
        in_bounds = [m for m in scored if not m.out_of_bounds(shape)]
        valid = [m for m in in_bounds if math.isfinite(m.score(exp.score))]
        if not valid:
            LOGGER.warning("level %d: every candidate failed; stopping", depth)
            break
        valid.sort(key=lambda m: m.score(exp.score))
        levels.append(valid)
        LOGGER.info(
            "level %d: %d fitted, %d out of bounds, %d failed, %.0fs; best %s=%.1f  %s",
            depth,
            len(scored),
            len(scored) - len(in_bounds),
            len(in_bounds) - len(valid),
            time.perf_counter() - tic,
            exp.score.upper(),
            valid[0].score(exp.score),
            valid[0].kernel.structure(),
        )

        best_models = valid[: exp.k]
        new_best = valid[0].score(exp.score)
        stalled_levels = (
            stalled_levels + 1
            if new_best >= best_score - exp.improvement_tolerance
            else 0
        )
        if "no_improvement" in exp.stopping_criteria and stalled_levels >= exp.patience:
            LOGGER.info("no improvement for %d level(s); stopping", stalled_levels)
            break
        best_score = min(best_score, new_best)
        current = _expand(best_models, ndim, exp)

    if not levels:
        raise RuntimeError("no candidate structure could be fitted")
    winners = (level[0] for level in levels)
    overall = min(winners, key=lambda m: m.score(exp.score))
    return SearchResult(overall, levels, exp, time.perf_counter() - started)


def _as_tuples(value: Any) -> Any:
    """Turn JSON lists back into the tuples grammar rules are made of.

    Parameters
    ----------
    value : Any
        A rule, or part of one, as JSON read it.

    Returns
    -------
    Any
        The same with every list a tuple; dictionaries are kept.
    """
    if isinstance(value, list):
        return tuple(_as_tuples(v) for v in value)
    return value
