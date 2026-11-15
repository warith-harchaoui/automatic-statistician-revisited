"""The studio: a local page that loads a series, searches it and shows why.

What this module does
    It adds the routes of the studio to the local GUI host
    (:func:`autostat.gui.create_gui_app`). The page (``static/studio.html``)
    loads a CSV, a Parquet file or a bundled example, lets the reader pick
    the columns and state what the business already knows (expected cycles,
    dated events, positive values), runs the structure search in a
    background thread and streams its levels as they finish. The result
    comes back as one JSON document the page draws: the data with the
    posterior mean and its 95 % band extended past the last point, each
    additive component with its English sentences and its checks, the BIC
    of every level, the probability of each property over every scored
    structure and the features read against the context.

Why a separate module
    :mod:`autostat.gui` serves the command line's form. The studio speaks
    in data rather than in options, so its state (uploaded tables, runs and
    their progress) and its payload live here, behind the same loopback
    host and the same rule: nothing leaves the machine.

What it consumes
    A table as the raw body of ``POST /api/datasets`` (CSV, TSV or
    Parquet; at most :data:`MAX_UPLOAD` bytes), or the name of one of the
    thirteen 2014 series. A run is a JSON object (see :func:`run_request`).

What it produces
    ``GET /api/runs/{id}`` returns the run's status, the levels finished so
    far and, once done, the payload of :func:`payload`.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import benchmarks
from .context import Context
from .describe import describe
from .features import _slug, features, structure_probabilities
from .model import posterior
from .search import Experiment, SearchResult, search

#: Largest table the studio accepts, in bytes.
MAX_UPLOAD = 50_000_000

#: Tables and runs kept in memory before the oldest is dropped.
KEEP = 20

#: Share of the time span the forecast extends past the last point.
EXTEND = 0.15

#: Points of the grid the curves are drawn on.
GRID = 400

#: Workers by default: two, so the machine stays usable.
DEFAULT_JOBS = 2


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded table.

    Parameters
    ----------
    content : bytes
        The file as sent.
    filename : str
        Its name, whose extension picks the reader.

    Returns
    -------
    pandas.DataFrame
        The table.

    Raises
    ------
    ValueError
        When the file is too large, empty or unreadable.
    """
    if len(content) > MAX_UPLOAD:
        raise ValueError(f"the file is larger than {MAX_UPLOAD // 1_000_000} MB")
    if not content:
        raise ValueError("the file is empty")
    name = filename.lower()
    try:
        if name.endswith(".parquet"):
            frame = pd.read_parquet(io.BytesIO(content))
        else:
            sep = "\t" if name.endswith((".tsv", ".tab")) else None
            frame = pd.read_csv(io.BytesIO(content), sep=sep, engine="python")
    except Exception as error:
        raise ValueError(f"could not read {filename}: {error}") from error
    if frame.empty:
        raise ValueError(f"{filename} has no rows")
    return frame


def describe_columns(frame: pd.DataFrame) -> dict[str, Any]:
    """Say what each column holds and guess the time and value columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table.

    Returns
    -------
    dict
        ``columns`` (name and kind: ``time``, ``number`` or ``text``),
        ``time`` and ``value`` (the guesses, or ``None``), ``rows`` and a
        ``preview`` of the first five rows as strings.
    """
    columns = []
    for name in frame.columns:
        values = frame[name]
        if pd.api.types.is_numeric_dtype(values):
            kind = "number"
        elif _dates(values) is not None:
            kind = "time"
        else:
            kind = "text"
        columns.append({"name": str(name), "kind": kind})
    times = [c["name"] for c in columns if c["kind"] == "time"]
    numbers = [c["name"] for c in columns if c["kind"] == "number"]
    named = [n for n in numbers if n.lower() in ("time", "t", "year", "date", "x")]
    time_guess = (times or named or [None])[0]
    values = [n for n in numbers if n != time_guess]
    return {
        "columns": columns,
        "time": time_guess,
        "value": values[0] if values else None,
        "rows": len(frame),
        "preview": frame.head(5).astype(str).to_dict(orient="records"),
    }


def _dates(values: pd.Series) -> pd.Series | None:
    """Parse a column as dates when most of it reads as dates.

    Parameters
    ----------
    values : pandas.Series
        One column.

    Returns
    -------
    pandas.Series or None
        The dates, or ``None`` when fewer than nine values in ten parse.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(values, errors="coerce", utc=True)
    return dates if dates.notna().mean() >= 0.9 else None


def series_of(frame: pd.DataFrame, time: str | None, value: str) -> pd.Series:
    """Build the series the search reads from two columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table.
    time : str or None
        Time column (dates or numbers); the row number when ``None``.
    value : str
        Value column.

    Returns
    -------
    pandas.Series
        Values indexed by time, rows with a missing time dropped.

    Raises
    ------
    ValueError
        When a column is missing or the value column is not numeric.
    """
    for column in (time, value):
        if column is not None and column not in frame.columns:
            raise ValueError(f"no column {column!r}")
    values = pd.to_numeric(frame[value], errors="coerce")
    if values.notna().sum() < 3:
        raise ValueError(f"the column {value!r} has fewer than three numbers")
    if time is None:
        index = pd.RangeIndex(len(frame))
    elif pd.api.types.is_numeric_dtype(frame[time]):
        index = pd.Index(frame[time].astype(float))
    else:
        dates = _dates(frame[time])
        if dates is None:
            raise ValueError(f"the column {time!r} is neither dates nor numbers")
        index = pd.DatetimeIndex(dates)
    series = pd.Series(values.to_numpy(), index=index, name=value)
    return series[series.index.notna()]


def run_request(fields: dict[str, Any]) -> dict[str, Any]:
    """Check a run request and fill in its defaults.

    Parameters
    ----------
    fields : dict
        ``dataset`` (an id from ``POST /api/datasets``) or ``example`` (a
        2014 series), ``time``, ``value``, ``name``, ``context`` (as
        :meth:`autostat.context.Context.from_dict` reads it), ``preset``
        (``abcd-2014`` or ``cks-2013``), ``max_depth``, ``max_points``,
        ``n_jobs``, ``seed`` and ``check``.

    Returns
    -------
    dict
        The same, typed and bounded.

    Raises
    ------
    ValueError
        When a field is missing, of the wrong type or out of range.
    """
    known = {
        "dataset", "example", "time", "value", "name", "context", "preset",
        "max_depth", "max_points", "n_jobs", "seed", "check",
    }  # fmt: skip
    unknown = set(fields) - known
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
    if bool(fields.get("dataset")) == bool(fields.get("example")):
        raise ValueError("give either a dataset or an example")
    if fields.get("example") and fields["example"] not in benchmarks.names():
        raise ValueError(f"no example {fields['example']!r}")
    preset = fields.get("preset", "abcd-2014")
    if preset not in ("abcd-2014", "cks-2013"):
        raise ValueError(f"unknown preset {preset!r}")

    def bounded(key: str, default: int, low: int, high: int) -> int:
        """Read an integer field and check its range.

        Parameters
        ----------
        key : str
            The field.
        default : int
            Its value when absent.
        low, high : int
            Inclusive bounds.

        Returns
        -------
        int
            The value.

        Raises
        ------
        ValueError
            When the value lies outside the bounds.
        """
        value = int(fields.get(key, default))
        if not low <= value <= high:
            raise ValueError(f"{key} must lie between {low} and {high}")
        return value

    context = fields.get("context") or {}
    Context.from_dict(context)  # fails here, not in the background thread
    return {
        "dataset": fields.get("dataset"),
        "example": fields.get("example"),
        "time": fields.get("time") or None,
        "value": fields.get("value") or None,
        "name": str(fields.get("name") or fields.get("example") or "data")[:200],
        "context": context,
        "preset": preset,
        "max_depth": bounded("max_depth", 4, 1, 10),
        "max_points": bounded("max_points", 500, 20, 2000),
        "n_jobs": bounded("n_jobs", DEFAULT_JOBS, 1, 64),
        "seed": bounded("seed", 1, 0, 2**31 - 1),
        "check": bool(fields.get("check", True)),
    }


@dataclass
class Run:
    """One search in progress or done.

    Parameters
    ----------
    id : str
        Its handle.
    request : dict
        What :func:`run_request` returned.
    status : str
        ``queued``, ``searching``, ``describing``, ``done`` or ``failed``.
    levels : list of dict
        One entry per finished level: depth, best BIC, best structure,
        candidates fitted and seconds.
    result : dict or None
        The payload once done.
    error : str or None
        Why it failed.
    started : float
        When it started, from :func:`time.time`.
    """

    id: str
    request: dict[str, Any]
    status: str = "queued"
    levels: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)

    def view(self) -> dict[str, Any]:
        """Return what the page polls.

        Returns
        -------
        dict
            Status, levels, elapsed seconds, the planned depth; the
            result or the error when there is one.
        """
        return {
            "id": self.id,
            "status": self.status,
            "levels": self.levels,
            "max_depth": self.request["max_depth"],
            "elapsed": time.time() - self.started,
            "result": self.result,
            "error": self.error,
        }


class _Levels(logging.Handler):
    """Copy the search's per-level log lines of one thread into its run."""

    def __init__(self, run: Run, thread: int) -> None:
        """Watch one thread.

        Parameters
        ----------
        run : Run
            Where the levels go.
        thread : int
            The identifier of the thread running the search.
        """
        super().__init__(logging.INFO)
        self.run = run
        self.thread = thread

    def emit(self, record: logging.LogRecord) -> None:
        """Keep a finished level.

        Parameters
        ----------
        record : logging.LogRecord
            A line of :mod:`autostat.search`.
        """
        if record.thread != self.thread or not isinstance(record.args, tuple):
            return
        if str(record.msg).startswith("level %d"):
            depth, fitted, _, _, seconds, _, best, structure = record.args
            self.run.levels.append(
                {
                    "depth": int(depth) + 1,
                    "bic": float(best),
                    "structure": str(structure),
                    "fitted": int(fitted),
                    "seconds": float(seconds),
                }
            )
        elif str(record.msg).startswith("event stage"):
            _, best, structure = record.args
            self.run.levels.append(
                {"depth": "event", "bic": float(best), "structure": str(structure)}
            )


def _finite(values: Any) -> list[float | None]:
    """Turn an array into a JSON list, non-finite values as ``None``.

    Parameters
    ----------
    values : array_like
        Numbers.

    Returns
    -------
    list
        Floats, or ``None`` where a value is not finite.
    """
    return [float(v) if math.isfinite(v) else None for v in np.ravel(values)]


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the JSON-ready entries of a feature row.

    Parameters
    ----------
    row : dict
        As :func:`autostat.features.features` returns it.

    Returns
    -------
    dict
        Numbers (finite or ``None``), strings, booleans and ``None``.
    """
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, bool | str) or value is None:
            out[key] = value
        elif isinstance(value, int | float | np.number):
            out[key] = float(value) if math.isfinite(float(value)) else None
    return out


def payload(
    result: SearchResult,
    x: np.ndarray,
    y: np.ndarray,
    unit: str,
    name: str,
    context: Context,
    check: bool,
) -> dict[str, Any]:
    """Describe a finished search as the page draws it.

    Parameters
    ----------
    result : SearchResult
        The search.
    x, y : numpy.ndarray
        The inputs (shape ``(n, 1)``) and the values the search read.
    unit : str
        ``"year"`` for dates, ``""`` otherwise.
    name : str
        The series' name.
    context : Context
        What the business knows (possibly empty).
    check : bool
        Run the model checks.

    Returns
    -------
    dict
        Data, grid, posterior mean and 95 % band of the whole model and of
        each component, sentences, statistics, checks, levels, formula,
        structure probabilities, features and the Markdown report.
    """
    model = result.best
    report = describe(model, x, y, name=name, unit=unit, check=check)
    low, high = float(x.min()), float(x.max())
    grid = np.linspace(low, high + EXTEND * (high - low), GRID)[:, None]
    mean, var = posterior(model, x, y, x_star=grid)
    sd = np.sqrt(np.clip(var, 0, None))
    parts = []
    # The sentences, statistics and checks follow the error-reduction order;
    # report.parts keeps the model's own order.
    ordered = [report.parts[j] for j in report.order["order"]]
    for i, part in enumerate(ordered):
        m, v = posterior(report.model, x, y, x_star=grid, component=part)
        s = np.sqrt(np.clip(v, 0, None))
        summary, details, extrapolation = report.texts[i]
        check_row = report.checks[i] if report.checks else None
        parts.append(
            {
                "summary": summary,
                "details": list(details),
                "extrapolation": list(extrapolation),
                "structure": part.structure(),
                "stats": _plain(report.stats[i]),
                "mae_reduction": float(report.order["mae_reductions"][i]),
                "mean": _finite(m),
                "lower": _finite(m - 1.96 * s),
                "upper": _finite(m + 1.96 * s),
                "check": _plain(vars(check_row)) if check_row else None,
            }
        )
    try:
        written = report.formula()
        formula = {"model": written.model, "laws": list(written.laws)}
    except Exception:  # a formula is a bonus: never lose a search to it
        formula = None
    models = [m for level in result.levels for m in level]
    return {
        "name": name,
        "unit": unit,
        "x": _finite(x),
        "y": _finite(y),
        "grid": _finite(grid),
        "mean": _finite(mean),
        "lower": _finite(mean - 1.96 * sd),
        "upper": _finite(mean + 1.96 * sd),
        "model": {
            "structure": model.kernel.structure(),
            "bic": float(model.bic),
            "nll": float(model.nll),
            "n": int(model.ndata),
        },
        "components": parts,
        "levels": [
            {
                "depth": d + 1,
                "bic": float(level[0].bic),
                "structure": level[0].kernel.structure(),
            }
            for d, level in enumerate(result.levels)
        ],
        "formula": formula,
        "probabilities": _plain(structure_probabilities(models, context)),
        "features": _plain(features(report, context)),
        "context": {
            "positive": context.positive,
            "cycles": [
                {"name": c.name, "slug": _slug(c.name), "period": c.period}
                for c in context.cycles
            ],
            "events": [
                {"name": e.name, "slug": _slug(e.name), "at": e.at}
                for e in context.events
            ],
        },
        "report": report.markdown(),
    }


class Studio:
    """The tables and runs of one host with the thread that searches.

    Searches run one at a time: each already uses ``n_jobs`` processes, so
    two at once would only slow both down.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.tables: dict[str, pd.DataFrame] = {}
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()

    def add_table(self, frame: pd.DataFrame) -> str:
        """Keep a table and return its id.

        Parameters
        ----------
        frame : pandas.DataFrame
            The table.

        Returns
        -------
        str
            Its id.
        """
        key = uuid.uuid4().hex
        self.tables[key] = frame
        while len(self.tables) > KEEP:
            self.tables.pop(next(iter(self.tables)))
        return key

    def start(self, request: dict[str, Any]) -> Run:
        """Queue a run and start its thread.

        Parameters
        ----------
        request : dict
            As :func:`run_request` returned it.

        Returns
        -------
        Run
            The run, queued.

        Raises
        ------
        ValueError
            When the dataset id is unknown.
        """
        if request["dataset"] and request["dataset"] not in self.tables:
            raise ValueError("this dataset is no longer loaded; load it again")
        run = Run(uuid.uuid4().hex, request)
        self.runs[run.id] = run
        while len(self.runs) > KEEP:
            self.runs.pop(next(iter(self.runs)))
        threading.Thread(target=self._work, args=(run,), daemon=True).start()
        return run

    def _work(self, run: Run) -> None:
        """Search, describe and keep the payload; record any failure.

        Parameters
        ----------
        run : Run
            The run to carry out.

        Raises
        ------
        ValueError
            Inside the thread, when the table has no numeric column; it is
            caught and recorded as the run's error, never raised to the host.
        """
        from . import prepare

        r = run.request
        logger = logging.getLogger("autostat.search")
        handler = _Levels(run, threading.get_ident())
        try:
            if r["example"]:
                x, y = benchmarks.load(r["example"])
                x, unit = np.asarray(x, float).reshape(-1, 1), "year"
                y = np.asarray(y, float).ravel()
            else:
                frame = self.tables[r["dataset"]]
                value = r["value"] or describe_columns(frame)["value"]
                if value is None:
                    raise ValueError("no numeric column to analyse")
                series = series_of(frame, r["time"], value)
                x, y, unit = prepare(series, r["max_points"])
            context = Context.from_dict(r["context"])
            preset = (
                Experiment.abcd_2014
                if r["preset"] == "abcd-2014"
                else Experiment.cks_2013
            )
            exp = preset(
                max_depth=r["max_depth"],
                n_jobs=r["n_jobs"],
                random_seed=r["seed"],
                **context.search_hints(),
            )
            with self.lock:
                run.status = "searching"
                run.started = time.time()
                logger.addHandler(handler)
                previous = logger.level
                logger.setLevel(logging.INFO)
                try:
                    result = search(x, y, exp)
                finally:
                    logger.removeHandler(handler)
                    logger.setLevel(previous)
                run.status = "describing"
                run.result = payload(result, x, y, unit, r["name"], context, r["check"])
            run.status = "done"
        except Exception as error:  # the page shows it rather than hanging
            run.error = f"{type(error).__name__}: {error}"
            run.status = "failed"


def examples() -> list[dict[str, Any]]:
    """List the thirteen 2014 series the page offers as examples.

    Returns
    -------
    list of dict
        Name, number of points and first and last time of each.
    """
    out = []
    for name in benchmarks.names():
        x, _ = benchmarks.load(name)
        out.append(
            {
                "name": name,
                "n": len(x),
                "start": float(np.min(x)),
                "end": float(np.max(x)),
            }
        )
    return out


#: Static files of the studio and their media types.
STATIC = {
    "studio.js": "text/javascript",
    "studio.css": "text/css",
}

#: Font files served under ``/fonts/``, from ``static/fonts``.
FONT_TYPES = {".woff2": "font/woff2", ".css": "text/css", ".txt": "text/plain"}


def add_routes(host: Any) -> None:
    """Attach the studio to the local GUI host.

    Parameters
    ----------
    host : fastapi.FastAPI
        The host of :func:`autostat.gui.create_gui_app`.
    """
    studio = Studio()
    static = resources.files("autostat").joinpath("static")

    @host.get("/", response_class=HTMLResponse)
    def page() -> str:
        """Serve the studio page.

        Returns
        -------
        str
            The HTML document.
        """
        return static.joinpath("studio.html").read_text(encoding="utf-8")

    @host.get("/static/{name}")
    def asset(name: str) -> Response:
        """Serve the studio's script or stylesheet.

        Parameters
        ----------
        name : str
            A key of :data:`STATIC`.

        Returns
        -------
        fastapi.responses.Response
            The file.

        Raises
        ------
        HTTPException
            404 for any other name.
        """
        if name not in STATIC:
            raise HTTPException(404, "no such file")
        return Response(static.joinpath(name).read_bytes(), media_type=STATIC[name])

    @host.get("/fonts/{family}/{name}")
    def font(family: str, name: str) -> Response:
        """Serve one self-hosted Roboto file.

        Parameters
        ----------
        family : str
            ``roboto``, ``roboto-serif`` or ``roboto-mono``.
        name : str
            A font, its stylesheet or its licence.

        Returns
        -------
        fastapi.responses.Response
            The file.

        Raises
        ------
        HTTPException
            404 for anything outside the three families.
        """
        suffix = "." + name.rsplit(".", 1)[-1]
        families = ("roboto", "roboto-serif", "roboto-mono")
        if family not in families or suffix not in FONT_TYPES or "/" in name:
            raise HTTPException(404, "no such file")
        path = static.joinpath("fonts", family, name)
        if not path.is_file():
            raise HTTPException(404, "no such file")
        return Response(path.read_bytes(), media_type=FONT_TYPES[suffix])

    @host.get("/api/examples")
    def example_list() -> list[dict[str, Any]]:
        """List the bundled 2014 series.

        Returns
        -------
        list of dict
            See :func:`examples`.
        """
        return examples()

    @host.get("/api/examples/{name}")
    def example_data(name: str) -> dict[str, Any]:
        """Return one bundled series, to draw before searching.

        Parameters
        ----------
        name : str
            A 2014 series.

        Returns
        -------
        dict
            ``x`` and ``y``.

        Raises
        ------
        HTTPException
            404 for an unknown name.
        """
        if name not in benchmarks.names():
            raise HTTPException(404, "no such example")
        x, y = benchmarks.load(name)
        return {"x": _finite(x), "y": _finite(y)}

    @host.post("/api/datasets")
    async def upload(request: Request, filename: str = "data.csv") -> JSONResponse:
        """Keep an uploaded table and describe its columns.

        Parameters
        ----------
        request : fastapi.Request
            The file as the raw body.
        filename : str, optional
            Its name, whose extension picks the reader.

        Returns
        -------
        fastapi.responses.JSONResponse
            The id and :func:`describe_columns`, or a 422 error envelope.
        """
        try:
            frame = read_table(await request.body(), filename)
        except ValueError as error:
            return JSONResponse({"error": str(error)}, 422)
        return JSONResponse({"id": studio.add_table(frame), **describe_columns(frame)})

    @host.post("/api/datasets/{key}/series")
    async def preview(key: str, request: Request) -> JSONResponse:
        """Return the series two columns make, to draw before searching.

        Parameters
        ----------
        key : str
            A table id.
        request : fastapi.Request
            JSON ``{"time": ..., "value": ...}``.

        Returns
        -------
        fastapi.responses.JSONResponse
            ``x``, ``y`` and ``unit``, or a 422 error envelope.
        """
        from . import prepare

        if key not in studio.tables:
            return JSONResponse({"error": "this dataset is no longer loaded"}, 422)
        fields = await request.json()
        try:
            series = series_of(
                studio.tables[key], fields.get("time") or None, fields["value"]
            )
            x, y, unit = prepare(series, int(fields.get("max_points", 500)))
        except (KeyError, ValueError) as error:
            return JSONResponse({"error": str(error)}, 422)
        return JSONResponse({"x": _finite(x), "y": _finite(y), "unit": unit})

    @host.post("/api/runs")
    async def start(request: Request) -> JSONResponse:
        """Start a search.

        Parameters
        ----------
        request : fastapi.Request
            JSON fields of :func:`run_request`.

        Returns
        -------
        fastapi.responses.JSONResponse
            The run as :meth:`Run.view` shows it, or a 422 error envelope.
        """
        fields = await request.json()
        try:
            run = studio.start(run_request(fields if isinstance(fields, dict) else {}))
        except (TypeError, ValueError) as error:
            return JSONResponse({"error": str(error)}, 422)
        return JSONResponse(run.view())

    @host.get("/api/runs/{key}")
    def status(key: str) -> dict[str, Any]:
        """Report a run's progress, then its result once done.

        Parameters
        ----------
        key : str
            A run id.

        Returns
        -------
        dict
            See :meth:`Run.view`.

        Raises
        ------
        HTTPException
            404 for an unknown run.
        """
        if key not in studio.runs:
            raise HTTPException(404, "no such run")
        return studio.runs[key].view()
