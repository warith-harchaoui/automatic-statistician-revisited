"""The studio: its tables, its checks on a run request and its routes.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from autostat import studio
from autostat.gui import create_gui_app

CSV = b"date,sales,store\n" + b"".join(
    f"2024-01-{d:02d},{10 + d % 7},north\n".encode() for d in range(1, 29)
)


@pytest.fixture(scope="module")
def client():
    """A host with the studio, without opening a browser.

    Returns
    -------
    fastapi.testclient.TestClient
        The client.
    """
    return TestClient(create_gui_app())


def test_a_csv_is_read_and_its_columns_guessed():
    """Dates become the time column, the first number the value."""
    frame = studio.read_table(CSV, "sales.csv")
    seen = studio.describe_columns(frame)
    kinds = {c["name"]: c["kind"] for c in seen["columns"]}
    assert kinds == {"date": "time", "sales": "number", "store": "text"}
    assert (seen["time"], seen["value"], seen["rows"]) == ("date", "sales", 28)


def test_an_empty_or_unreadable_file_says_why():
    """The reader learns what is wrong instead of seeing a crash."""
    with pytest.raises(ValueError, match="empty"):
        studio.read_table(b"", "x.csv")
    with pytest.raises(ValueError, match="could not read"):
        studio.read_table(b"\x00\x01", "x.parquet")


def test_two_columns_make_a_dated_series():
    """The series is indexed by dates and keeps every numeric value."""
    series = studio.series_of(studio.read_table(CSV, "s.csv"), "date", "sales")
    assert isinstance(series.index, pd.DatetimeIndex)
    assert len(series) == 28
    with pytest.raises(ValueError, match="fewer than three numbers"):
        studio.series_of(studio.read_table(CSV, "s.csv"), "date", "store")


def test_a_run_request_is_bounded_and_defaulted():
    """Two workers and four levels unless asked; nonsense is refused."""
    request = studio.run_request({"example": "01-airline"})
    assert (request["n_jobs"], request["max_depth"], request["check"]) == (2, 4, True)
    for fields, message in [
        ({}, "either a dataset or an example"),
        ({"example": "nope"}, "no example"),
        ({"example": "01-airline", "max_depth": 99}, "max_depth"),
        ({"example": "01-airline", "shell": "rm"}, "unknown fields"),
        (
            {
                "example": "01-airline",
                "context": {"cycles": [{"name": "w", "period": "a week"}]},
            },
            "duration",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            studio.run_request(fields)


def test_every_string_of_the_page_exists_in_both_languages():
    """No key of the page falls back to its id, in English or in French."""
    import re
    from importlib import resources

    static = resources.files("autostat").joinpath("static")
    used = set()
    for name in ("studio.html", "studio.js"):
        text = static.joinpath(name).read_text()
        used |= set(re.findall(r'data-i18n(?:-label)?="(studio\.\w+)"', text))
        used |= set(re.findall(r't\("(studio\.\w+)"', text))
        used |= set(re.findall(r'"(studio\.\w+)"\]', text))
    table = yaml.safe_load(
        resources.files("autostat").joinpath("locales/i18n.yaml").read_text()
    )
    missing = sorted(k for k in used if not {"en", "fr"} <= set(table.get(k, {})))
    assert not missing


def test_the_page_its_files_and_the_form_are_served(client):
    """The studio owns the root; the command line's form moved to /form.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        The fixture.
    """
    assert "autostat" in client.get("/").text
    assert (
        client.get("/static/studio.js")
        .headers["content-type"]
        .startswith("text/javascript")
    )
    assert client.get("/static/studio.css").status_code == 200
    assert client.get("/fonts/roboto/Roboto-Variable.woff2").status_code == 200
    assert client.get("/fonts/roboto/../../gui.py").status_code == 404
    assert client.get("/static/gui.py").status_code == 404
    assert len(client.get("/api/examples").json()) == 13


def test_an_upload_previews_its_series(client):
    """Upload, then draw the series two columns make.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        The fixture.
    """
    body = client.post("/api/datasets?filename=sales.csv", content=CSV).json()
    series = client.post(
        f"/api/datasets/{body['id']}/series", json={"time": "date", "value": "sales"}
    ).json()
    assert len(series["x"]) == 28
    assert series["unit"] == "year"
    assert client.post("/api/datasets?filename=x.csv", content=b"").status_code == 422


@pytest.mark.slow
def test_a_short_search_runs_to_a_result(client):
    """One level on a small upload comes back with every part of the payload.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        The fixture.
    """
    t = np.arange(40)
    y = (
        0.3 * t
        + np.sin(2 * np.pi * t / 7)
        + np.random.default_rng(0).normal(0, 0.1, 40)
    )
    csv = "t,y\n" + "".join(f"{a},{b}\n" for a, b in zip(t, y, strict=True))
    key = client.post("/api/datasets?filename=s.csv", content=csv.encode()).json()["id"]
    run = client.post(
        "/api/runs",
        json={
            "dataset": key,
            "time": "t",
            "value": "y",
            "max_depth": 1,
            "n_jobs": 1,
            "check": False,
            "context": {"cycles": [{"name": "weekly", "period": 7}]},
        },
    ).json()
    for _ in range(600):
        view = client.get(f"/api/runs/{run['id']}").json()
        if view["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert view["status"] == "done", view["error"]
    result = view["result"]
    assert view["levels"] and len(result["grid"]) == studio.GRID
    assert result["components"] and result["report"].startswith("---")
    assert "cycle_weekly_present" in result["features"]
    assert "p_cycle_weekly" in result["probabilities"]


def test_each_sentence_goes_with_its_own_component():
    """The cycle's sentence comes with the periodic curve, noise with noise.

    The report keeps its components in the model's order but writes its
    sentences in the order of their error reduction; the payload must not
    pair one component's sentence with another's curve.
    """
    from autostat import benchmarks
    from autostat.context import Context
    from autostat.search import SearchResult

    x, y = benchmarks.load("01-airline")
    model = benchmarks.published("01-airline")[-1]
    result = SearchResult(model, [[model]], None, 0.0)
    out = studio.payload(
        result,
        np.asarray(x, float).reshape(-1, 1),
        np.ravel(y),
        "year",
        "airline",
        Context(),
        False,
    )
    for part in out["components"]:
        if "periodic" in part["summary"]:
            assert "Per" in part["structure"]
        if "noise" in part["summary"].lower():
            assert "WN" in part["structure"]


def test_a_language_model_name_is_checked_before_the_run():
    """An Ollama model name passes; anything that is not one is refused."""
    from autostat.studio import run_request

    assert (
        run_request({"example": "01-airline", "propose": "gemma3:12b"})["propose"]
        == "gemma3:12b"
    )
    assert run_request({"example": "01-airline"})["propose"] is None
    with pytest.raises(ValueError, match="Ollama"):
        run_request({"example": "01-airline", "propose": "x; rm -rf /"})
