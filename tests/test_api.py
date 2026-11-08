"""The HTTP API: its routes, its error envelope and its OpenAPI contract.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autostat.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A test client whose reports go to a temporary directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Sets ``AUTOSTAT_REPORTS_DIR``.

    Returns
    -------
    fastapi.testclient.TestClient
        The client.
    """
    monkeypatch.setenv("AUTOSTAT_REPORTS_DIR", str(tmp_path))
    return TestClient(create_app())


def test_health_says_ok_and_the_version(client):
    """GET /health is cheap and says which version answers.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the API.
    """
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_the_series_are_the_thirteen_of_the_paper(client):
    """GET /v1/series lists the 2014 benchmark.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the API.
    """
    names = client.get("/v1/series").json()
    assert (len(names), names[0]) == (13, "01-airline")


def test_operation_ids_are_stable(client):
    """The OpenAPI document names every route as the MCP tools expect.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the API.
    """
    paths = client.get("/openapi.json").json()["paths"]
    ids = {op["operationId"] for route in paths.values() for op in route.values()}
    assert ids == {
        "health",
        "list_series",
        "describe_published",
        "analyse_series",
        "get_report",
        "get_report_file",
    }


@pytest.mark.parametrize(
    ("route", "payload", "status", "code"),
    [
        ("/v1/published", {"series": "99-none"}, 422, "invalid_input"),
        (
            "/v1/published",
            {"series": "01-airline", "report_format": "html"},
            422,
            "invalid_input",
        ),
        ("/v1/analyses", {"values": [1.0, 2.0]}, 422, "invalid_input"),
        (
            "/v1/analyses",
            {"values": [1.0, 2.0, 3.0], "max_depth": 7},
            422,
            "invalid_input",
        ),
        (
            "/v1/analyses",
            {"values": [1.0, 2.0, 3.0], "times": [1.0]},
            422,
            "invalid_input",
        ),
    ],
)
def test_invalid_input_gets_the_envelope(client, route, payload, status, code):
    """Every refusal has the same shape and a reason.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the API.
    route : str
        The route posted to.
    payload : dict
        The invalid body.
    status : int
        The expected status.
    code : str
        The expected error code.
    """
    response = client.post(route, json=payload)
    assert (response.status_code, response.json()["error"]["code"]) == (status, code)
    assert response.json()["error"]["message"]


@pytest.mark.parametrize(
    "path",
    [
        "/v1/reports/not-an-id",
        "/v1/reports/" + "0" * 32,
        "/v1/reports/" + "0" * 32 + "/files/x",
    ],
)
def test_unknown_reports_are_404_without_paths(client, path, tmp_path):
    """A missing report is a 404 whose message names no server path.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the API.
    path : str
        A route for a report that does not exist.
    tmp_path : pathlib.Path
        The reports directory, which must not leak.
    """
    response = client.get(path)
    assert response.status_code == 404
    assert str(tmp_path) not in response.text


@pytest.mark.slow
def test_a_published_model_is_described_and_read_back(client):
    """POST /v1/published writes a LaTeX report that GET reads back.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the API.
    """
    body = client.post(
        "/v1/published",
        json={
            "series": "01-airline",
            "report_format": "latex",
            "check": False,
            "pdf": False,
        },
    ).json()
    assert body["structure"] == "SE + Lin + WN x Lin + SE x Per x Lin"
    source = client.get(f"/v1/reports/{body['report_id']}").json()
    assert source["report_format"] == "latex"
    assert source["source"].startswith(r"\documentclass")
    figure = client.get(f"/v1/reports/{body['report_id']}/files/fit.pdf")
    assert figure.content.startswith(b"%PDF")
