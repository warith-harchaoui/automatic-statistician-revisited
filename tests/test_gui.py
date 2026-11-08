"""The GUI host: its page, its strings and the checks on what a form sends.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import re
import sys

import pytest
import yaml
from fastapi.testclient import TestClient

from autostat.cli import command
from autostat.gui import create_gui_app, page, run_options, strings

#: A stand-in Tailwind CLI: writes one rule per class it finds in the content.
STAND_IN_CLI = """\
import glob, json, re, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
source = open(args["--config"]).read()
config = json.loads(source[len("module.exports = "):].rstrip().rstrip(";"))
names = set()
for pattern in config["content"]:
    for path in glob.glob(pattern):
        text = open(path).read()
        for value in re.findall(r'class(?:Name)?\\s*=\\s*"([^"]*)"', text):
            names.update(value.split())
        for value in re.findall(r'"([a-z0-9 :-]+)"', text):
            names.update(value.split())
open(args["--output"], "w").write("/*stand-in*/" + "".join("." + n + "{}" for n in sorted(names)))
"""


@pytest.fixture(autouse=True)
def offline_tailwind(tmp_path, monkeypatch):
    """Compile with a stand-in Tailwind CLI from a temporary cache, offline.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Sets sprezzature-cli-gui's cache and offline switches.
    """
    from sprezzature_cli_gui.tailwind import cli_name

    folder = tmp_path / "tailwind"
    folder.mkdir()
    stand_in = folder / f"0.0.1-{cli_name()}"
    stand_in.write_text(f"#!{sys.executable}\n{STAND_IN_CLI}")
    stand_in.chmod(0o755)
    monkeypatch.setenv("SPREZZATURE_CLI_GUI_CACHE", str(tmp_path))
    monkeypatch.setenv("SPREZZATURE_CLI_GUI_OFFLINE", "1")


@pytest.fixture
def client():
    """A client of the local GUI host.

    Returns
    -------
    fastapi.testclient.TestClient
        The client.
    """
    return TestClient(create_gui_app())


def test_the_page_covers_every_option_of_the_command():
    """Each Click option has its field, bar the two a form has no use for."""
    from sprezzature_cli_gui import audit, walk

    html = page()
    assert '<script type="module" src="/run.js"></script>' in html
    missing = {f.subject for f in audit(walk(command, prog="autostat"), html)}
    assert missing <= {"--version", "--json"}


def test_the_page_loads_nothing_from_another_host(client):
    """No CDN: Tailwind is embedded, every script is served by the host itself.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the GUI host.
    """
    html = client.get("/").text
    assert re.findall(r'(?:src|href)="(?:https?:)?//', html) == []
    assert "only the utilities this page uses" in html


def test_the_run_panel_classes_are_compiled_into_the_page():
    """Classes run.js adds at run time are in the page's stylesheet."""
    style = page().split("<style>", 1)[1].split("</style>", 1)[0]
    assert ".rounded-full{}" in style and ".list-decimal{}" in style


def test_every_language_has_every_key():
    """English and French translate the same keys."""
    from importlib import resources

    table = yaml.safe_load(
        resources.files("autostat").joinpath("locales/i18n.yaml").read_text()
    )
    assert all(set(by_lang) == {"en", "fr"} for by_lang in table.values())


def test_an_unknown_language_falls_back_to_english():
    """A browser in German reads the English strings."""
    assert strings("de")["gui.run.button"] == "Run on this machine"


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"path": "x.csv", "shell": "rm -rf /"}, "unknown field"),
        ({"path": "x.csv", "max_depth": "ten"}, "max_depth"),
        ({"path": "x.csv", "report_format": "html"}, "report_format"),
        ({}, "path"),
    ],
)
def test_fields_are_checked_against_the_command(fields, reason):
    """The form gets exactly what the command line accepts, nothing else.

    Parameters
    ----------
    fields : dict
        What a form might send.
    reason : str
        A word the refusal must contain.
    """
    with pytest.raises(ValueError, match=reason):
        run_options(fields)


def test_defaults_come_from_the_command():
    """A field left empty takes the command line's default."""
    options = run_options({"path": "x.csv", "no_check": True})
    assert (options["max_depth"], options["report_format"], options["check"]) == (
        10,
        "markdown",
        False,
    )


def test_a_failed_run_answers_with_the_envelope(client, tmp_path):
    """A missing file is a 422 with a reason, not a crash.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the GUI host.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    response = client.post("/run", json={"path": str(tmp_path / "absent.csv")})
    assert (response.status_code, response.json()["error"]["code"]) == (
        422,
        "invalid_input",
    )


def test_files_outside_a_run_are_not_served(client):
    """The host serves only the files of reports it wrote.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        Fixture: the GUI host.
    """
    assert client.get("/runs/unknown/../../etc/passwd").status_code == 404


def test_the_page_is_built_once_per_host(monkeypatch):
    """Building the page compiles its stylesheet, so a second visit must not.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Counts the builds.
    """
    import autostat.gui as gui

    built = []
    monkeypatch.setattr(gui, "page", lambda: built.append(1) or "page")
    host = TestClient(gui.create_gui_app())
    assert [host.get("/").text, host.get("/").text] == ["page", "page"]
    assert len(built) == 1
