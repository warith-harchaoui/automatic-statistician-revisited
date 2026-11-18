"""The command line: help, version, errors and one real run.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json

import click
import numpy as np
import pandas as pd
import pytest

from autostat import __version__
from autostat.cli import EXIT_INVALID, command, main


def test_importing_the_command_runs_nothing():
    """The Click command is an importable object that knows its name."""
    assert isinstance(command, click.Command)
    assert command.name == "autostat"


def test_help_exits_with_status_zero(capsys):
    """--help prints usage and returns 0.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Pytest's capture of standard output and error.
    """
    assert main(["--help"]) == 0
    assert "--format [markdown|latex]" in capsys.readouterr().out


def test_version_prints_the_package_version(capsys):
    """--version prints the installed version.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Pytest's capture of standard output and error.
    """
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_a_missing_file_returns_the_invalid_input_status(tmp_path):
    """A path that does not exist gives exit status 2.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    assert main([str(tmp_path / "absent.csv")]) == EXIT_INVALID


def test_a_missing_column_returns_the_invalid_input_status(tmp_path):
    """Naming a column the file lacks gives exit status 2.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    path = tmp_path / "data.csv"
    pd.DataFrame({"t": [1, 2, 3], "y": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    assert main([str(path), "--time", "t", "--value", "z"]) == EXIT_INVALID


@pytest.mark.parametrize("option", [["--format", "html"], ["--preset", "abcd-2015"]])
def test_an_invalid_choice_returns_the_invalid_input_status(tmp_path, capsys, option):
    """An unknown format or preset gives exit status 2 and says why on stderr.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    capsys : pytest.CaptureFixture
        Pytest's capture of standard output and error.
    option : list of str
        The offending option and value.
    """
    assert main([str(tmp_path / "data.csv"), *option]) == EXIT_INVALID
    captured = capsys.readouterr()
    assert (captured.out, "Invalid value" in captured.err) == ("", True)


@pytest.mark.slow
def test_a_run_prints_one_json_object(tmp_path, capsys):
    """--json prints the structure, its score and the report paths.

    The path holds a space and a non-ASCII letter, as a user's file may.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    capsys : pytest.CaptureFixture
        Pytest's capture of standard output and error.
    """
    path = tmp_path / "series with spaces é.csv"
    t = np.arange(60)
    y = np.sin(2 * np.pi * t / 12) + 0.05 * t
    pd.DataFrame({"t": t, "y": y}).to_csv(path, index=False)
    status = main(
        [
            str(path),
            "--time",
            "t",
            "--max-depth",
            "1",
            "--n-jobs",
            "1",
            "--no-check",
            "--no-pdf",
            "--format",
            "latex",
            "--json",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert (status, sorted(payload)) == (
        0,
        ["bic", "components", "formula", "pdf", "report", "search_path", "structure"],
    )
    assert payload["report"].endswith("report.tex")


def test_importing_the_library_loads_no_surface():
    """``import autostat`` needs neither Click nor FastAPI nor the MCP server."""
    import subprocess
    import sys

    code = (
        "import sys, autostat; "
        "print([m for m in ('click', 'fastapi', 'fastapi_mcp', 'uvicorn') if m in sys.modules])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.stdout.strip() == "[]", done.stderr


def test_a_missing_context_file_returns_the_invalid_input_status(tmp_path):
    """A --context path that does not exist is invalid input, not a crash.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    path = tmp_path / "s.csv"
    pd.DataFrame({"t": np.arange(10), "y": np.arange(10.0)}).to_csv(path, index=False)
    status = main([str(path), "--time", "t", "--context", str(tmp_path / "no.yaml")])
    assert status == EXIT_INVALID


def test_the_context_seeds_the_search_starts():
    """The expected periods and event dates reach the search settings."""
    from autostat.context import Context
    from autostat.service import experiment_for

    context = Context.from_dict(
        {
            "cycles": [{"name": "yearly", "period": "1y"}],
            "events": [{"name": "reform", "at": 2005.0}],
        }
    )
    exp = experiment_for("abcd-2014", 2, hints=context.search_hints())
    assert (exp.period_hints, exp.location_hints) == ([1.0], [2005.0])


def test_spectral_asks_for_periodogram_starts(tmp_path, monkeypatch):
    """--spectral N and --breaks N reach the search settings; the presets keep 0.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the analysis with a recorder.
    """
    import autostat.service

    seen = []

    def record(series, name, out_dir, experiment, *args, **kwargs):
        """Keep the settings and stop before any search.

        Parameters
        ----------
        series, name, out_dir, experiment, *args, **kwargs
            What :func:`autostat.service.analyse_series` passes.

        Raises
        ------
        ValueError
            Always, once the settings are kept.
        """
        seen.append((experiment.spectral_starts, experiment.break_starts))
        raise ValueError("recorded")

    monkeypatch.setattr(autostat.service, "analyse", record)
    path = tmp_path / "data.csv"
    pd.DataFrame({"t": np.arange(24.0), "y": np.arange(24.0)}).to_csv(path, index=False)
    for extra in ([], ["--spectral", "3"], ["--breaks", "2"]):
        assert main([str(path), "--time", "t", *extra]) == EXIT_INVALID
    assert seen == [(0, 0), (3, 0), (0, 2)]


@pytest.mark.slow
def test_a_language_model_is_asked_at_every_level(tmp_path, capsys, monkeypatch):
    """--propose asks the named model once per level, told the context.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    capsys : pytest.CaptureFixture
        Pytest's capture of standard output and error.
    monkeypatch : pytest.MonkeyPatch
        Replaces the Ollama client.
    """
    import autostat.propose

    asked = []

    def fake(model: str):
        """Stand in for :func:`autostat.propose.ollama`.

        Parameters
        ----------
        model : str
            The model named on the command line.

        Returns
        -------
        callable
            Records each prompt and proposes one structure.
        """

        def ask(prompt: str) -> str:
            """Record the prompt and answer.

            Parameters
            ----------
            prompt : str
                What the search asks.

            Returns
            -------
            str
                One structure.
            """
            asked.append((model, prompt))
            return "Per x SE + WN"

        return ask

    monkeypatch.setattr(autostat.propose, "ollama", fake)
    path = tmp_path / "s.csv"
    t = np.arange(120)
    pd.DataFrame({"t": t, "y": np.sin(2 * np.pi * t / 12)}).to_csv(path, index=False)
    context = tmp_path / "context.yaml"
    context.write_text("cycles:\n  - name: yearly\n    period: 12\n")
    status = main(
        [
            str(path),
            "--time",
            "t",
            "--max-depth",
            "2",
            "--n-jobs",
            "1",
            "--no-check",
            "--no-pdf",
            "--json",
            "--out",
            str(tmp_path / "out"),
            "--context",
            str(context),
            "--propose",
            "some-model",
        ]
    )
    capsys.readouterr()
    assert status == 0
    assert [m for m, _ in asked] == ["some-model", "some-model"]
    assert "'yearly'" in asked[0][1]
