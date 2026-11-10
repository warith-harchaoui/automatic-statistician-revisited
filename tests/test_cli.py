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
