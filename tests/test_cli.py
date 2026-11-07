"""The command line: help, version, errors and one real run.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autostat import __version__
from autostat.cli import EXIT_INVALID, main, make_parser


def test_building_the_parser_runs_nothing():
    """make_parser has no side effect and knows its program name."""
    assert make_parser().prog == "autostat"


def test_help_exits_with_status_zero(capsys):
    """--help prints usage and exits 0."""
    with pytest.raises(SystemExit) as stop:
        main(["--help"])
    assert stop.value.code == 0


def test_version_prints_the_package_version(capsys):
    """--version prints the installed version."""
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_a_missing_file_returns_the_invalid_input_status(tmp_path):
    """A path that does not exist gives exit status 2."""
    assert main([str(tmp_path / "absent.csv")]) == EXIT_INVALID


def test_a_missing_column_returns_the_invalid_input_status(tmp_path):
    """Naming a column the file lacks gives exit status 2."""
    path = tmp_path / "data.csv"
    pd.DataFrame({"t": [1, 2, 3], "y": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    assert main([str(path), "--time", "t", "--value", "z"]) == EXIT_INVALID


@pytest.mark.slow
def test_a_run_prints_one_json_object(tmp_path, capsys):
    """--json prints the structure, its score and the report path."""
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
            "--json",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert (status, sorted(payload)) == (
        0,
        ["bic", "components", "report", "structure"],
    )
