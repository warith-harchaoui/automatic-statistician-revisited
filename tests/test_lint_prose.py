"""The prose linter catches what it claims to catch; the repository passes it.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "lint_prose", ROOT / "scripts" / "lint_prose.py"
)
lint_prose = importlib.util.module_from_spec(_spec)
# Registered before execution: dataclasses look their module up by name.
sys.modules["lint_prose"] = lint_prose
_spec.loader.exec_module(lint_prose)


@pytest.mark.parametrize(
    "text",
    [
        "a dash " + chr(0x2014) + " here",
        "a range 1" + chr(0x2013) + "2",
        "word " + "-" * 2 + " word",
        "red" + "," + " and blue",
        "rouge" + "," + " et bleu",
    ],
)
def test_the_linter_flags_a_banned_habit(tmp_path, text):
    """Each banned habit produces a finding.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    text : str
        A line holding one banned habit.
    """
    path = tmp_path / "note.md"
    path.write_text(text + "\n")
    assert lint_prose.scan([path], tmp_path)


def test_the_linter_accepts_command_line_options(tmp_path):
    """A double hyphen starting an option is not a dash.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    path = tmp_path / "note.md"
    path.write_text("run with --max-depth 3 and --json\n")
    assert not lint_prose.scan([path], tmp_path)


def test_the_repository_follows_its_prose_rules():
    """Every tracked file passes."""
    assert lint_prose.scan(lint_prose.tracked_files(ROOT), ROOT) == []
