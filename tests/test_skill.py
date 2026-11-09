"""The Agent Skill follows the rules of Anthropic's skills guide.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "autostat"


def frontmatter() -> tuple[str, dict]:
    """Read the raw frontmatter of SKILL.md and parse it.

    Returns
    -------
    str, dict
        The text between the delimiters; its YAML.
    """
    raw = (SKILL / "SKILL.md").read_text().split("---\n")[1]
    return raw, yaml.safe_load(raw)


def test_the_frontmatter_follows_the_guide():
    """Kebab-case name matching the folder, bounded fields, no angle brackets."""
    raw, meta = frontmatter()
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
    assert meta["name"] == SKILL.name
    assert not re.search("claude|anthropic", meta["name"])
    assert len(meta["description"]) < 1024
    assert 1 <= len(meta["compatibility"]) <= 500
    assert "<" not in raw and ">" not in raw.replace(">-", "")
    assert meta["metadata"]["version"]


def test_the_description_says_what_and_when():
    """It names the outcome, the moment to use it and when not to."""
    description = frontmatter()[1]["description"]
    assert "formula" in description
    assert "Use for" in description
    assert "Not for" in description


def test_the_folder_is_uploadable():
    """No README inside, no symbolic link, a short body."""
    assert not list(SKILL.rglob("README*"))
    assert not [p for p in SKILL.rglob("*") if p.is_symlink()]
    assert len((SKILL / "SKILL.md").read_text().split()) < 5000


def test_the_routing_catalogue_is_in_step():
    """references/triggers.md is TRIGGERS.md: run scripts/build_skill.py if not."""
    copied = (SKILL / "references" / "triggers.md").read_text()
    assert copied == (ROOT / "TRIGGERS.md").read_text()


def test_the_trigger_set_has_the_size_the_guide_asks():
    """10 to 20 queries that should trigger, others that should not."""
    cases = yaml.safe_load((SKILL / "evals" / "triggers.yaml").read_text())
    assert 10 <= len(cases["should_trigger"]) <= 20
    assert len(cases["should_not_trigger"]) >= 3


def test_the_skill_names_only_tools_that_exist():
    """Every MCP tool the instructions call is published by the server."""
    from autostat.mcp import TOOLS

    named = set(re.findall(r"`([a-z]+_[a-z]+)`", (SKILL / "SKILL.md").read_text()))
    assert {
        "analyse_series",
        "get_report",
        "list_series",
        "describe_published",
    } <= named
    assert {"analyse_series", "get_report", "list_series", "describe_published"} <= set(
        TOOLS
    )


def _checker():
    """Load the skill's validation script.

    Returns
    -------
    module
        ``check_report``.
    """
    path = SKILL / "scripts" / "check_report.py"
    spec = importlib.util.spec_from_file_location("check_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_checker_passes_a_complete_report(tmp_path):
    """Data first, both sections, figures present.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    (tmp_path / "data.svg").write_text("x")
    (tmp_path / "bic.svg").write_text("x")
    (tmp_path / "report.md").write_text(
        "![d](data.svg)\n## Executive summary\n## The formula found\n"
        "## Search path and stopping criterion\n![b](bic.svg)\n"
    )
    assert _checker().problems(tmp_path) == []


def test_the_checker_names_what_is_wrong(tmp_path):
    """A report with no data figure and a lost figure is refused.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    (tmp_path / "report.md").write_text("## Executive summary\n![b](bic.svg)\n")
    found = " ".join(_checker().problems(tmp_path))
    assert "The formula found" in found
    assert "data figure is missing" in found
    assert "figure not found: bic.svg" in found


def test_the_checker_accepts_an_autogp_summary(tmp_path):
    """A posterior summary needs its data figure and its BIC figure, no path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    (tmp_path / "data.svg").write_text("x")
    (tmp_path / "report.md").write_text(
        "![d](data.svg)\n## Posterior over structures\n"
    )
    assert _checker().problems(tmp_path) == ["the BIC curve is missing"]
    (tmp_path / "bic.svg").write_text("x")
    (tmp_path / "report.md").write_text(
        "![d](data.svg)\n## Posterior over structures\n![b](bic.svg)\n"
    )
    assert _checker().problems(tmp_path) == []


def test_the_checker_accepts_a_missing_curve_the_report_explains(tmp_path):
    """A one-level search says it has no curve; that is not a defect.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    (tmp_path / "data.svg").write_text("x")
    (tmp_path / "report.md").write_text(
        "![d](data.svg)\n## Executive summary\n## The formula found\n"
        "## Search path and stopping criterion\n"
        "One level was searched: there is no BIC curve to draw.\n"
    )
    assert _checker().problems(tmp_path) == []


def test_the_checker_needs_no_dependency():
    """It runs in an isolated interpreter, as an uploaded skill would."""
    script = SKILL / "scripts" / "check_report.py"
    done = subprocess.run(
        [sys.executable, "-I", str(script)], capture_output=True, text=True
    )
    assert done.returncode == 2
    assert "usage" in done.stderr
