#!/usr/bin/env python3
"""Check that an autostat report is complete, without any dependency.

What this script does
    It reads a report directory and verifies what an answer will rely on:
    the data figure comes first, the formula and search-path sections are
    there, the BIC figure is drawn unless the report says why it cannot be
    and every figure the source names exists. An AutoGP summary has no
    formula or search path of its own: its posterior over structures stands
    for both.

Usage
    ``python check_report.py REPORT_DIRECTORY``

What it produces
    ``report OK``, exit status 0; otherwise one problem per line, exit
    status 1. A missing PDF is a note: the source is still usable.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORMULA = "The formula found"
PATH = "Search path and stopping criterion"
SUMMARY = "Executive summary"
POSTERIOR = "Posterior over structures"
NO_BIC = ("no BIC curve to draw", "no BIC to compare")


def problems(folder: Path) -> list[str]:
    """List what is wrong with a report directory.

    Parameters
    ----------
    folder : pathlib.Path
        The report directory.

    Returns
    -------
    list of str
        One sentence per problem; empty for a complete report.
    """
    markdown, latex = folder / "report.md", folder / "report.tex"
    source = markdown if markdown.exists() else latex
    if not source.exists():
        return [f"no report.md or report.tex in {folder}"]
    text = source.read_text(encoding="utf-8")
    pattern = (
        r"!\[[^\]]*\]\(([^)]+)\)"
        if source == markdown
        else r"includegraphics[^{]*\{([^}]+)\}"
    )
    figures = re.findall(pattern, text)
    found = []
    required = (POSTERIOR,) if POSTERIOR in text else (SUMMARY, FORMULA, PATH)
    for section in required:
        if section not in text:
            found.append(f"section missing: {section}")
    data = [name for name in figures if Path(name).stem == "data"]
    if not data:
        found.append("the data figure is missing")
    elif SUMMARY in text and text.index(data[0]) > text.index(SUMMARY):
        found.append("the data figure does not come first")
    explained = any(reason in text for reason in NO_BIC)
    if not explained and not any(Path(name).stem == "bic" for name in figures):
        found.append("the BIC curve is missing")
    found += [f"figure not found: {n}" for n in figures if not (folder / n).exists()]
    return found


def main(argv: list[str]) -> int:
    """Check the directory named on the command line.

    Parameters
    ----------
    argv : list of str
        Arguments without the program name.

    Returns
    -------
    int
        0 for a complete report, 1 for problems, 2 for a usage error.
    """
    if len(argv) != 1:
        print("usage: check_report.py REPORT_DIRECTORY", file=sys.stderr)
        return 2
    folder = Path(argv[0])
    found = problems(folder)
    for problem in found:
        print(problem)
    if found:
        return 1
    if not (folder / "report.pdf").exists():
        print("note: no report.pdf (source only)")
    print("report OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
