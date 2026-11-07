#!/usr/bin/env python3
"""Check the repository's prose against its writing rules.

What this script does
    It scans every tracked text file (Markdown, Python, YAML, TOML, CFF) for
    four habits the project's writing charter bans, in text and in code
    comments alike:

    * the em dash and the en dash, in any language;
    * a spaced double hyphen used as a dash;
    * a comma before "and" in English;
    * a comma before "et" in French.

    It prints one line per finding and exits with status 1 when there is any.

Why it exists
    A rule that nobody measures is a wish. Running this in ``make check`` and
    in continuous integration keeps the rule true after every change.

What it skips
    Files reproduced verbatim from elsewhere, where the wording is not ours:
    the original licence, the 2014 result files and ``CODING.md``.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".cff", ".txt"}

#: Verbatim copies of other people's text.
VERBATIM = {"LICENSE-ORIGINAL", "CODING.md"}
VERBATIM_DIRS = ("src/autostat/reference/",)

RULES: dict[str, re.Pattern[str]] = {
    "em or en dash": re.compile("[" + chr(0x2013) + chr(0x2014) + "]"),
    "spaced double hyphen used as a dash": re.compile(r"(?<=\w) -- (?=\w)"),
    "comma before 'and'": re.compile(r",\s+and\b"),
    "comma before 'et'": re.compile(r",\s+et\b"),
}


@dataclass(frozen=True)
class Finding:
    """One rule broken at one place.

    Parameters
    ----------
    path : str
        File, relative to the repository root.
    line : int
        One-based line number.
    rule : str
        Which rule.
    text : str
        The offending line, stripped.
    """

    path: str
    line: int
    rule: str
    text: str


def tracked_files(root: Path) -> list[Path]:
    """List the files git tracks or would track, falling back to a walk.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Returns
    -------
    list of pathlib.Path
        Candidate files with a prose-bearing suffix.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        paths = [root / p for p in listed]
    except (OSError, subprocess.CalledProcessError):
        paths = [p for p in root.rglob("*") if ".git" not in p.parts]
    return [p for p in paths if p.is_file() and p.suffix in SUFFIXES]


def is_verbatim(relative: str) -> bool:
    """Tell whether a file is a verbatim copy exempt from the rules.

    Parameters
    ----------
    relative : str
        Path relative to the repository root, with forward slashes.

    Returns
    -------
    bool
        True for exempt files.
    """
    return relative in VERBATIM or relative.startswith(VERBATIM_DIRS)


def scan(paths: Iterable[Path], root: Path) -> list[Finding]:
    """Apply every rule to every line of every file.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        Files to scan.
    root : pathlib.Path
        Repository root, for relative paths.

    Returns
    -------
    list of Finding
        Every finding, in file and line order.
    """
    findings = []
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if is_verbatim(relative) or relative == "scripts/lint_prose.py":
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for rule, pattern in RULES.items():
                if pattern.search(line):
                    findings.append(Finding(relative, number, rule, line.strip()))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Scan the repository, or the given files, and report.

    Parameters
    ----------
    argv : sequence of str, optional
        Files to scan instead of the whole repository.

    Returns
    -------
    int
        0 when clean, 1 when anything was found.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(a).resolve() for a in args] if args else tracked_files(ROOT)
    findings = scan(paths, ROOT)
    for f in findings:
        sys.stdout.write(f"{f.path}:{f.line}: {f.rule}: {f.text}\n")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
