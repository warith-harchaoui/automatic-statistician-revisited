"""Measure how often the autostat skill loads, one fresh conversation per query.

What this script does
    It reads ``triggers.yaml`` beside it, copies the skill into a scratch
    folder as a project skill, then runs each query through ``claude -p`` in
    that folder, in a conversation of its own. It records whether the
    conversation called the ``Skill`` tool with ``autostat``. Tools that write
    or reach the network are disabled, so a query can load the skill but not
    act on anything.

Why it exists
    A skill's description is a guess about how people ask. The Anthropic
    skills guide sets the bar: at least 90 % of the queries that should
    trigger it, none of those that should not. Only a real conversation says
    whether the guess holds.

Usage
    ``python skills/autostat/evals/measure_triggers.py [--out results.json]``

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
ROOT = SKILL.parent.parent


def skills_called(stream: str) -> list[str]:
    """List the skills a headless conversation called.

    Parameters
    ----------
    stream : str
        The ``stream-json`` output of ``claude -p``.

    Returns
    -------
    list of str
        Skill names, in call order.
    """
    names = []
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        for part in event["message"].get("content", []):
            if part.get("type") == "tool_use" and part.get("name") == "Skill":
                names.append(str(part["input"].get("skill")))
    return names


def main() -> int:
    """Run every query and print one line per query, then the two rates.

    Returns
    -------
    int
        0 when the guide's bar is met, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    args = parser.parse_args()
    queries = yaml.safe_load((HERE / "triggers.yaml").read_text(encoding="utf-8"))
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copytree(SKILL, work / ".claude" / "skills" / "autostat")
        shutil.copy2(ROOT / "examples" / "sales.csv", work / "sales.csv")
        for kind in ("should_trigger", "should_not_trigger"):
            for query in queries[kind]:
                query = query.strip("«» ").strip()
                done = subprocess.run(
                    [
                        "claude",
                        "-p",
                        query,
                        "--output-format",
                        "stream-json",
                        "--verbose",
                        "--max-turns",
                        "3",
                        "--disallowedTools",
                        "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Agent",
                    ],
                    cwd=work,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                called = skills_called(done.stdout)
                hit = "autostat" in called
                results.append(
                    {"kind": kind, "query": query, "triggered": hit, "skills": called}
                )
                print(f"{kind:18s} {'yes' if hit else 'no ':3s} {query}", flush=True)
    wanted = [r["triggered"] for r in results if r["kind"] == "should_trigger"]
    unwanted = [r["triggered"] for r in results if r["kind"] == "should_not_trigger"]
    print(
        f"should trigger: {sum(wanted)}/{len(wanted)}; should not: {sum(unwanted)}/{len(unwanted)}"
    )
    if args.out:
        args.out.write_text(
            json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0 if sum(wanted) >= 0.9 * len(wanted) and not any(unwanted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
