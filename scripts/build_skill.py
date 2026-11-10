"""Package the autostat skill for upload.

What this script does
    It copies ``TRIGGERS.md`` into the skill's ``references/`` folder, as a
    real file because a zip may not keep a symbolic link, then writes
    ``dist/autostat-skill.zip`` with the ``autostat`` folder at its root.

Usage
    ``python scripts/build_skill.py``

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "autostat"


def main() -> int:
    """Synchronise the routing catalogue and write the archive.

    Returns
    -------
    int
        0 on success.
    """
    shutil.copyfile(ROOT / "TRIGGERS.md", SKILL / "references" / "triggers.md")
    target = ROOT / "dist" / "autostat-skill.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SKILL.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("autostat") / path.relative_to(SKILL))
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
