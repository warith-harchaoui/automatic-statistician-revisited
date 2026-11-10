"""Entry point for ``python -m autostat``; the interface lives in :mod:`autostat.cli`.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import sys

from .cli import main

sys.exit(main())
