"""Read one curve with PySR, in a process of its own.

What this script does
    It reads a curve as JSON on standard input, runs PySR on it and writes
    the shortest formula within the tolerance as JSON on standard output.
    :func:`autostat.formula.symbolic_form` runs it as a subprocess.

Why a separate process
    PySR runs on Julia through ``juliacall``; importing ``juliacall``
    after ``torch`` can crash the process; ``autostat`` imports ``torch``.
    This file is run by its path, so it imports neither the package nor
    ``torch``; a crash in Julia costs one formula rather than the report.

What it consumes
    ``{"u": [...], "curve": [...], "tolerance": 0.01}``.

What it produces
    ``{"latex": "...", "gap": 0.003}``, or ``{"latex": null}`` when no
    formula on PySR's complexity front is close enough. Exit status 3 when
    PySR is not installed.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import sys
import tempfile


def main() -> int:
    """Run PySR on the curve read from standard input.

    Returns
    -------
    int
        0 on success, 3 when PySR is not installed.
    """
    request = json.load(sys.stdin)
    try:
        import numpy as np
        import sympy
        from pysr import PySRRegressor
    except ImportError:
        return 3
    u = np.asarray(request["u"], float)[:, None]
    curve = np.asarray(request["curve"], float)
    scale = max(float(np.max(np.abs(curve))), 1e-12)
    answer: dict[str, object] = {"latex": None}
    with tempfile.TemporaryDirectory() as folder:
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "sin", "cos", "tanh"],
            maxsize=25,
            random_state=0,
            deterministic=True,
            parallelism="serial",
            progress=False,
            verbosity=0,
            output_directory=folder,
        )
        model.fit(u, curve)
        # The front runs from the simplest formula to the most accurate one:
        # the first close enough is the shortest honest reading.
        for index in range(len(model.equations_)):
            gap = float(np.max(np.abs(model.predict(u, index=index) - curve))) / scale
            if gap <= request["tolerance"]:
                expression = model.sympy(index=index)
                rounded = expression.xreplace(
                    {n: sympy.Float(n, 4) for n in expression.atoms(sympy.Float)}
                ).subs(sympy.Symbol("x0"), sympy.Symbol("u"))
                answer = {"latex": sympy.latex(rounded), "gap": gap}
                break
    # Julia may have written to standard output: the answer is the last line.
    print("\n" + json.dumps(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
