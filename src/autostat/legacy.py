"""Read the result files the original Automatic Statistician wrote.

What this module does
    It parses the ``GPModel(...)`` lines of a 2014 ``*_result.txt`` file into
    :class:`autostat.model.GPModel` objects, grouped by search level.

Why it exists as a separate module
    The original wrote those lines with ``repr`` and read them back with
    ``eval``. Evaluating a downloaded file executes whatever it contains, so
    this module parses the same lines with :mod:`ast` and accepts only the
    constructor names and literal arguments the original printed.

What it consumes
    Result files such as ``autostat/reference/01-airline_result.txt``.

What it produces
    Models carrying the published negative log likelihood. Re-scoring them
    with :func:`autostat.model.negative_log_likelihood` at the same parameters
    is how this package is checked against the 2014 numbers.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

from . import kernels as ff
from .model import GPModel

#: Constructor names the original printed, mapped to this package's classes.
LEGACY_KERNELS: dict[str, type[ff.Kernel]] = {
    name: cls for name, cls in ff.KERNEL_CLASSES.items() if name != "NoneKernel"
}


def _literal(node: ast.AST) -> Any:
    """Convert one argument node into a Python value.

    Parameters
    ----------
    node : ast.AST
        A number, a negated number, ``np.Inf``, a list or a nested call.

    Returns
    -------
    object
        The value.

    Raises
    ------
    ValueError
        For any other expression.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    if isinstance(node, ast.Attribute) and node.attr in ("Inf", "inf"):
        return math.inf
    if isinstance(node, ast.Name) and node.id in ("inf", "Inf", "nan"):
        return float(node.id.lower())
    if isinstance(node, ast.List):
        return [_build(e) for e in node.elts]
    if isinstance(node, ast.Call):
        return _build(node)
    raise ValueError(f"unsupported expression in legacy result: {ast.dump(node)}")


def _build(node: ast.AST) -> Any:
    """Convert one constructor call into the object it describes.

    Parameters
    ----------
    node : ast.AST
        A call to a kernel, mean, likelihood or ``GPModel`` constructor.

    Returns
    -------
    object
        A kernel, a ``("mean", value)`` or ``("lik", value)`` tag, or a model.

    Raises
    ------
    ValueError
        For anything that is not a known constructor.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("expected a constructor call")
    if node.args:
        raise ValueError("legacy constructors take keyword arguments only")
    name = node.func.id
    kwargs = {kw.arg: _literal(kw.value) for kw in node.keywords}
    if name in LEGACY_KERNELS:
        cls = LEGACY_KERNELS[name]
        if cls in (ff.SumKernel, ff.ProductKernel):
            return cls(kwargs.get("operands"))
        return cls(**kwargs)
    if name == "MeanZero":
        return ("mean", None)
    if name == "MeanConst":
        return ("mean", kwargs.get("c"))
    if name == "LikGauss":
        return ("lik", kwargs.get("sf", -math.inf))
    if name == "GPModel":
        mean = kwargs.get("mean", ("mean", None))[1]
        lik = kwargs.get("likelihood", ("lik", -math.inf))[1]
        nll = kwargs.get("nll")
        return GPModel(
            kwargs["kernel"],
            mean,
            lik,
            math.nan if nll is None else float(nll),
            int(kwargs.get("ndata") or 0),
        )
    raise ValueError(f"unknown constructor in legacy result: {name}")


def parse_model(line: str) -> GPModel:
    """Parse one ``GPModel(...)`` line.

    Parameters
    ----------
    line : str
        As written by the original ``experiment.py``.

    Returns
    -------
    GPModel
        With the reported ``nll`` and ``ndata``.

    Raises
    ------
    ValueError
        When the line is anything other than a model description.
    SyntaxError
        When the line is not valid Python syntax.
    """
    built = _build(ast.parse(line.strip(), mode="eval").body)
    if not isinstance(built, GPModel):
        raise ValueError("the line does not describe a GPModel")
    return built


def read_results(path: str | Path) -> list[list[GPModel]]:
    """Read every model in a result file, grouped by search level.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``*_result.txt`` file.

    Returns
    -------
    list of list of GPModel
        One list per ``Level`` block, in file order.
    """
    levels: list[list[GPModel]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("%") and "Level" in line:
            levels.append([])
        elif line.startswith("GPModel"):
            if not levels:
                levels.append([])
            levels[-1].append(parse_model(line))
    return levels
