"""Every Python file and every function carries a numpy-style docstring.

What this module checks
    For every ``.py`` file of the repository (virtual environments, ``ext/``
    and ``build/`` excluded):

    * the module docstring holds a numpy ``Author`` section signed
      ``Warith Harchaoui <warithmetics@deraison.ai>``;
    * every function and method, private, nested and decorated ones
      included, has a docstring;
    * a function with arguments (a leading ``self`` or ``cls`` aside) has a
      ``Parameters`` section naming exactly its arguments;
    * a function that returns a value has a ``Returns`` section, a generator
      a ``Yields`` section;
    * a function with a ``raise`` statement of its own has a ``Raises``
      section.

    A one-line docstring is accepted only for a function with no argument
    and no return value. The test collects every violation before failing,
    so one run lists the whole backlog.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Directory names never scanned: environments, vendored code, build output.
EXCLUDED = {
    ".venv",
    ".venv-bench",
    "ext",
    "build",
    "__pycache__",
    ".git",
    "site-packages",
}
AUTHOR = "Warith Harchaoui <warithmetics@deraison.ai>"


def python_files(root: Path = ROOT) -> list[Path]:
    """Return the repository's own Python files.

    Parameters
    ----------
    root : pathlib.Path, optional
        Directory to scan. Defaults to the repository root.

    Returns
    -------
    list of pathlib.Path
        Every ``.py`` file whose path crosses no excluded directory, sorted.
    """
    return sorted(
        p
        for p in root.rglob("*.py")
        if not EXCLUDED.intersection(p.relative_to(root).parts)
    )


def section(doc: str, name: str) -> str | None:
    """Return the body of one numpy section of a docstring.

    Parameters
    ----------
    doc : str
        A docstring, already dedented by :func:`ast.get_docstring`.
    name : str
        Section title, such as ``"Parameters"``.

    Returns
    -------
    str or None
        The text between the section's underline and the next section title,
        or None when the docstring has no such section.
    """
    match = re.search(rf"^{name}\n-+\n(.*?)(?=^\w[\w ]*\n-+\n|\Z)", doc, re.M | re.S)
    return match.group(1) if match else None


def documented_names(body: str) -> set[str]:
    """Argument names listed in a ``Parameters`` section.

    Parameters
    ----------
    body : str
        The section's text, as :func:`section` returns it.

    Returns
    -------
    set of str
        Every name at the start of an unindented line, before its colon;
        ``a, b : int`` documents both ``a`` and ``b``.
    """
    names: set[str] = set()
    for line in body.splitlines():
        if line and not line[0].isspace():
            head = line.split(" : ")[0].split(":")[0]
            names.update(n.strip().lstrip("*") for n in head.split(",") if n.strip())
    return names


def own_nodes(function: ast.AST) -> list[ast.AST]:
    """Return the nodes of a function body, nested functions and classes excluded.

    Parameters
    ----------
    function : ast.AST
        A function definition.

    Returns
    -------
    list of ast.AST
        Every node reached from the body without entering a nested function,
        lambda or class, whose returns and raises belong to them.
    """
    found, stack = [], list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def function_problems(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """List what the docstring of one function lacks.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        The function to check.

    Returns
    -------
    list of str
        One message per missing element; empty when the docstring is complete.
    """
    doc = ast.get_docstring(node)
    if not doc:
        return ["no docstring"]
    arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    names = [a.arg for a in arguments]
    # ``self`` and ``cls`` are the bound instance or class only in first place.
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    names += [a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None]
    body = own_nodes(node)
    returns = any(isinstance(n, ast.Return) and n.value is not None for n in body)
    yields = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in body)
    raises = any(isinstance(n, ast.Raise) and n.exc is not None for n in body)
    problems = []
    if names:
        documented = section(doc, "Parameters")
        if documented is None:
            problems.append("no Parameters section")
        else:
            missing = set(names) - documented_names(documented)
            extra = documented_names(documented) - set(names)
            if missing:
                problems.append(f"Parameters misses {sorted(missing)}")
            if extra:
                problems.append(f"Parameters names unknown {sorted(extra)}")
    if yields and section(doc, "Yields") is None:
        problems.append("no Yields section")
    elif returns and not yields and section(doc, "Returns") is None:
        problems.append("no Returns section")
    if raises and section(doc, "Raises") is None:
        problems.append("no Raises section")
    return problems


def file_problems(path: Path) -> list[str]:
    """Every docstring violation of one file.

    Parameters
    ----------
    path : pathlib.Path
        A Python file.

    Returns
    -------
    list of str
        Messages of the form ``path:line name: problem``.
    """
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems = []
    module = ast.get_docstring(tree)
    if not module:
        problems.append(f"{shown}:1 module: no docstring")
    else:
        author = section(module, "Author")
        if author is None or AUTHOR not in author:
            problems.append(f"{shown}:1 module: no Author section signed {AUTHOR}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for problem in function_problems(node):
                problems.append(f"{shown}:{node.lineno} {node.name}: {problem}")
    return problems


def test_every_file_and_function_has_a_numpy_docstring():
    """Check that the whole repository passes, listing every violation at once."""
    problems = [p for path in python_files() for p in file_problems(path)]
    assert not problems, f"{len(problems)} docstring problems:\n" + "\n".join(problems)


def test_the_checker_catches_what_it_promises(tmp_path):
    """A planted file with each kind of defect yields each message.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's temporary directory, unique to this test.
    """
    planted = tmp_path / "planted.py"
    planted.write_text(
        '"""No author here."""\n'
        "def bare(x):\n    return x\n"
        "def partial(a, b):\n"
        '    """Adds.\n\n    Parameters\n    ----------\n'
        '    a : int\n        One.\n    """\n'
        "    return a + b\n"
        "def gen():\n"
        '    """Counts."""\n'
        "    yield 1\n"
        "def fails():\n"
        '    """Fails."""\n'
        "    raise ValueError('no')\n",
        encoding="utf-8",
    )
    found = "\n".join(file_problems(planted))
    for expected in (
        "module: no Author section",
        "bare: no docstring",
        "partial: Parameters misses ['b']",
        "partial: no Returns section",
        "gen: no Yields section",
        "fails: no Raises section",
    ):
        assert expected in found, (expected, found)
