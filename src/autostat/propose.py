"""Structures proposed from outside the grammar, judged by the same score.

What this module does
    It reads a structure written as text (``Per x SE + Lin + WN``,
    ``CP(SE, SE + Per)``), the inverse of :meth:`Kernel.structure_key`. On
    top of that reader it builds a proposal operator. At every level a
    language model is shown the series' summary, the business context and
    the best structures so far; it answers with structures to try. The
    search fits them beside the grammar's neighbours and BIC alone decides
    what is kept.

Why it exists as a separate module
    Large Language Bayes (Domke, NeurIPS 2025), CAKE (NeurIPS 2025) and
    BoxLM (ICML 2024) made "a language model proposes, a Bayesian score
    decides" the expected design of model discovery. None of them feeds
    typed business knowledge (expected cycles, dated events) into a kernel
    search; :class:`Proposer` does, from a :class:`~autostat.context.Context`.
    The grammar stays complete without it: proposals only add candidates.

What it consumes
    Text from any model behind a callable ``ask(prompt) -> str``;
    :func:`ollama` gives one for a local Ollama server, so nothing leaves the
    machine.

What it produces
    :func:`parse` gives a canonical kernel; a :class:`Proposer` is passed to
    :func:`autostat.search.search` as ``propose`` and keeps a record of what
    it asked and what it parsed.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import logging
import math
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import kernels as ff

if TYPE_CHECKING:
    from .context import Context
    from .model import GPModel

LOGGER = logging.getLogger("autostat.propose")

#: Names a structure may use, mapped to the base kernel they stand for.
NAMES: dict[str, str] = {
    "SE": "SE",
    "Per": "Per",
    "Lin": "Lin",
    "RQ": "RQ",
    "GE": "GE",
    "C": "Const",
    "Const": "Const",
    "WN": "Noise",
    "Noise": "Noise",
}

#: The multiplication sign a language model may write for a product.
_TIMES = "\u00d7"

_TOKEN = re.compile(
    r"\s*(CP|CW|[A-Za-z]+|[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|[()+,*\u00d7])"
)


def _tokens(text: str) -> list[str]:
    """Cut a structure into names, operators and brackets.

    Parameters
    ----------
    text : str
        E.g. ``"Per x SE + WN"``.

    Returns
    -------
    list of str
        ``x``, ``*`` and the multiplication sign all become ``"x"``.

    Raises
    ------
    ValueError
        On a character no structure uses.
    """
    out, at = [], 0
    text = text.strip()
    while at < len(text):
        match = _TOKEN.match(text, at)
        if match is None:
            raise ValueError(f"unexpected {text[at:]!r} in {text!r}")
        token = match.group(1)
        out.append("x" if token in ("*", _TIMES) else token)
        at = match.end()
    return out


def parse(text: str, dimension: int = 0) -> ff.Kernel:
    """Read a structure written as text into an unfitted kernel.

    The grammar: a sum of products of atoms, an atom being a base kernel
    name (``SE``, ``Per``, ``Lin``, ``RQ``, ``GE``, ``C``, ``WN``), a
    bracketed structure or a switch ``CP(a, b)`` / ``CW(a, b)``. Products
    bind tighter than sums. ``Per(0.25)`` is a cycle that starts its
    optimisation at a period of 0.25 time units: language models told the
    business's expected period write it so. Told of a dated event, they
    write ``CP(a, b, 2014.5)``, a switch starting at that date, or
    ``CW(a, b, 2010, 2012)``, a window starting between those dates.

    Parameters
    ----------
    text : str
        E.g. ``"CW(WN + SE, C) + Per x SE"``.
    dimension : int, optional
        Input dimension of every kernel.

    Returns
    -------
    Kernel
        Canonical, with every parameter unset.

    Raises
    ------
    ValueError
        When the text is not a structure.
    """
    tokens = _tokens(text)
    at = 0

    def peek() -> str | None:
        """Return the next token without taking it.

        Returns
        -------
        str or None
            ``None`` at the end.
        """
        return tokens[at] if at < len(tokens) else None

    def take(expected: str | None = None) -> str:
        """Take the next token, checking it when one is expected.

        Parameters
        ----------
        expected : str, optional
            The token required here.

        Returns
        -------
        str
            The token.

        Raises
        ------
        ValueError
            At the end, or on a token other than the one expected.
        """
        nonlocal at
        token = peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"expected {expected or 'more'} in {text!r}")
        at += 1
        return token

    def number() -> float:
        """Take a number.

        Returns
        -------
        float
            Its value.

        Raises
        ------
        ValueError
            When the next token is not a number.
        """
        token = take()
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"expected a number, not {token!r}, in {text!r}") from None

    def total() -> ff.Kernel:
        """Read a sum.

        Returns
        -------
        Kernel
            One product, or their sum.
        """
        terms = [product()]
        while peek() == "+":
            take()
            terms.append(product())
        return terms[0] if len(terms) == 1 else ff.SumKernel(terms)

    def product() -> ff.Kernel:
        """Read a product.

        Returns
        -------
        Kernel
            One atom, or their product.
        """
        factors = [atom()]
        while peek() == "x":
            take()
            factors.append(atom())
        return factors[0] if len(factors) == 1 else ff.ProductKernel(factors)

    def atom() -> ff.Kernel:
        """Read a name, a bracketed structure or a switch.

        Returns
        -------
        Kernel
            The atom.

        Raises
        ------
        ValueError
            On an unknown name.
        """
        token = take()
        if token == "(":
            inner = total()
            take(")")
            return inner
        if token in ("CP", "CW"):
            take("(")
            first = total()
            take(",")
            second = total()
            dates = []
            while peek() == ",":
                take()
                dates.append(number())
            take(")")
            if token == "CP":
                if len(dates) > 1:
                    raise ValueError(f"a changepoint takes one date in {text!r}")
                location = dates[0] if dates else None
                return ff.ChangePointKernel(
                    dimension, location=location, operands=[first, second]
                )
            if len(dates) not in (0, 2):
                raise ValueError(f"a window takes two dates in {text!r}")
            location = width = None
            if dates:
                start, end = dates
                if not end > start:
                    raise ValueError(f"a window must end after it starts in {text!r}")
                location, width = (start + end) / 2, math.log(end - start)
            return ff.ChangeWindowKernel(
                dimension, location=location, width=width, operands=[first, second]
            )
        if token not in NAMES:
            raise ValueError(f"unknown kernel {token!r} in {text!r}")
        cls = ff.BASE_KERNELS[NAMES[token]]
        if token == "Per" and peek() == "(":
            take("(")
            period = number()
            take(")")
            if not period > 0:
                raise ValueError(f"a period must be positive in {text!r}")
            return cls(dimension=dimension, period=math.log(period))
        return cls() if cls.is_thunk else cls(dimension=dimension)

    kernel = total()
    if peek() is not None:
        raise ValueError(f"unexpected {peek()!r} in {text!r}")
    return kernel.canonical()


def prompt(
    shape: dict[str, Any],
    models: list[GPModel],
    context: Context | None,
    count: int,
) -> str:
    """Write what the language model is asked at one level.

    Parameters
    ----------
    shape : dict
        Data summary from :func:`autostat.search.data_shape`.
    models : list of GPModel
        Best models so far, best first; empty before the first level.
    context : Context, optional
        What the business knows.
    count : int
        Structures asked for.

    Returns
    -------
    str
        The prompt.
    """
    lines = [
        "You choose Gaussian process kernel structures for one time series.",
        "Write structures with these base kernels: SE (smooth), Per (periodic),",
        "Lin (linear trend), RQ (multi-scale smooth), C (constant), WN (white",
        "noise); ' x ' multiplies, ' + ' adds, brackets group, CP(a, b) switches",
        "from a to b at one date and CW(a, b) is b inside a window and a outside.",
        "Per(p) starts a cycle at a period of p years; CP(a, b, t) starts the",
        "switch at date t; CW(a, b, s, e) starts the window from s to e.",
        "",
        f"The series spans {shape['x_min'][0]:.2f} to {shape['x_max'][0]:.2f}"
        " (time in years).",
    ]
    if context is not None:
        for cycle in context.cycles:
            lines.append(
                f"The business expects a cycle '{cycle.name}' of {cycle.period:.4g} years."
            )
        for event in context.events:
            lines.append(f"An event '{event.name}' happened at {event.at:.2f}.")
        if context.positive:
            lines.append("The values are always positive.")
    if models:
        lines.append("Best structures so far, with their BIC (lower is better):")
        lines += [f"  {m.kernel.structure_key()}  BIC {m.bic:.1f}" for m in models]
    lines += [
        "",
        f"Propose {count} different structures likely to have a lower BIC.",
        "Answer with one structure per line and nothing else.",
    ]
    return "\n".join(lines)


@dataclass
class Proposer:
    """Ask a language model for structures at every level of a search.

    Parameters
    ----------
    ask : callable
        Takes a prompt, returns the model's text.
    context : Context, optional
        What the business knows, written into every prompt.
    count : int, optional
        Structures asked for per level.
    history : list of dict
        Filled as the search runs: the level, the answer and what parsed.
    """

    ask: Callable[[str], str]
    context: Context | None = None
    count: int = 5
    history: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, models: list[GPModel], shape: dict[str, Any], depth: int
    ) -> list[ff.Kernel]:
        """Propose structures for one level.

        A line that does not parse is recorded and skipped; a failure of the
        model itself gives no proposal rather than stopping the search.

        Parameters
        ----------
        models : list of GPModel
            Best models so far, best first.
        shape : dict
            Data summary.
        depth : int
            The level about to be fitted.

        Returns
        -------
        list of Kernel
            Distinct parsed structures, unfitted.
        """
        try:
            answer = self.ask(prompt(shape, models, self.context, self.count))
        except Exception as error:
            self.history.append({"level": depth, "error": str(error)[:300]})
            LOGGER.warning("level %d: the language model failed: %s", depth, error)
            return []
        kernels, rejected = [], []
        for line in answer.splitlines():
            line = line.strip().lstrip("-*0123456789.) ").strip().strip("`")
            if not line:
                continue
            try:
                kernels.append(parse(line))
            except ValueError:
                rejected.append(line)
        kernels = ff.remove_duplicates(kernels)[: self.count]
        LOGGER.info(
            "level %d: the language model proposed %s",
            depth,
            ", ".join(k.structure_key() for k in kernels) or "nothing usable",
        )
        self.history.append(
            {
                "level": depth,
                "proposed": [k.structure_key() for k in kernels],
                "rejected": rejected,
            }
        )
        return kernels


def ollama(
    model: str = "gemma3:12b",
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.7,
    timeout: float = 600.0,
) -> Callable[[str], str]:
    """Ask a local Ollama server.

    Parameters
    ----------
    model : str, optional
        An installed Ollama model.
    host : str, optional
        The server, local by default.
    temperature : float, optional
        Sampling temperature: proposals should vary between levels.
    timeout : float, optional
        Seconds before a request is abandoned.

    Returns
    -------
    callable
        ``ask(prompt) -> str``.
    """

    def ask(text: str) -> str:
        """Send one prompt and return the whole answer.

        Parameters
        ----------
        text : str
            The prompt.

        Returns
        -------
        str
            The model's text.
        """
        body = json.dumps(
            {
                "model": model,
                "prompt": text,
                "stream": False,
                "options": {"temperature": temperature},
            }
        ).encode()
        request = urllib.request.Request(
            f"{host}/api/generate", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["response"]

    return ask
