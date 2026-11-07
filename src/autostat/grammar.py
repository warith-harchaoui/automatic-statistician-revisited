"""The search operators: how one kernel expression becomes its neighbours.

What this module does
    It rewrites a kernel expression into every expression one grammar rule
    away: add a base kernel, multiply by one, replace one, wrap a part in a
    changepoint or a change window, or delete a part.

Why it exists as a separate module
    The grammar decides which structures the search ever sees. Keeping it
    apart from the search loop lets the two published grammars (2013 and
    2014) share one search.

What it consumes
    A kernel expression, the base kernels allowed and the number of input
    dimensions.

What it produces
    The list of distinct canonical neighbour expressions.

Provenance
    A port of ``grammar.py`` from ``github.com/jamesrobertlloyd/gpss-research``
    (2014) with the rule set of ``github.com/jamesrobertlloyd/gp-structure-search``
    (2013). The expansion visits sub-expressions exactly as the original did.
    That includes its treatment of n-ary sums: every non-trivial *subset* of
    the operands is expanded as a unit, so ``A + B + C`` reaches
    ``(A + B) x Per + C`` in one step.

Author
------
Warith Harchaoui <warith@deraison.ai>. The ported design is (c) 2012-2015
James Robert Lloyd, David Duvenaud and Roger Grosse, MIT licence, reproduced
in ``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from . import kernels as ff

#: A rewrite rule ``(lhs, rhs, types)``. ``rhs`` is a Polish-notation tuple
#: over the variables that ``types`` declares.
Rule = tuple[str, Any, dict[str, str]]

#: The rules of every published 2014 configuration (the original's
#: ``MULTI_D_RULES``, verbatim).
RULES: list[Rule] = [
    ("A", ("+", "A", "B"), {"A": "kernel", "B": "base"}),
    ("A", ("*", "A", "B"), {"A": "kernel", "B": "base-not-const"}),
    ("A", ("*-const", "A", "B"), {"A": "kernel", "B": "base-not-const"}),
    ("A", "B", {"A": "kernel", "B": "base"}),
    ("A", ("CP", "d", "A"), {"A": "kernel", "d": "dimension"}),
    ("A", ("CW", "d", "A"), {"A": "kernel", "d": "dimension"}),
    ("A", ("B", "d", "A"), {"A": "kernel", "d": "dimension"}),
    ("A", ("BL", "d", "A"), {"A": "kernel", "d": "dimension"}),
    ("A", ("None",), {"A": "kernel"}),
]

#: The three rules of the 2013 compositional kernel search
#: (``gp-structure-search``'s ``ONE_D_RULES``): add a base kernel, multiply by
#: one, or replace a base kernel by another.
RULES_2013: list[Rule] = [
    ("A", ("+", "A", "B"), {"A": "kernel", "B": "base"}),
    ("A", ("*", "A", "B"), {"A": "kernel", "B": "base"}),
    ("A", "B", {"A": "base", "B": "base"}),
]


@dataclass
class Grammar:
    """The vocabulary of one search: base kernels, dimensions and rules.

    Parameters
    ----------
    ndim : int
        Input dimensions.
    base_kernels : str
        Comma-separated base kernel identifiers.
    rules : list of Rule, optional
        Rewrite rules; :data:`RULES` when omitted.
    """

    ndim: int
    base_kernels: str = "SE"
    rules: list[Rule] | None = None

    def __post_init__(self) -> None:
        """Fall back to the 2014 rules when none are given."""
        if not self.rules:
            self.rules = RULES

    def type_matches(self, candidate: Any, tp: str) -> bool:
        """Tell whether ``candidate`` may stand for a variable of type ``tp``.

        Parameters
        ----------
        candidate : Kernel or int
            Expression or dimension index.
        tp : str
            ``"base"``, ``"kernel"``, ``"base-not-const"`` or ``"dimension"``.

        Returns
        -------
        bool
            True when the binding is allowed.

        Raises
        ------
        ValueError
            On an unknown type.
        """
        if tp == "base":
            return isinstance(candidate, ff.Kernel) and not candidate.is_operator
        if tp == "kernel":
            return isinstance(candidate, ff.Kernel)
        if tp == "base-not-const":
            return (
                isinstance(candidate, ff.Kernel)
                and not candidate.is_operator
                and not isinstance(candidate, ff.ConstKernel)
            )
        if tp == "dimension":
            return isinstance(candidate, int)
        raise ValueError(f"unknown type: {tp}")

    def list_options(self, tp: str) -> list[Any]:
        """List every value a variable of type ``tp`` is bound to.

        Parameters
        ----------
        tp : str
            ``"base"``, ``"base-not-const"`` or ``"dimension"``.

        Returns
        -------
        list
            Base kernels or dimension indices.

        Raises
        ------
        ValueError
            For ``"kernel"``, which is never enumerated, or an unknown type.
        """
        if tp == "base":
            return list(ff.base_kernels(self.ndim, self.base_kernels))
        if tp == "base-not-const":
            return [
                k
                for k in self.list_options("base")
                if not isinstance(k, ff.ConstKernel)
            ]
        if tp == "dimension":
            return list(range(self.ndim))
        raise ValueError(f"cannot enumerate the {tp!r} type")


def _replace_all(expr: Any, mapping: dict[str, Any]) -> Any:
    """Substitute bound values for the variables of a Polish expression.

    Parameters
    ----------
    expr : tuple, str or Kernel
        Polish-notation expression.
    mapping : dict
        Variable name to kernel or dimension index.

    Returns
    -------
    tuple, str, Kernel or int
        The expression with every variable replaced by a fresh copy.
    """
    if isinstance(expr, tuple):
        return tuple(_replace_all(e, mapping) for e in expr)
    if isinstance(expr, str):
        if expr in mapping:
            value = mapping[expr]
            return value if isinstance(value, int) else value.copy()
        return expr
    return expr.copy()


def _polish_to_kernel(expr: Any) -> Any:
    """Build the kernel a substituted Polish expression describes.

    Parameters
    ----------
    expr : tuple, Kernel or int
        Expression whose variables are already bound.

    Returns
    -------
    Kernel or int
        The kernel, or the dimension index for a bare dimension.

    Raises
    ------
    ValueError
        On an unknown operator symbol.
    """
    if not isinstance(expr, tuple):
        return expr.copy() if isinstance(expr, ff.Kernel) else expr
    head = expr[0]
    if head == "+":
        return ff.SumKernel([_polish_to_kernel(e) for e in expr[1:]])
    if head == "*":
        return ff.ProductKernel([_polish_to_kernel(e) for e in expr[1:]])
    if head == "*-const":
        # Multiply by (B + C): the constant lets the product keep A's own
        # contribution, so the move is "add a modulated copy" in one step.
        a, b = (_polish_to_kernel(e) for e in expr[1:])
        return ff.ProductKernel([a, ff.SumKernel([b, ff.ConstKernel()])])
    if head == "None":
        return ff.NoneKernel()
    dim, base = expr[1], _polish_to_kernel(expr[2])
    if head == "CP":
        operands = [base.copy(), base.copy()]
        return ff.ChangePointKernel(dimension=dim, operands=operands)
    if head == "CW":
        operands = [base.copy(), base.copy()]
        return ff.ChangeWindowKernel(dimension=dim, operands=operands)
    if head == "B":
        operands = [ff.ConstKernel(), base.copy()]
        return ff.ChangeWindowKernel(dimension=dim, operands=operands)
    if head == "BL":
        operands = [base.copy(), ff.ConstKernel()]
        return ff.ChangeWindowKernel(dimension=dim, operands=operands)
    raise ValueError(f"unknown operator: {head}")


def expand_single_tree(kernel: ff.Kernel, grammar: Grammar) -> list[ff.Kernel]:
    """Apply every rule at the root of ``kernel``.

    Parameters
    ----------
    kernel : Kernel
        Expression to rewrite.
    grammar : Grammar
        Rules and vocabulary.

    Returns
    -------
    list of Kernel
        One rewrite per rule and per binding of its free variables.
    """
    result = []
    for lhs, rhs, types in grammar.rules or []:
        if not grammar.type_matches(kernel, types[lhs]):
            continue
        free = [v for v in types if v != lhs]
        options = (grammar.list_options(types[v]) for v in free)
        for choice in itertools.product(*options):
            mapping = dict(zip(free, choice, strict=True))
            mapping[lhs] = kernel
            result.append(_polish_to_kernel(_replace_all(rhs, mapping)))
    return result


def expand(kernel: ff.Kernel, grammar: Grammar) -> list[ff.Kernel]:
    """Apply every rule at every sub-expression of ``kernel``.

    Parameters
    ----------
    kernel : Kernel
        Expression to rewrite.
    grammar : Grammar
        Rules and vocabulary.

    Returns
    -------
    list of Kernel
        Neighbours, not yet canonical or deduplicated.
    """
    result = expand_single_tree(kernel, grammar)
    if not kernel.is_operator:
        return result
    if kernel.arity == 2:
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                k = kernel.copy()
                before = [o.copy() for o in kernel.operands[:i]]
                after = [o.copy() for o in kernel.operands[i + 1 :]]
                k.operands = [*before, e, *after]
                result.append(k)
        return result
    # n-ary operators: every proper, non-empty subset of the operands is
    # expanded as a group, the rest left untouched.
    n = len(kernel.operands)
    for subset in itertools.product((0, 1), repeat=n):
        if sum(subset) in (0, n):
            continue
        pairs = list(zip(kernel.operands, subset, strict=True))
        unexpanded = [op.copy() for op, s in pairs if not s]
        chosen = [op.copy() for op, s in pairs if s]
        if len(chosen) > 1:
            group = kernel.copy()
            group.operands = chosen
            expansions = expand_single_tree(group, grammar)
        else:
            expansions = expand(chosen[0], grammar)
        for expanded in expansions:
            k = kernel.copy()
            k.operands = [expanded, *unexpanded]
            result.append(k)
    return result


def expand_kernels(
    ndim: int,
    seeds: list[ff.Kernel],
    base_kernels: str = "SE",
    rules: list[Rule] | None = None,
) -> list[ff.Kernel]:
    """List every distinct canonical neighbour of the seed kernels.

    Parameters
    ----------
    ndim : int
        Input dimensions.
    seeds : list of Kernel
        Kernels to expand.
    base_kernels : str, optional
        Comma-separated base kernel identifiers.
    rules : list of Rule, optional
        Defaults to :data:`RULES`.

    Returns
    -------
    list of Kernel
        Canonical, deduplicated, with deletions to nothing removed.
    """
    grammar = Grammar(ndim, base_kernels, rules)
    out = [k.canonical() for seed in seeds for k in expand(seed, grammar)]
    unique = ff.remove_duplicates(out)
    return [k for k in unique if not isinstance(k, ff.NoneKernel)]
