"""The grammar reaches the structures the original reached.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from autostat import grammar
from autostat import kernels as ff


def test_the_first_2014_level_contains_the_expected_neighbours():
    """White noise expands to its sums, replacements and changepoints."""
    found = {
        k.structure()
        for k in grammar.expand_kernels(1, [ff.NoiseKernel()], "SE,Per,Lin,Const,Noise")
    }
    assert {"WN + SE", "WN + Per", "Lin + WN", "SE", "Per", "CP(WN, WN)"} <= found


def test_an_expansion_contains_no_duplicates():
    """Deduplication works on the canonical form."""
    kernels = grammar.expand_kernels(1, [ff.NoiseKernel()], "SE,Per,Lin,Const,Noise")
    assert len(kernels) == len({repr(k) for k in kernels})


def test_the_2013_grammar_starts_from_the_base_kernels():
    """An empty start under the 2013 rules gives exactly the base kernels."""
    found = {
        k.structure()
        for k in grammar.expand_kernels(
            1, [ff.NoneKernel()], "SE,RQ,Per,Lin,Const", grammar.RULES_2013
        )
    }
    assert found == {"SE", "RQ", "Per", "Lin", "C"}
