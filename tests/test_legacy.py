"""The legacy reader parses result lines and never executes them.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import pytest

from autostat import legacy


@pytest.mark.parametrize(
    "line",
    [
        "__import__('os').system('true')",
        "GPModel(kernel=open('x'))",
        "GPModel(kernel=NoiseKernel(sf=1.0).__class__)",
    ],
)
def test_a_result_line_with_code_is_refused(line):
    """Anything beyond constructors and literals raises ValueError."""
    with pytest.raises(ValueError):
        legacy.parse_model(line)


def test_a_published_line_parses_to_its_structure():
    """The reader rebuilds the original's kernel."""
    line = "GPModel(mean=MeanZero(), kernel=SumKernel(operands=[NoiseKernel(sf=-1.0), SqExpKernel(dimension=0, lengthscale=0.5, sf=2.0)]), likelihood=LikGauss(sf=-np.Inf), nll=10.0, ndata=5)"
    assert legacy.parse_model(line).kernel.structure() == "WN + SE"
