"""Tests of the structure reader and of the proposals a language model makes.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autostat import Experiment, benchmarks, search
from autostat.context import Context
from autostat.propose import Proposer, parse, prompt
from autostat.search import data_shape


@pytest.mark.parametrize("name", benchmarks.names())
def test_every_published_structure_reads_back_to_itself(name):
    """The reader inverts ``structure_key`` on all the 2014 level winners.

    Parameters
    ----------
    name : str
        One of the thirteen series.
    """
    for model in benchmarks.published(name):
        key = model.kernel.simplified().structure_key()
        assert parse(key).structure_key() == key


@pytest.mark.parametrize("text", ["SE +", "Foo + SE", "CP(SE)", "SE SE", "(SE", ""])
def test_a_malformed_structure_is_refused(text):
    """Text that is not a structure raises rather than guessing.

    Parameters
    ----------
    text : str
        Malformed input.
    """
    with pytest.raises(ValueError):
        parse(text)


def test_the_usual_ways_of_writing_a_product_agree():
    """``x``, ``*`` and the multiplication sign read the same."""
    keys = {
        parse(t).structure_key()
        for t in ("SE x Per", "SE * Per", "SE " + chr(0xD7) + " Per")
    }
    assert keys == {"Per x SE"}


def test_the_prompt_carries_the_context_and_the_best_so_far():
    """The model is told the expected cycle, the event and the current best."""
    x = np.linspace(2000, 2010, 50)
    shape = data_shape(x[:, None], np.sin(x), Experiment())
    context = Context.from_dict(
        {
            "cycles": [{"name": "yearly", "period": "1y"}],
            "events": [{"name": "reform", "at": 2005.0}],
        }
    )
    text = prompt(shape, benchmarks.published("01-airline")[:1], context, 3)
    assert "'yearly' of 1 years" in text
    assert "'reform' happened at 2005.00" in text
    assert "BIC" in text and "Propose 3" in text


def test_a_chatty_answer_still_gives_its_structures():
    """Numbering, bullets and prose lines are tolerated; bad lines recorded."""
    answer = (
        "Here are some:\n1. Per x SE + WN\n- Lin + SE\n* `CP(SE, SE)`\n2. Per + Per"
    )
    proposer = Proposer(ask=lambda _: answer, count=5)
    kernels = proposer([], {"x_min": [0.0], "x_max": [1.0]}, 0)
    assert [k.structure_key() for k in kernels] == [
        "Per x SE + WN",
        "Lin + SE",
        "CP(SE, SE)",
        "Per + Per",
    ]
    assert proposer.history[0]["rejected"] == ["Here are some:"]


def test_a_failing_model_gives_no_proposal_and_no_crash():
    """A language model that errors leaves the search to the grammar."""

    def broken(_: str) -> str:
        """Fail as an unreachable server would.

        Parameters
        ----------
        _ : str
            The prompt, ignored.

        Raises
        ------
        ConnectionError
            Always.
        """
        raise ConnectionError("no server")

    proposer = Proposer(ask=broken)
    assert proposer([], {"x_min": [0.0], "x_max": [1.0]}, 0) == []
    assert "no server" in proposer.history[0]["error"]


def test_a_proposed_structure_is_fitted_beside_the_grammar():
    """A structure the grammar cannot reach in one level is still scored."""
    # Ten years: the search wants five cycles in the span before it keeps
    # a period, so a yearly cycle on four years is out of bounds.
    x = np.linspace(2000, 2010, 80)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.default_rng(0).standard_normal(80)
    proposer = Proposer(ask=lambda _: "Per x SE + Lin + WN")
    exp = Experiment.abcd_2014(max_depth=1, n_jobs=1, n_rand=3, iters=20)
    result = search(x, y, exp, propose=proposer)
    keys = {m.kernel.simplified().structure_key() for m in result.levels[0]}
    assert "Lin + Per x SE + WN" in keys


def test_a_period_written_by_the_model_becomes_a_starting_value():
    """``Per(0.25)`` starts the cycle at a quarter; malformed periods fail."""
    import math

    kernel = parse("SE + Per(0.25)")
    periods = [k.period for k in kernel.operands if hasattr(k, "period")]
    assert periods == [pytest.approx(math.log(0.25))]
    for bad in ("Per(0)", "Per(", "Per(a)", "SE(1)"):
        with pytest.raises(ValueError):
            parse(bad)


def test_a_proposal_with_a_period_is_kept_beside_the_same_structure():
    """A cycle with a starting period is fitted even if the grammar offers ``Per``."""
    x = np.linspace(2000, 2010, 80)
    y = np.sin(2 * np.pi * x / 0.7) + 0.1 * np.random.default_rng(0).standard_normal(80)
    proposer = Proposer(ask=lambda _: "Per(0.7) + WN")
    # The other parameters still start at random: give them a few starts.
    exp = Experiment.abcd_2014(max_depth=1, n_jobs=1, n_rand=3, iters=50)
    result = search(x, y, exp, propose=proposer)
    periods = [
        math.exp(k.period)
        for m in result.levels[0]
        for k in [m.kernel, *getattr(m.kernel, "operands", [])]
        if getattr(k, "period", None) is not None
    ]
    assert any(abs(p - 0.7) < 0.02 for p in periods)
