"""The stopping study: replaying a rule on a path; a search that obeys it.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest

from autostat import Experiment, benchmarks, search
from autostat.stopping import SearchPath, Step, evidence, findings, search_path


def _path(bics: list[float], structures: list[str] | None = None) -> SearchPath:
    """Build a path from BICs alone.

    Parameters
    ----------
    bics : list of float
        One per level.
    structures : list of str, optional
        One per level; all different when omitted.

    Returns
    -------
    SearchPath
        The path, with a tolerance of 0.1.
    """
    structures = structures or [f"S{i}" for i in range(len(bics))]
    steps = [
        Step(
            level=i + 1,
            bic=b,
            nll=b / 2,
            parameters=i + 2,
            structure=s,
            components=i + 1,
            closed_forms=0,
        )
        for i, (b, s) in enumerate(zip(bics, structures, strict=True))
    ]
    return SearchPath(steps, "test")


def test_patience_replays_where_a_search_would_have_stopped():
    """One stall stops patience 1; patience 2 needs two in a row."""
    path = _path([100.0, 90.0, 90.05, 85.0, 85.0, 85.0])
    assert path.stop_with_patience(1) == (3, path.steps[1])
    assert path.stop_with_patience(2) == (6, path.steps[3])
    assert path.kept.level == 4


def test_a_path_that_never_stalls_does_not_stop():
    """A rule that never triggers says so rather than naming the last level."""
    path = _path([100.0, 90.0, 80.0])
    assert path.stop_with_patience(2) == (None, path.steps[2])


@pytest.mark.parametrize(
    ("difference", "label"),
    [
        (1.0, "barely worth mentioning"),
        (4.0, "positive"),
        (8.0, "strong"),
        (25.0, "very strong"),
    ],
)
def test_bic_differences_read_on_the_kass_raftery_scale(difference, label):
    """The four bands of Kass and Raftery (1995).

    Parameters
    ----------
    difference : float
        A BIC difference.
    label : str
        Its band.
    """
    assert evidence(difference) == label


def test_findings_separate_refits_from_new_structure():
    """Levels that only refit the kept structure are not counted as new structure."""
    path = _path([100.0, 80.0, 79.0, 78.9], ["A", "B", "B", "B"])
    text = " ".join(findings(path, 100))
    assert "first reached at level 2" in text
    assert "The best different structure, A (level 1), is 21.1 above" in text
    assert "very strong evidence" in text


def test_the_published_airline_path_converged():
    """The 2014 airline structure settles at level 7; BIC stops falling at level 9."""
    x, y = benchmarks.load("01-airline")
    path = search_path(benchmarks.published("01-airline"), x, y, "published")
    text = " ".join(findings(path, len(y)))
    assert path.kept.level == 10
    assert "first reached at level 7" in text
    assert "BIC stopped falling at level 9" in text
    assert path.settled.level == 9


@pytest.mark.slow
def test_a_search_with_patience_stops_past_its_best_level():
    """With patience 2 the search runs two levels past its last improvement."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 60)[:, None]
    y = 0.5 * t[:, 0] + rng.normal(0, 0.1, 60)
    exp = Experiment.abcd_2014(
        max_depth=8, n_jobs=1, stopping_criteria=["no_improvement"], patience=2
    )
    result = search(t, y, exp)
    bics = [m.bic for m in result.best_per_level()]
    best = int(np.argmin(bics))
    assert len(bics) < 8
    assert len(bics) - 1 - best >= 2 or len(bics) == 8


def test_path_weights_count_each_structure_once():
    """A refitted structure is one candidate, at its lowest BIC."""
    path = _path([100.0, 90.0, 88.0], ["A", "B", "B"])
    rows = path.plausibility()
    assert [r.structure for r in rows] == ["B", "A"]
    assert rows[0].bic == 88.0
    assert rows[1].weight == pytest.approx(np.exp(-6) / (1 + np.exp(-6)))


def test_the_published_call_centre_path_has_a_close_rival():
    """The report states the weights of the kept structure and of the next."""
    x, y = benchmarks.load("07-call-centre")
    path = search_path(benchmarks.published("07-call-centre"), x, y, "published")
    rows = path.plausibility()
    assert rows[0].weight == pytest.approx(0.713, abs=0.01)
    assert rows[1].weight == pytest.approx(0.281, abs=0.01)
    text = " ".join(findings(path, len(y)))
    assert "posterior weight of 0.713" in text
    assert "barely tell them apart" in text
