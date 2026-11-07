"""The report reads like the original's and its arithmetic holds.

Author
------
Warith Harchaoui <warith@deraison.ai>
"""

from __future__ import annotations

import pytest

from autostat import describe
from autostat.describe import english_length

AIRLINE = [
    "A linearly increasing function",
    "An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude",
    "A smooth function",
    "Uncorrelated noise with linearly increasing standard deviation",
]


@pytest.fixture(scope="module")
def airline_report(airline, airline_final):
    """Describe the published airline model without model checking.

    Returns
    -------
    Report
        The description.
    """
    x, y = airline
    return describe(airline_final, x, y, name="airline", check=False)


def test_the_airline_summaries_match_the_original_word_for_word(airline_report):
    """The four component summaries are the 2014 system's."""
    assert [t[0] for t in airline_report.texts] == AIRLINE


def test_the_components_explain_all_the_variance_together(airline_report):
    """Cumulative R squared reaches 100 per cent with the last component."""
    assert airline_report.stats[-1]["cum_var"] == pytest.approx(100.0, abs=0.1)


def test_the_markdown_report_contains_no_em_dash(airline_report):
    """The rendered report follows the repository's prose rules."""
    assert chr(0x2014) not in airline_report.markdown()


@pytest.mark.parametrize(
    ("years", "text"),
    [
        (2.0, "2.0 years"),
        (0.5, "6.0 months"),
        (1 / 52, "7.0 days"),
        (1 / 365 / 24 * 3, "3.0 hours"),
    ],
)
def test_durations_are_written_in_the_most_readable_unit(years, text):
    """english_length picks the unit the original picked."""
    assert english_length(years, "year") == text
