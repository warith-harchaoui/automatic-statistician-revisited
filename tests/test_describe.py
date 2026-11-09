"""The report reads like the original's and its arithmetic holds.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import pytest

from autostat import describe
from autostat.describe import english_length, tex

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

    Parameters
    ----------
    airline : tuple of numpy.ndarray
        Fixture: the airline inputs and passenger counts.
    airline_final : GPModel
        Fixture: the final published airline model.
    """
    x, y = airline
    return describe(airline_final, x, y, name="airline", check=False)


def test_the_airline_summaries_match_the_original_word_for_word(airline_report):
    """The four component summaries are the 2014 system's.

    Parameters
    ----------
    airline_report : Report
        Fixture: the description of the published airline model.
    """
    assert [t[0] for t in airline_report.texts] == AIRLINE


def test_the_components_explain_all_the_variance_together(airline_report):
    """Cumulative R squared reaches 100 per cent with the last component.

    Parameters
    ----------
    airline_report : Report
        Fixture: the description of the published airline model.
    """
    assert airline_report.stats[-1]["cum_var"] == pytest.approx(100.0, abs=0.1)


def test_the_markdown_report_contains_no_em_dash(airline_report):
    """The rendered report follows the repository's prose rules.

    Parameters
    ----------
    airline_report : Report
        Fixture: the description of the published airline model.
    """
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
    """english_length picks the unit the original picked.

    Parameters
    ----------
    years : float
        Length in years.
    text : str
        The expected English rendering.
    """
    assert english_length(years, "year") == text


def test_an_unknown_report_format_is_refused(airline_report, tmp_path):
    """Report.write names the formats it accepts.

    Parameters
    ----------
    airline_report : Report
        Fixture: the description of the published airline model.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    with pytest.raises(ValueError, match="markdown"):
        airline_report.write(tmp_path, "html", pdf=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("report_format", "source", "figure"),
    [("markdown", "report.md", "fit.svg"), ("latex", "report.tex", "fit.pdf")],
)
def test_each_format_writes_its_source_and_figures(
    airline_report, tmp_path, report_format, source, figure
):
    """Markdown comes with SVG figures, LaTeX with PDF figures it includes.

    Parameters
    ----------
    airline_report : Report
        Fixture: the description of the published airline model.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    report_format : str
        The format asked for.
    source : str
        The source file it must write.
    figure : str
        The full-posterior figure it must write.
    """
    written = airline_report.write(tmp_path, report_format, pdf=False)
    assert written == tmp_path / source
    assert (tmp_path / figure).exists()
    assert figure in written.read_text()


def test_latex_escapes_what_latex_reads_as_commands():
    """A dataset name with LaTeX's special characters reaches the page as text."""
    assert tex("50% of A&B_c #1 {x} ~ ^ $") == (
        r"50\% of A\&B\_c \#1 \{x\} \textasciitilde{} \textasciicircum{} \$"
    )


@pytest.mark.parametrize("latex", [False, True])
def test_the_search_path_section_weighs_the_structures(latex):
    """The path section tabulates each structure's posterior weight.

    Parameters
    ----------
    latex : bool
        LaTeX rather than Markdown.
    """
    from types import SimpleNamespace

    from autostat import benchmarks
    from autostat.describe import Report
    from autostat.stopping import search_path

    x, y = benchmarks.load("07-call-centre")
    path = search_path(benchmarks.published("07-call-centre"), x, y, "published")
    report = SimpleNamespace(
        path=path, bic_figure=None, model=SimpleNamespace(ndata=len(y))
    )
    text = "\n".join(Report._path_lines(report, latex))
    assert "Posterior weight of each structure met" in text
    assert "0.713" in text and "0.281" in text
