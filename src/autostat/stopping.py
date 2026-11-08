"""The search path, level by level and what a stopping rule makes of it.

What this module does
    A structure search keeps, at each level, its best expression. This
    module turns those winners into a :class:`SearchPath`: for every level
    its BIC, the fall from the level before, the structure, how many
    additive components it splits into, how many of those have a closed
    form and how many parameters BIC charges for. It then studies the
    stopping question the original configuration never asked: where BIC
    stopped falling, which level a rule stopping after one or two levels
    without improvement would have kept and what the levels searched past
    the best one found.

Why it exists
    The published configuration searches a fixed number of levels and keeps
    the best. Whether those levels were too few, enough or more than needed
    is a fact about each series and the report should show it rather than
    leave it to the reader.

How strong a BIC difference is
    BIC differences are read on the scale of Kass and Raftery (1995), for
    which a difference of BIC approximates twice the log Bayes factor: under
    2 barely worth mentioning, 2 to 6 positive, 6 to 10 strong, over 10
    very strong.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formula import formula
from .model import GPModel
from .search import Experiment


@dataclass
class Step:
    """One level of a search: its winner, described.

    Parameters
    ----------
    level : int
        Level number, from 1.
    bic : float
        The winner's BIC.
    nll : float
        Its negative log likelihood.
    parameters : int
        Parameters BIC charges for.
    structure : str
        The winner's simplified structure.
    components : int
        Additive components it splits into.
    closed_forms : int
        Components with a closed form (polynomial or Fourier series).
    """

    level: int
    bic: float
    nll: float
    parameters: int
    structure: str
    components: int
    closed_forms: int


@dataclass
class SearchPath:
    """A search's winners, level by level and the rule that ended it.

    Parameters
    ----------
    steps : list of Step
        One per level searched.
    rule : str
        How the search was told to stop, in words.
    tolerance : float
        The BIC fall that counts as an improvement.
    """

    steps: list[Step]
    rule: str
    tolerance: float = 0.1

    @property
    def kept(self) -> Step:
        """Return the level whose winner the search keeps: the lowest BIC.

        Returns
        -------
        Step
            The kept level.
        """
        return min(self.steps, key=lambda s: s.bic)

    @property
    def settled(self) -> Step:
        """Return the level where BIC stopped falling: the first within tolerance of its minimum.

        Levels after it lower BIC by less than :attr:`tolerance` in all; they
        are what a search finds by going on past the point where BIC says to
        stop.

        Returns
        -------
        Step
            That level.
        """
        floor = self.kept.bic + self.tolerance
        return next(s for s in self.steps if s.bic <= floor)

    def stop_with_patience(self, patience: int) -> tuple[int | None, Step]:
        """Replay the path under "stop after ``patience`` levels without improvement".

        Parameters
        ----------
        patience : int
            Levels without a fall above :attr:`tolerance` before stopping.

        Returns
        -------
        int or None, Step
            The last level such a search would have run (None when it would
            not have stopped within the path) and the level it would have
            kept.
        """
        best, stalled = float("inf"), 0
        for i, step in enumerate(self.steps):
            stalled = stalled + 1 if step.bic >= best - self.tolerance else 0
            best = min(best, step.bic)
            if stalled >= patience:
                ran = self.steps[: i + 1]
                return step.level, min(ran, key=lambda s: s.bic)
        return None, self.kept


def rule_of(experiment: Experiment) -> str:
    """Say in words how a search was told to stop.

    Parameters
    ----------
    experiment : Experiment
        The search settings.

    Returns
    -------
    str
        E.g. ``"ten levels, no stopping rule"``.
    """
    if "no_improvement" in experiment.stopping_criteria:
        return (
            f"stopping after {experiment.patience} level(s) without a BIC fall above "
            f"{experiment.improvement_tolerance:g}, at most {experiment.max_depth}"
        )
    return f"{experiment.max_depth} levels, no stopping rule"


def evidence(difference: float) -> str:
    """Name the strength of a BIC difference on the Kass and Raftery scale.

    Parameters
    ----------
    difference : float
        A BIC difference, positive for the better model.

    Returns
    -------
    str
        ``"very strong"``, ``"strong"``, ``"positive"`` or
        ``"barely worth mentioning"``.
    """
    if difference > 10:
        return "very strong"
    if difference > 6:
        return "strong"
    if difference > 2:
        return "positive"
    return "barely worth mentioning"


def search_path(
    winners: list[GPModel],
    x: np.ndarray,
    y: np.ndarray,
    rule: str,
    unit: str = "year",
    tolerance: float = 0.1,
) -> SearchPath:
    """Describe each level's winner for the report.

    Parameters
    ----------
    winners : list of GPModel
        The best model of each level, in order.
    x, y : numpy.ndarray
        The data the search ran on.
    rule : str
        How the search was told to stop, in words.
    unit : str, optional
        ``"year"`` or ``""``.
    tolerance : float, optional
        The BIC fall that counts as an improvement.

    Returns
    -------
    SearchPath
        The path.
    """
    # Imported here: describe imports this module for its report section.
    from .describe import components

    xs = np.asarray(x, float).reshape(len(np.ravel(y)), -1)
    ys = np.asarray(y, float).ravel()
    steps = []
    for level, model in enumerate(winners, start=1):
        simple = model.simplified()
        parts = components(simple)
        written = formula(simple, parts, list(range(len(parts))), xs, ys, unit)
        steps.append(
            Step(
                level=level,
                bic=float(model.bic),
                nll=float(model.nll),
                parameters=int(simple.kernel.effective_params),
                structure=simple.kernel.structure(),
                components=len(parts),
                closed_forms=sum(e is not None for e in written.explicit),
            )
        )
    return SearchPath(steps, rule, tolerance)


def findings(path: SearchPath, ndata: int) -> list[str]:
    """State in sentences what the path shows about stopping.

    A level can change the structure or only refit the parameters of the
    structure it inherited; the two are told apart, because a refit says
    nothing about whether more structure is needed.

    Parameters
    ----------
    path : SearchPath
        The path.
    ndata : int
        Number of data points, for the BIC penalty per parameter.

    Returns
    -------
    list of str
        Plain sentences, one per finding; no markup.
    """
    steps, kept, tol = path.steps, path.kept, path.tolerance
    out = [
        f"The search ran {len(steps)} levels ({path.rule}) and keeps the model of "
        f"level {kept.level}, {kept.structure}, BIC {kept.bic:.1f}."
    ]
    first = min(s.level for s in steps if s.structure == kept.structure)
    if first < kept.level:
        refit = steps[first - 1].bic - kept.bic
        out.append(
            f"That structure was first reached at level {first}; the levels after "
            f"it only refitted its parameters, lowering BIC by {refit:.1f} in all."
        )
    others = [s for s in steps if s.structure != kept.structure]
    if others:
        rival = min(others, key=lambda s: s.bic)
        gap = rival.bic - kept.bic
        out.append(
            f"The best different structure, {rival.structure} (level {rival.level}), "
            f"is {gap:.1f} above it: {evidence(gap)} evidence for the kept one on "
            "the Kass and Raftery scale."
        )
    settled = path.settled
    beyond = [s for s in steps if s.level > settled.level]
    last_fall = steps[-2].bic - steps[-1].bic if len(steps) > 1 else 0.0
    if beyond:
        gain = settled.bic - kept.bic
        richer = max(s.components for s in beyond)
        changed = sorted({s.structure for s in beyond} - {settled.structure})
        what = (
            f"tried {len(changed)} other structure(s), with up to {richer} components,"
            if changed
            else "kept its structure"
        )
        out.append(
            f"BIC stopped falling at level {settled.level}. The {len(beyond)} level(s) "
            f"searched past it {what} and lowered BIC by {gain:.2f} in all, less than "
            f"the tolerance of {tol:g}: going on buys nothing the data pay for."
        )
    elif last_fall > tol and steps[-1].structure != steps[-2].structure:
        out.append(
            f"The last level still changed the structure and lowered BIC by "
            f"{last_fall:.1f}: more levels might find a better expression."
        )
    else:
        out.append(
            f"The last level lowered BIC by {last_fall:.1f} through its parameters "
            "only: the structure had settled, its fit had not quite."
        )
    for patience in (1, 2):
        stop, chosen = path.stop_with_patience(patience)
        same = (
            "the same model"
            if chosen.level == kept.level
            else "the same structure, less refitted"
            if chosen.structure == kept.structure
            else f"{chosen.structure} (level {chosen.level}, BIC {chosen.bic - kept.bic:.1f} above)"
        )
        rule = f"A rule stopping after {patience} level(s) without a fall above {tol:g}"
        if stop is None:
            out.append(
                f"{rule} would not have stopped within these {len(steps)} levels."
            )
        else:
            out.append(
                f"{rule} would have ended the search at level {stop} and kept {same}."
            )
    penalty = kept.parameters * np.log(ndata)
    out.append(
        f"The kept expression has {kept.parameters} parameters for {ndata} points: "
        f"its BIC is a fit term 2 NLL = {2 * kept.nll:.1f} plus a complexity term "
        f"p ln n = {penalty:.1f}, each extra parameter costing {np.log(ndata):.2f}; "
        f"it splits into {kept.components} components, {kept.closed_forms} of them "
        "with a closed form."
    )
    return out
