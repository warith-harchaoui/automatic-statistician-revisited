"""English reports and model checks for an AutoGP posterior, with the 2014 writer.

What this module does
    It turns each structure of a fitted :class:`~autostat.smc.AutoGP`
    population into the kernel objects of :mod:`autostat.kernels`, on the
    data's own time and value scales, then hands them to the 2014 report
    writer (:func:`autostat.describe.describe`): additive components ordered
    by how much each reduces extrapolation error, an English description of
    each, figures and the model-checking p-values (quantile-quantile,
    autocorrelation, periodogram, maximum mean discrepancy). A summary page
    lists the posterior's structures with their weights and links to one
    report per structure.

Why the translation is exact
    Each AutoGP base kernel has an exact counterpart, so the covariance the
    report decomposes is the one the sampler scored:

    * ``Lin(c, b, a)``, ``b + a (t - c)(s - c)``, becomes ``Const + Lin``;
    * AutoGP's periodic kernel keeps its mean, the 2014 one removes it, so
      ``Per`` becomes the 2014 ``Per`` plus a constant of variance
      ``a * i0e(1 / l**2)``;
    * the changepoint's ``tanh`` switch is the 2014 one with steepness
      ``1 / scale``;
    * the gamma-exponential kernel is :class:`autostat.kernels.GammaExpKernel`;
    * the observation noise becomes a white-noise component, as in 2014.

    ``tests/test_smc_report.py`` checks the covariances and the likelihood.

What it produces
    :class:`PosteriorReport`; with ``out_dir``, a summary page there and one
    folder per structure, in Markdown (compiled by md2star) or LaTeX
    (compiled by ``latexmk``), as :meth:`autostat.describe.Report.write`.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import special

from .. import kernels as ff
from ..describe import (
    Report,
    compile_latex,
    compile_markdown,
    describe,
    plot_data,
    tex,
)
from ..model import GPModel, negative_log_likelihood
from .inference import AutoGP, Particle
from .tree import (
    CHANGEPOINT,
    CHANGEPOINT_SCALE,
    CONSTANT,
    FIELDS,
    GAMMA_EXPONENTIAL,
    LINEAR,
    PERIODIC,
    PLUS,
    SQUARED_EXPONENTIAL,
    TIMES,
    Config,
    Scaling,
    Tree,
    children,
    noise_variance,
)


def _params(tree: Tree, config: Config, idx: int) -> dict[str, float]:
    """Read one node's parameters on their natural scale.

    Parameters
    ----------
    tree : Tree
        The structure.
    config : Config
        Holds the transforms from the sampler's unconstrained values.
    idx : int
        The node's index in ``tree``.

    Returns
    -------
    dict of str to float
        Parameter values by field name.
    """
    node = tree[idx]
    return {
        name: float(config.transform(name, z))
        for name, z in zip(FIELDS[node.kind], node.z, strict=True)
    }


def _log_sf(variance: float) -> float:
    """Turn a variance into the log standard deviation the 2014 kernels store.

    Parameters
    ----------
    variance : float
        A positive variance.

    Returns
    -------
    float
        ``log(sqrt(variance))``.
    """
    return 0.5 * math.log(variance)


def to_kernel(
    tree: Tree,
    config: Config,
    scaling: Scaling,
    idx: int = 1,
    variance: float | None = None,
) -> ff.Kernel:
    """The 2014 kernel equal to an AutoGP structure, on the original scales.

    Parameters
    ----------
    tree : Tree
        Structure and parameters.
    config : Config
        The prior, for the parameter transforms.
    scaling : Scaling
        The affine maps the sampler applied to times and values.
    idx : int, optional
        Root of the subtree to translate.
    variance : float, optional
        Factor carried down to one leaf of each product, ``1 / y_slope**2``
        at the root: it undoes the value scaling once per product, as
        ``GP.rescale`` does.

    Returns
    -------
    Kernel
        Same covariance, in original time and value units.

    Raises
    ------
    ValueError
        When the tree holds a node of an unknown kind.
    """
    if variance is None:
        variance = 1.0 / scaling.y_slope**2
    a, b = scaling.t_slope, scaling.t_intercept
    node = tree[idx]
    p = _params(tree, config, idx)
    if node.kind == SQUARED_EXPONENTIAL:
        return ff.SqExpKernel(
            dimension=0,
            lengthscale=math.log(p["lengthscale"] / a),
            sf=_log_sf(p["amplitude"] * variance),
        )
    if node.kind == GAMMA_EXPONENTIAL:
        return ff.GammaExpKernel(
            dimension=0,
            lengthscale=math.log(p["lengthscale"] / a),
            gamma=p["gamma"],
            sf=_log_sf(p["amplitude"] * variance),
        )
    if node.kind == PERIODIC:
        b0 = float(special.i0e(1.0 / p["lengthscale"] ** 2))
        periodic = ff.PeriodicKernel(
            dimension=0,
            lengthscale=math.log(p["lengthscale"]),
            period=math.log(p["period"] / a),
            sf=_log_sf(p["amplitude"] * variance * (1 - b0)),
        )
        return ff.SumKernel(
            [periodic, ff.ConstKernel(sf=_log_sf(p["amplitude"] * variance * b0))]
        )
    if node.kind == LINEAR:
        return ff.SumKernel(
            [
                ff.ConstKernel(sf=_log_sf(p["bias"] * variance)),
                ff.LinearKernel(
                    dimension=0,
                    sf=_log_sf(p["amplitude"] * variance * a**2),
                    location=(p["intercept"] - b) / a,
                ),
            ]
        )
    if node.kind == CONSTANT:
        return ff.ConstKernel(sf=_log_sf(p["value"] * variance))
    left, right = children(idx)
    if node.kind == PLUS:
        return ff.SumKernel(
            [
                to_kernel(tree, config, scaling, left, variance),
                to_kernel(tree, config, scaling, right, variance),
            ]
        )
    if node.kind == TIMES:
        return ff.ProductKernel(
            [
                to_kernel(tree, config, scaling, left, variance),
                to_kernel(tree, config, scaling, right, 1.0),
            ]
        )
    if node.kind == CHANGEPOINT:
        return ff.ChangePointKernel(
            dimension=0,
            location=(p["location"] - b) / a,
            steepness=math.log(a / CHANGEPOINT_SCALE),
            operands=[
                to_kernel(tree, config, scaling, left, variance),
                to_kernel(tree, config, scaling, right, variance),
            ],
        )
    raise ValueError(f"unknown node type {node.kind}")


def to_model(
    particle: Particle, config: Config, scaling: Scaling, x: np.ndarray, y: np.ndarray
) -> GPModel:
    """The 2014 model equal to one particle, scored on the data.

    Parameters
    ----------
    particle : Particle
        Structure, parameters and noise.
    config : Config
        The prior.
    scaling : Scaling
        The sampler's affine maps.
    x, y : numpy.ndarray
        The data, original units.

    Returns
    -------
    GPModel
        Kernel plus white noise, constant mean at the data's mean, its
        negative log likelihood set.
    """
    noise = float(noise_variance(config, particle.z_noise)) / scaling.y_slope**2
    kernel = ff.SumKernel(
        [
            to_kernel(particle.tree, config, scaling),
            ff.NoiseKernel(sf=_log_sf(noise)),
        ]
    )
    model = GPModel(kernel, mean_const=-scaling.y_intercept / scaling.y_slope)
    xs = np.asarray(x, dtype=float).reshape(-1, 1)
    with torch.no_grad():
        nll = negative_log_likelihood(
            model,
            torch.as_tensor(xs),
            torch.as_tensor(np.asarray(y, dtype=float)),
            torch.as_tensor(model.theta_vector()),
        )
    model.nll = float(nll)
    model.ndata = len(xs)
    return model


@dataclass
class PosteriorReport:
    """The posterior's structures, their weights and one report per structure.

    Parameters
    ----------
    name : str
        Dataset name.
    structures : list of tuple
        ``(weight, AutoGP formula, 2014 structure)`` for every distinct
        structure, heaviest first.
    reports : list of Report
        One per described structure, in the same order.
    folders : list of str
        Where each report was written, relative to the summary.
    data_figure : str, optional
        The data alone, drawn when the summary is written.
    """

    name: str
    structures: list[tuple[float, str, str]]
    reports: list[Report] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    data_figure: str | None = None

    def markdown(self) -> str:
        """Render the summary page.

        Returns
        -------
        str
            Markdown.
        """
        lines = [
            f"# An automatic report for the dataset: {self.name}",
            "",
            "*Produced by `autostat.smc` (automatic-statistician-revisited), a "
            "Python port of AutoGP (Saad et al., ICML 2023), with the report "
            "writer and model checks of the Automatic Statistician (Lloyd et "
            "al., AAAI 2014; Lloyd and Ghahramani, NIPS 2015).*",
            "",
            *([f"![The data]({self.data_figure})", ""] if self.data_figure else []),
            "## Posterior over structures",
            "",
            "AutoGP keeps no single best structure and runs no BIC search: its "
            "stopping point is the end of the data schedule; the weights "
            "below replace the comparison of BIC between levels. Each "
            "structure's own report gives its formula.",
            "",
            "Sequential Monte Carlo kept a weighted population of structures. "
            "Forecasts average them by weight. Each structure below is "
            "described on its own, with the parameters of its heaviest "
            "particle.",
            "",
            "| # | weight | AutoGP structure | as 2014 kernels | report |",
            "|---|---|---|---|---|",
        ]
        for i, (weight, formula, structure) in enumerate(self.structures):
            link = (
                f"[{self.folders[i]}]({self.folders[i]}/report.md)"
                if i < len(self.folders)
                else ""
            )
            lines.append(
                f"| {i + 1} | {weight:.3f} | `{formula}` | `{structure}` | {link} |"
            )
        for i, report in enumerate(self.reports):
            lines += [
                "",
                f"## Structure {i + 1} (weight {self.structures[i][0]:.3f})",
                "",
            ]
            lines += [f"{j + 1}. {t[0]}." for j, t in enumerate(report.texts)]
        return "\n".join(lines) + "\n"

    def latex(self) -> str:
        """Render the summary page as a LaTeX document.

        Returns
        -------
        str
            LaTeX; each structure's report is linked by its folder.
        """
        lines = [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}",
            r"\usepackage[margin=2.5cm]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage[hidelinks]{hyperref}",
            "",
            rf"\title{{An automatic report for the dataset: {tex(self.name)}}}",
            r"\author{\texttt{autostat.smc} (automatic-statistician-revisited)}",
            r"\date{}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\noindent\emph{Produced by \texttt{autostat.smc}, a Python port of "
            r"AutoGP (Saad et al., ICML 2023), with the report writer and model "
            r"checks of the Automatic Statistician (Lloyd et al., AAAI 2014; Lloyd "
            r"and Ghahramani, NIPS 2015).}",
            "",
            *(
                [
                    r"\begin{center}",
                    rf"\includegraphics[width=\linewidth]{{{self.data_figure}}}",
                    r"\end{center}",
                ]
                if self.data_figure
                else []
            ),
            r"\section*{Posterior over structures}",
            "",
            "AutoGP keeps no single best structure and runs no BIC search: its "
            "stopping point is the end of the data schedule; the weights "
            "below replace the comparison of BIC between levels. Each "
            "structure's own report gives its formula.",
            "",
            "Sequential Monte Carlo kept a weighted population of structures. "
            "Forecasts average them by weight. Each structure below is described "
            "on its own, with the parameters of its heaviest particle.",
            "",
            r"\begin{center}",
            r"\begin{tabular}{rrll}",
            r"\toprule",
            r"\# & weight & as 2014 kernels & report \\",
            r"\midrule",
        ]
        for i, (weight, _, structure) in enumerate(self.structures):
            link = (
                rf"\href{{{self.folders[i]}/report.pdf}}"
                rf"{{\texttt{{{tex(self.folders[i])}}}}}"
                if i < len(self.folders)
                else ""
            )
            lines.append(
                rf"{i + 1} & {weight:.3f} & \texttt{{{tex(structure)}}} & {link} \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
        for i, report in enumerate(self.reports):
            lines += [
                "",
                rf"\section*{{Structure {i + 1} (weight {self.structures[i][0]:.3f})}}",
                "",
                rf"AutoGP formula: \texttt{{{tex(self.structures[i][1])}}}.",
                r"\begin{enumerate}",
            ]
            lines += [rf"  \item {tex(t[0])}." for t in report.texts]
            lines.append(r"\end{enumerate}")
        lines += ["", r"\end{document}", ""]
        return "\n".join(lines)


def describe_posterior(
    model: AutoGP,
    name: str = "data",
    unit: str = "year",
    out_dir: str | Path | None = None,
    top: int = 3,
    mass: float = 0.95,
    check: bool = True,
    seed: int = 0,
    report_format: str = "markdown",
    pdf: bool = True,
) -> PosteriorReport:
    """Describe and check the heaviest structures of a fitted population.

    Parameters
    ----------
    model : AutoGP
        Fitted.
    name : str, optional
        Dataset name.
    unit : str, optional
        ``"year"`` or ``""``, as :func:`autostat.describe.describe`.
    out_dir : str or pathlib.Path, optional
        Where to write the summary page and one folder per structure.
    top : int, optional
        Describe at most this many structures.
    mass : float, optional
        Stop once the described structures hold this much weight.
    check : bool, optional
        Compute the model-checking p-values.
    seed : int, optional
        Seeds the checks.
    report_format : {"markdown", "latex"}, optional
        See :meth:`autostat.describe.Report.write`.
    pdf : bool, optional
        Compile every page to PDF.

    Returns
    -------
    PosteriorReport
        Structures, weights and reports.
    """
    groups: dict[str, list[tuple[float, Particle]]] = {}
    for weight, particle in zip(model.particle_weights(), model.particles, strict=True):
        groups.setdefault(particle.tree.formula(), []).append((float(weight), particle))
    ranked = sorted(groups.items(), key=lambda item: -sum(w for w, _ in item[1]))
    structures, chosen = [], []
    for formula, members in ranked:
        weight = sum(w for w, _ in members)
        heaviest = max(members, key=lambda member: member[0])[1]
        gp = to_model(heaviest, model.config, model.scaling, model.t, model.y)
        structures.append((weight, formula, gp.kernel.simplified().structure()))
        chosen.append(gp)
    out = PosteriorReport(name, structures)
    held = 0.0
    for i, gp in enumerate(chosen[:top]):
        if held >= mass:
            break
        held += structures[i][0]
        folder = f"structure_{i + 1}"
        report = describe(
            gp,
            model.t,
            model.y,
            name=f"{name}, structure {i + 1} of the posterior",
            unit=unit,
            out_dir=None if out_dir is None else Path(out_dir) / folder,
            check=check,
            seed=seed,
            selected_by=(f"With posterior weight {structures[i][0]:.3f}, AutoGP kept"),
            report_format=report_format,
            pdf=pdf,
        )
        out.reports.append(report)
        out.folders.append(folder)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        latex = report_format == "latex"
        out.data_figure = plot_data(
            np.asarray(model.t, float).reshape(-1, 1),
            np.asarray(model.y, float).ravel(),
            name,
            Path(out_dir),
            unit,
            ".pdf" if latex else ".svg",
        )
        source = Path(out_dir) / ("report.tex" if latex else "report.md")
        source.write_text(out.latex() if latex else out.markdown())
        if pdf:
            (compile_latex if latex else compile_markdown)(source)
    return out
