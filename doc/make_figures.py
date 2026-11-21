"""Draw the figures of the repository report, in SVG (Markdown) and PDF (LaTeX).

What this script does
    It draws the airline passengers series alone, then the BIC of each
    level of two published 2014 searches: airline passengers, which
    converged; internet traffic, which stalled for two levels before
    finding a much better structure. It then draws the extrapolation
    benchmark (every method's median with its bootstrap interval; the
    published 2014 models against the port, series by series) and two
    studies: the changepoint detector as a proposer of dates; the
    periodogram starts. The figures go to ``doc/figures/``.

    The benchmark figures read ``benchmark/results/summary.parquet`` and
    ``scores.parquet``, which ``benchmark/score.py`` writes; the studies
    read ``doc/research/changepoints/results.json`` and
    ``doc/research/recovery/results*.json``.

Usage
    ``python doc/make_figures.py`` (``doc/build.sh`` runs it).

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from autostat import benchmarks
from autostat.describe import plot_bic, plot_data
from autostat.stopping import search_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
RULE = "the published 2014 search: ten levels, no stopping rule"

#: Benchmark method names as the report writes them.
METHODS = {
    "chronos2": "Chronos-2",
    "abcd_2014_port": "ABCD port, best model",
    "abcd_2014_port_mixture": "ABCD port, BIC mixture",
    "chronos_bolt_base": "Chronos-Bolt",
    "timestral_209m": "Timestral 209M",
    "kumo_lags": "KumoTabular",
    "timesfm_2p5": "TimesFM 2.5",
    "auto_arima": "AutoARIMA",
    "nhits": "N-HiTS",
    "TCI": "TCI (2014)",
    "auto_theta": "AutoTheta",
    "nbeats": "N-BEATS",
    "auto_ets": "AutoETS",
    "lightgbm_lags": "LightGBM",
    "seasonal_naive": "Seasonal naive",
    "ABCD-2014": "ABCD, published 2014",
    "SP": "Spectral mixture (2014)",
    "naive": "Naive",
    "CKS-2013": "CKS, published 2013",
    "Eureqa": "Eureqa (2014)",
    "CP": "Changepoint (2014)",
    "pysr": "PySR",
    "MKL": "MKL (2014)",
    "SE": "SE (2014)",
    "LN": "Linear (2014)",
    "gplearn": "gplearn",
}

#: Benchmark series names as the report writes them.
SERIES = {
    "01-airline": "Airline",
    "02-solar": "Solar",
    "03-mauna2003": "Mauna Loa",
    "beveridge-wheat-price-index-1500": "Wheat",
    "daily-minimum-temperatures-in-me": "Temperature",
    "internet-traffic-data-in-bits-fr-2": "Internet",
    "monthly-average-daily-calls-to-d": "Call centre",
    "monthly-critical-radio-frequenci": "Radio",
    "monthly-production-of-gas-in-aus": "Gas",
    "monthly-production-of-sulphuric-": "Sulphuric",
    "monthly-us-male-1619-years-unemp": "Unemployment",
    "number-of-daily-births-in-quebec": "Births",
    "real-daily-wages-in-pounds-engla": "Wages",
}


def _pdf(svg: Path) -> None:
    """Write the PDF twin of an SVG figure, as vectors when ``rsvg-convert`` exists.

    Parameters
    ----------
    svg : pathlib.Path
        The SVG file; the PDF goes beside it with the same stem.
    """
    pdf = svg.with_suffix(".pdf")
    exe = shutil.which("rsvg-convert")
    if exe:
        subprocess.run([exe, "-f", "pdf", "-o", str(pdf), str(svg)], check=True)
        return
    from sprezzature_figures.exports.pdf import _raster_page

    pdf.write_bytes(_raster_page(svg.read_text(encoding="utf-8")))


def _render(kind: str, rows: list[dict], name: str, **options) -> Path:
    """Draw one sprezzature-figures kind to ``figures/<name>.svg``.

    Parameters
    ----------
    kind : str
        The catalogue kind.
    rows : list of dict
        The data, keyed by the kind's roles.
    name : str
        The file stem.
    **options
        Forwarded to ``make_figure``.

    Returns
    -------
    pathlib.Path
        The SVG written.
    """
    from sprezzature_figures import make_figure

    svg = OUT / f"{name}.svg"
    with contextlib.redirect_stdout(io.StringIO()):
        make_figure(
            kind, rows, out=str(svg), mode="static", embed_fonts=False, **options
        )
    return svg


def _drop_pooled(svg: Path, ticks: tuple[float, ...]) -> None:
    """Take the meta-analysis summary out of a forest plot and set its ticks.

    The forest kind always adds a pooled diamond, an inverse-variance mean
    of the rows. Pooling the medians of different methods means nothing,
    so its row and tooltip are removed. Its log axis picks ticks for odds
    ratios (0.9, 1, 1.1 collide here); they are replaced by ``ticks``,
    placed on the same log scale, read from two of the original ticks.

    Parameters
    ----------
    svg : pathlib.Path
        The forest SVG, edited in place.
    ticks : tuple of float
        The tick values to draw.
    """
    text = svg.read_text(encoding="utf-8")
    pooled = max(int(n) for n in re.findall(r'<g id="row-(\d+)"', text))
    text = re.sub(rf'<g id="row-{pooled}".*?</g>\n?', "", text, count=1, flags=re.S)
    text = re.sub(rf'<g id="rowtip-{pooled}".*?</g>\n?', "", text, count=1, flags=re.S)
    tick = re.compile(
        r'<line x1="([\d.]+)" y1="([\d.]+)" x2="\1" y2="([\d.]+)"[^>]*/>\n'
        r'(<text x="\1" y="[\d.]+"[^>]*>)([\d.]+)</text>\n'
    )
    found = tick.findall(text)
    (x0, _, _, _, v0), (x1, y1, y2, head, v1) = found[0], found[-1]
    per_log = (float(x1) - float(x0)) / (math.log(float(v1)) - math.log(float(v0)))
    drawn = []
    for v in ticks:
        x = float(x0) + per_log * (math.log(v) - math.log(float(v0)))
        label = re.sub(r'x="[\d.]+"', f'x="{x:.1f}"', head, count=1)
        drawn.append(
            f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y2}" stroke="#1D1D1F" '
            f'stroke-width="1"/>\n{label}{v:g}</text>\n'
        )
    first = tick.search(text)
    text = tick.sub("", text)
    text = text[: first.start()] + "".join(drawn) + text[first.start() :]
    svg.write_text(text, encoding="utf-8")


def _colour_rows(svg: Path, colours: dict[str, str], rest: str) -> None:
    """Colour the boxes of a forest plot by row, so the rows compared stand out.

    The forest kind fills every box the same green. Here the rows the report
    discusses take a colour of their own and the others a quiet grey.

    Parameters
    ----------
    svg : pathlib.Path
        The forest SVG, edited in place.
    colours : dict
        Row label to fill colour.
    rest : str
        Fill colour of every other row.
    """
    text = svg.read_text(encoding="utf-8")

    def fill(row: re.Match) -> str:
        label = re.search(r'aria-label="([^"]*)"', row.group(0)).group(1)
        colour = next(
            (c for name, c in colours.items() if label.startswith(name + ":")), rest
        )
        return re.sub(
            r'(<rect class="box[^"]*"[^>]*?fill=")#[0-9A-Fa-f]{6}',
            rf"\g<1>{colour}",
            row.group(0),
        )

    text = re.sub(r'<g id="row-\d+".*?</g>', fill, text, flags=re.S)
    svg.write_text(text, encoding="utf-8")


def benchmark_figures() -> None:
    """Draw the ranking of every method and the 2014 models against the port."""
    results = ROOT / "benchmark" / "results"
    summary = pd.read_parquet(results / "summary.parquet")
    rows = [
        {
            "label": METHODS.get(r.method, r.method),
            "n": 13,
            "or_": round(r.median_standardised_rmse, 2),
            "lo": round(r.median_low, 2),
            "hi": round(r.median_high, 2),
        }
        for r in summary.itertuples()
    ]
    svg = _render(
        "forest",
        rows,
        "bench_ranking",
        title="Twenty-six methods on the thirteen series of 2014",
        subtitle="Median standardised RMSE and its 95 % bootstrap interval. Blue: the port; orange: the published 2013 and 2014 models",
        axis_title="Median standardised RMSE, log scale (lower is better)",
        row_label_header="Method",
        estimate_header="Median (95 %)",
    )
    _drop_pooled(svg, (1, 1.5, 2, 3, 4, 6))
    port, published = "#0088FF", "#FF8D28"
    _colour_rows(
        svg,
        {
            METHODS["abcd_2014_port"]: port,
            METHODS["abcd_2014_port_mixture"]: port,
            METHODS["ABCD-2014"]: published,
            METHODS["CKS-2013"]: published,
        },
        "#C7C7CC",
    )
    _pdf(svg)

    scores = pd.read_parquet(results / "scores.parquet")
    rmse = scores.pivot(index="dataset", columns="method", values="rmse")
    standard = scores.pivot(
        index="dataset", columns="method", values="standardised_rmse"
    )
    rows, highlight = [], {}
    for dataset, name in SERIES.items():
        old, new = (
            standard.loc[dataset, "ABCD-2014"],
            standard.loc[dataset, "abcd_2014_port"],
        )
        rows += [
            {"item": name, "period": "Published 2014", "v": round(old, 2)},
            {"item": name, "period": "Port, 2026", "v": round(new, 2)},
        ]
        gap = rmse.loc[dataset, "abcd_2014_port"] - rmse.loc[dataset, "ABCD-2014"]
        if abs(new - old) >= 0.005:
            highlight[name] = "Green" if gap < 0 else "Red"
    better = sum(c == "Green" for c in highlight.values())
    worse = sum(c == "Red" for c in highlight.values())
    svg = _render(
        "slope",
        rows,
        "bench_slope",
        title=f"The port beats the 2014 models on {better} series of 13",
        subtitle=f"Standardised RMSE per series, lower is better. Green: the port does better; red: worse ({worse}); grey: a tie",
        highlight=highlight,
        width=640,
        height=640,
    )
    _pdf(svg)


def study_figures() -> None:
    """Draw the detector study and the periodogram starts."""
    research = HERE / "research"
    arms = {
        "grammar": "Grammar alone",
        "spectral": "Periodogram starts",
        "breaks": "Detector dates",
    }
    runs = json.loads((research / "changepoints" / "results.json").read_text())
    rows = [
        {"item": f"Seed {r['seed']}", "period": arms[r["arm"]], "v": round(r["bic"], 1)}
        for r in runs
        if r["family"] == "jump"
    ]
    svg = _render(
        "slope",
        rows,
        "breaks_slope",
        title="A detector's dates let the search keep the jump",
        subtitle="Final BIC of five series with a level jump, by what starts the search; lower is better",
        highlight={f"Seed {r['seed']}": "Blue" for r in runs},
        width=760,
        height=500,
    )
    _pdf(svg)

    recovery = research / "recovery"
    counts = {}
    for label, name in (
        ("Fixed centres (2014)", "results.json"),
        ("Periodogram starts", "results-spectral.json"),
    ):
        runs = json.loads((recovery / name).read_text())
        cyclic = [r for r in runs if r["truth_periodic"]]
        counts[label] = {
            "Cycles found": sum(r["found_periodic"] for r in cyclic),
            "Periods within 5 %": sum(r["period_recovered"] for r in cyclic),
            "Cycles invented": sum(
                r["found_periodic"] for r in runs if not r["truth_periodic"]
            ),
        }
    rows = [
        {"item": item, "period": label, "v": v}
        for label, found in counts.items()
        for item, v in found.items()
    ]
    svg = _render(
        "slope",
        rows,
        "recovery_slope",
        title="Periodogram starts find the cycles the 2014 starts missed",
        subtitle="Forty series drawn from the grammar: ten with a cycle, thirty without",
        highlight={item: "Green" for item in counts[label]},
        width=640,
        height=440,
    )
    _pdf(svg)


def main() -> None:
    """Write every figure in both formats."""
    OUT.mkdir(exist_ok=True)
    x, y = benchmarks.load("01-airline")
    xs = np.asarray(x, float).reshape(len(np.ravel(y)), -1)
    for suffix in (".svg", ".pdf"):
        name = plot_data(xs, np.ravel(y), "airline passengers", OUT, "year", suffix)
        (OUT / name).rename(OUT / f"data_airline{suffix}")
    for series, label in (("01-airline", "airline"), ("06-internet", "internet")):
        xs, ys = benchmarks.load(series)
        path = search_path(benchmarks.published(series), xs, ys, RULE)
        for suffix in (".svg", ".pdf"):
            name = plot_bic(path, OUT, suffix)
            (OUT / name).rename(OUT / f"bic_{label}{suffix}")
    benchmark_figures()
    study_figures()


if __name__ == "__main__":
    main()
