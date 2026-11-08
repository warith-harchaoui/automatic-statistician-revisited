---
name: autostat
description: >-
  Explain a time series in English with the Automatic Statistician (ABCD,
  Lloyd et al. 2014), revisited in Python: search Gaussian process kernel
  structures, split the best one into additive components (trend, cycles,
  noise), describe each in a sentence, check each against its prior, then
  write a report in Markdown compiled to PDF by md2star or in LaTeX
  compiled by latexmk. Use when someone holds a series (a CSV with a date
  column, values in hand, a pandas Series) and asks what it is made of:
  trend, seasonality, period, structure, « décomposer », « expliquer cette
  série ». Not for forecast scores across many series, nor for a chart of
  the raw series alone.
license: MIT
compatibility: >-
  Python 3.11+, pip install automatic-statistician-revisited. PDF output
  needs Pandoc and LibreOffice (Markdown) or a TeX distribution with
  latexmk (LaTeX); without them the report source is still written.
metadata:
  author: Warith Harchaoui
  support: warithmetics@deraison.ai
  version: "0.1.0"
---

# autostat

The routing catalogue is [references/triggers.md](references/triggers.md):
read it when the request is ambiguous.

## Procedure

1. **Find the series.** A file: note its time and value columns
   (`head` it). Values in hand: keep them as a list, with times if given.
2. **Pick the surface.** In a shell with the package installed, the CLI.
   Through an MCP host connected to `autostat-mcp`, the tools. In Python,
   `autostat.analyse`.
3. **Bound the search.** Start at `--max-depth 2` (MCP: `max_depth=2`).
   Each level costs several times the last; say so before going past 4.
4. **Pick the format.** Markdown (default) unless LaTeX is asked for.
   Leave the PDF on unless the user only wants the source.
5. **Run.**
   - CLI: `autostat data.csv --time month --value units --max-depth 2 --out report/`
     (add `--format latex`, `--no-pdf`, `--no-check`, `--json` as needed).
   - MCP: `analyse_series` with `values`, `times`, `name`, `max_depth`,
     `report_format`; then `get_report` with the returned `report_id`.
   - Python: `result, report = analyse(series, name, "report/", Experiment.abcd_2014(max_depth=2), report_format="latex")`.
6. **Report back.** Quote the structure and each component's sentence as
   printed, then give the report path (CLI, Python) or `report_id` (MCP).
   Do not paraphrase the sentences into claims they do not make.

## Recovering

| Symptom | Meaning | Do |
|---|---|---|
| exit status 2, "no column" | wrong `--time` / `--value` | list the columns, ask or pick the obvious one |
| warning "md2pdf not found" / "latexmk not found" | no PDF compiler | the source was written; say which tool to install |
| `invalid_input` from the API | a bound or a name was refused | read `error.message`; it names the field |
| the search runs for long | depth too high | stop it, rerun one level lower, say so |

## Done when

The structure and component sentences are in the answer, the report
exists where you said. Any missing compiler or lowered depth is
stated.
