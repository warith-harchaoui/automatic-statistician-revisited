---
name: autostat
description: >-
  Explain a time series and find the formula behind it: a symbolic regression
  over Gaussian process kernels (the Automatic Statistician, ABCD). Splits a
  noisy curve into trend, cycles and noise, states each in a sentence and as
  a formula with fitted values, shows where the search should stop, writes a
  PDF report in Markdown or LaTeX. Use for a CSV or Parquet file with a date
  column, a list of values or a pandas Series when asked for "what is in this
  series", "find the trend and seasonality", "what is the period", "fit a
  formula to this noisy curve", "decompose this signal", « décompose cette
  série », « quelle formule pour cette courbe ». Not for scoring forecasts
  across many series, tabular regression on several inputs or a plain chart
  of the raw data.
license: MIT
compatibility: >-
  Python 3.11+ with the autostat package. PDF output needs Pandoc and
  LibreOffice (Markdown) or latexmk (LaTeX). Formulas for smooth components
  need the optional PySR extra, which installs Julia.
metadata:
  author: Warith Harchaoui
  support: warithmetics@deraison.ai
  version: "0.2.0"
  mcp-server: autostat
---

# autostat

## Critical

- Quote the structure, the sentences and the formula as printed. Never turn
  a covariance into a closed form the report does not give.
- A search of fixed depth can stop too early. Say which depth ran.
- Consult `references/triggers.md` when the request is ambiguous.

## Instructions

### Step 1: Find the series

A file: run `head` on it and note the time and value columns. Values in
hand: keep them as a list, with times if given.

### Step 2: Choose the surface

| You have | Use |
|---|---|
| a shell with the package | `autostat` |
| the `autostat` MCP server connected | `analyse_series`, then `get_report` |
| Python code around the data | `autostat.analyse` |

### Step 3: Choose the depth

Each level costs several times the last.

| Series | First run | Why |
|---|---|---|
| up to about 100 points | `--max-depth 20 --patience 2` | minutes: runs until BIC stops falling, then two more levels |
| more points | `--max-depth 3`, then offer the search above | the same search on 144 points took four hours |

State the cost before starting anything over four levels.

### Step 4: Run

```bash
autostat data.csv --time month --value units --max-depth 20 --patience 2 --out report/
```

Add `--format latex` for LaTeX, `--no-pdf` to skip compilation, `--no-check`
to skip model checks, `--symbolic` to look for formulas for smooth components
(about a minute each) and `--json` for machine-readable output.

Through MCP, call `analyse_series` with `values`, `times`, `name`, `max_depth`
and `patience`. The response holds `formula` and `search_path`. Call
`get_report` with its `report_id` for the full text.

### Step 5: Validate

```bash
python scripts/check_report.py report/
```

Expected output: `report OK`. If it lists problems, fix the cause and run the
report again before answering.

### Step 6: Report back

Give the structure, each component's sentence, its closed form when it has
one and the level where BIC stopped falling. Then give the report's path, or
its `report_id` through MCP.

## Examples

### Example 1: a file

User says: "What is going on in these eight years of sales?"

Actions:
1. Run `head sales.csv`: columns `month` and `units`, 96 rows.
2. Run the command of step 4.
3. Validate, then report.

Result: `Lin + WN + Per`, BIC 449.2. A linear trend `99.63 + 6.095 (t - 2016)`,
a yearly cycle of amplitude 7.95, noise of variance 3.578. BIC stopped falling
at level 5.

### Example 2: values through MCP

User says: "Fit a formula to these 40 measurements."

Actions:
1. Call `analyse_series` with the values, `max_depth=20` and `patience=2`.
2. Read `formula.explicit`. Each entry is a closed form or null.

Result: the closed forms, with the covariance for components that have none.

### Example 3: the 2014 results

User says: "What did the Automatic Statistician find on the airline series?"

Actions: call `list_series`, then `describe_published` with `01-airline`.

Result: the published model and its ten-level search, with no new search.

## Troubleshooting

### Exit status 2, "no column"

Cause: wrong `--time` or `--value`. Solution: list the columns, then use the
obvious ones or ask.

### Warning "md2pdf not found" or "latexmk not found"

Cause: no PDF compiler. The source was written. Solution: name the missing
tool to the user.

### Warning "PySR not found"

Cause: the optional extra is missing. Smooth components keep their covariance.
Solution: `pip install "automatic-statistician-revisited[symbolic]"`.

### `invalid_input` from the API or MCP

Cause: a bound or a name was refused. Solution: read `error.message`, which
names the field. Over six levels needs `patience`.

### The search runs for too long

Cause: too many levels for the series. Solution: stop it, rerun with fewer
levels and say so.
