# Triggers: autostat

Someone holds a time series and wants to know, in words, what it is made of.
This file says when to call autostat and which of its tools to call.

The reader here is an agent: a Claude Code or OpenCode skill, an MCP (Model
Context Protocol) host, any program that picks a tool on someone's behalf.
Humans are welcome too, but the routing rules are what the file is for.

---

## The rule, stated once

> **Route here when the request pairs a series with an explanation.** The
> vocabulary is closed:
>
> - **The series**: time series, series, signal, monthly / daily / yearly
>   data, sales, traffic, temperature, a CSV with a date column, « une série
>   temporelle », « des données mensuelles ».
> - **The explanation**: trend, seasonality, cycle, period, structure,
>   decomposition, components, "what drives it", "explain", report,
>   kernel, Gaussian process, « tendance », « saisonnalité », « décomposer »,
>   « expliquer », « rapport ».
>
> A request for **the formula of a noisy curve over time** routes here too.
> The structure search is a symbolic regression over kernels. Every report
> writes out the expression it found.
>
> The names of the original work route here alone: Automatic Statistician,
> ABCD, compositional kernel search, Lloyd 2014, Duvenaud 2013, AutoGP.

"What is going on in these eight years of sales?", « décris-moi cette
série », "is there a yearly cycle in this file": each one pairs the two
halves, so each one routes here.

This file interrupts one instinct in particular. Asked what a series is made
of, an agent fits a forecasting model and reads off its coefficients. autostat
answers in sentences instead: "A linearly increasing function", "An
approximately periodic function with a period of 1.0 years".

Send the request elsewhere when it asks for a forecast score or a model
comparison over many series (`benchmark/run_2026.py` in this repository, not a
surface), for a chart of the series alone (`sprezzature-figures`), or for a
series of several thousand points that has to keep every point (the search
bins to 1,000).

---

## Every tool with the request it answers

| Tool | Surface | The request it answers |
|---|---|---|
| `autostat FILE --time T --value V` | CLI | "Explain the series in this CSV or Parquet file." Writes a report directory and prints the structure with one line per component. |
| `autostat ... --format latex` | CLI | The same, with a LaTeX report compiled by `latexmk` in place of Markdown compiled by md2star. |
| `autostat ... --max-depth 20 --patience 2` | CLI | "Search until it stops improving." Runs two levels past the last fall in BIC, so the report shows what searching further finds. |
| `autostat ... --symbolic` | CLI | "Give me a formula for the smooth part too." PySR reads each smooth component, in about a minute each. |
| `autostat ... --json` | CLI | The same job, as one JSON object on standard output for a script, with the formula and the search path. |
| `autostat-gui` | GUI | "I would rather fill in a form." Opens the command line as a web form on 127.0.0.1, which runs it. |
| `analyse`, `describe`, `search` | Library | The same jobs from Python, on a `pandas.Series` or on arrays. |
| `autostat.smc.AutoGP` + `describe_posterior` | Library | "Give me a posterior over structures, not one winner." |
| `list_series` | MCP / HTTP `GET /v1/series` | "Which benchmark series are there?" The thirteen of the 2014 paper. |
| `describe_published` | MCP / HTTP `POST /v1/published` | "Show me what the 2014 system said about the airline series." No search runs: the answer takes about a minute. |
| `analyse_series` | MCP / HTTP `POST /v1/analyses` | "Explain these values." Values (and optional times) in JSON, 1,000 points at most, 6 levels, or 20 with a `patience`. Returns `formula` and `search_path`. |
| `get_report` | MCP / HTTP `GET /v1/reports/{id}` | "Show me the report." The Markdown or LaTeX source of a report written earlier. |
| `get_report_file` | HTTP `GET /v1/reports/{id}/files/{name}` | "Give me the PDF / a figure." Binary, so it is not an MCP tool. |
| `health` | HTTP `GET /health` | "Is the server up?" Not an MCP tool. |

---

## Phrases with where they go

| The user says | First call |
|---|---|
| "what's in this series", "describe this data", « que contient cette série » | `analyse_series` (values in hand) or `autostat FILE` (a file) |
| "is there seasonality", "find the period", « y a-t-il un cycle » | the same call: the report names each period in words |
| "trend + seasonality decomposition", « décomposer en tendance et saisonnalité » | the same call: each component gets a section with its figure |
| "write me a report", "a PDF I can send", « un rapport PDF » | the same call, with `pdf` on (the default) |
| "in LaTeX", "for my paper", « en LaTeX » | the same call, with `report_format="latex"` / `--format latex` |
| "the airline example", "what did the 2014 paper find" | `describe_published` with a name from `list_series` |
| "fit a formula to this curve", « quelle formule pour cette courbe » | the same call as the first row, then read `formula.explicit` |
| "when should the search stop", "is the depth enough" | the same call with a `patience`, then read `search_path.findings` |
| "how sure is it of that structure", « à quel point est-ce sûr » | the same call with a `patience`: `search_path.findings` gives the posterior weight of the kept structure and of the next |
| "explain it without the command line", « une interface » | `autostat-gui` |

## Before a long call

Each level of search costs several times the one before. Two levels take
seconds to minutes. Ten levels on 144 points took four hours on six
processes. A shallow search misses structure: on the sales example, three
levels gave an approximation and five gave back the formula the data were
drawn from. Up to about 100 points, use a patience. Beyond that, start at
depth 3 and tell the user so.
