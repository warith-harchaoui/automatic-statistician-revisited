# Triggers: autostat

When someone holds a time series and wants to know what is in it, in words,
and what to call.

This file is written for an agent: a Claude Code or OpenCode skill, an MCP
(Model Context Protocol) host, anything choosing a tool on someone's behalf.
Humans are welcome, but the routing rules below are the point.

---

## The generalisation, stated once

> **Route here when the request pairs a series with an explanation.** The
> vocabulary is closed:
>
> - **the series**: time series, series, signal, monthly / daily / yearly
>   data, sales, traffic, temperature, a CSV with a date column, « une série
>   temporelle », « des données mensuelles »;
> - **the explanation**: trend, seasonality, cycle, period, structure,
>   decomposition, components, "what drives it", "explain", report,
>   kernel, Gaussian process, « tendance », « saisonnalité », « décomposer »,
>   « expliquer », « rapport ».
>
> A request for **the formula of a noisy curve over time** routes here too:
> the structure search is a symbolic regression over kernels and every report
> states the expression it found.
>
> Words naming the original work route here on their own: Automatic
> Statistician, ABCD, compositional kernel search, Lloyd 2014, Duvenaud
> 2013, AutoGP.

"What is going on in these eight years of sales?", « décris-moi cette
série », "is there a yearly cycle in this file" all pair the two and all
route here.

The instinct this file exists to interrupt is fitting a forecasting model
and reading its coefficients when the person asked what the series is made
of. autostat answers in sentences: "A linearly increasing function", "An
approximately periodic function with a period of 1.0 years".

Not this one when the request is a forecast score or a model comparison on
many series (`benchmark/run_2026.py` in this repository, not a surface), a
chart of the series alone (`sprezzature-figures`), or a series longer than
a few thousand points that must keep every point (the search bins to 1,000).

---

## Every tool with the request it answers

| Tool | Surface | The request it answers |
|---|---|---|
| `autostat FILE --time T --value V` | CLI | "Explain the series in this CSV or Parquet file." Writes a report directory and prints the structure and one line per component. |
| `autostat ... --format latex` | CLI | Same, with a LaTeX report compiled by `latexmk` instead of Markdown compiled by md2star. |
| `autostat ... --max-depth 20 --patience 2` | CLI | "Search until it stops improving." Runs two levels past the last BIC fall so the report can show what going on finds. |
| `autostat ... --symbolic` | CLI | "Give me a formula for the smooth part too." PySR reads each smooth component, about a minute each. |
| `autostat ... --json` | CLI | Same, one JSON object on standard output for a script, with the formula and the search path. |
| `autostat-gui` | GUI | "I would rather fill in a form." Opens the command line as a web form on 127.0.0.1 that runs it. |
| `analyse`, `describe`, `search` | Library | The same jobs from Python, on a `pandas.Series` or arrays. |
| `autostat.smc.AutoGP` + `describe_posterior` | Library | "Give me a posterior over structures, not one winner." |
| `list_series` | MCP / HTTP `GET /v1/series` | "Which benchmark series are there?" The thirteen of the 2014 paper. |
| `describe_published` | MCP / HTTP `POST /v1/published` | "Show me what the 2014 system said about the airline series." No search; about a minute. |
| `analyse_series` | MCP / HTTP `POST /v1/analyses` | "Explain these values." Values (and optional times) in JSON, 1,000 points at most, 6 levels or 20 with a `patience`. Returns `formula` and `search_path`. |
| `get_report` | MCP / HTTP `GET /v1/reports/{id}` | "Show me the report." The Markdown or LaTeX source of a report written earlier. |
| `get_report_file` | HTTP `GET /v1/reports/{id}/files/{name}` | "Give me the PDF / a figure." Not an MCP tool: binary. |
| `health` | HTTP `GET /health` | "Is the server up?" Not an MCP tool. |

---

## Phrases with where they go

| The user says | First call |
|---|---|
| "what's in this series", "describe this data", « que contient cette série » | `analyse_series` (values in hand) or `autostat FILE` (a file) |
| "is there seasonality", "find the period", « y a-t-il un cycle » | same; the report names each period in words |
| "trend + seasonality decomposition", « décomposer en tendance et saisonnalité » | same; each component is a section with its figure |
| "write me a report", "a PDF I can send", « un rapport PDF » | same, `pdf` on (default) |
| "in LaTeX", "for my paper", « en LaTeX » | same, `report_format="latex"` / `--format latex` |
| "the airline example", "what did the 2014 paper find" | `describe_published` with a name from `list_series` |
| "fit a formula to this curve", « quelle formule pour cette courbe » | same; read `formula.explicit` |
| "when should the search stop", "is the depth enough" | same with a `patience`; read `search_path.findings` |
| "explain it without the command line", « une interface » | `autostat-gui` |

## Before a long call

A search of depth *d* costs several times depth *d − 1*. Two levels take
seconds to minutes; ten levels on 144 points took four hours on six
processes. A shallow search can miss most of the structure: on the sales
example, three levels gave an approximation and five the formula the data
were drawn from. Up to about 100 points, use a patience. Beyond, start at
depth 3 and say so.
