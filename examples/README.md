# examples

[En français](LISEZMOI.md)

Each example of [EXAMPLES.md](../EXAMPLES.md) has its script here, next to
the reports it wrote. A report folder holds the same report twice:

- `markdown/` has `report.md`, its SVG figures and `report.pdf`, compiled by
  md2star;
- `latex/` has `report.tex`, its PDF figures and `report.pdf`, compiled by
  `latexmk`.

| Script | What it does | Report |
|---|---|---|
| `01_rescore_published.py` | scores the 2014 airline winner again and prints the published likelihood beside ours | none |
| `02_describe_airline.py` | describes the 2014 airline winner in English | `02_airline_report/` |
| `03_search_airline.py` | searches the airline series from scratch until BIC stops falling, then two levels more | `03_search_report/` |
| `04_make_sales.py`, `04_sales_cli.sh` | makes eight years of synthetic monthly sales, then runs the command line on them | `04_sales_report/` |
| `05_sales_pandas.py` | runs the same sales from a pandas series, with the model checks | `05_sales_report/` |
| `06_cks_2013.py` | runs the 2013 configuration on the airline series | `06_cks_2013_report/` |
| `07_autogp_airline.py` | fits an AutoGP posterior and writes one report for each of its three heaviest structures | `07_autogp_report/` |
| `08_published_2014.py` | describes the thirteen models the 2014 system published | `08_published_2014/<series>/` |
| `redescribe.py` | writes the reports of examples 3 to 6 again from their saved `search.json`, with no new search | the same folders |

Every script runs from the repository root:

```bash
python examples/02_describe_airline.py
```

The search of example 3 took about four and a half hours on four worker
processes. When only the way reports are written has changed,
`python examples/redescribe.py 3` rewrites its reports from the saved search
and spares those hours.
