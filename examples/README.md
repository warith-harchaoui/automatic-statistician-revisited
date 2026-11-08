# examples

[En français](LISEZMOI.md)

One script per example of [EXAMPLES.md](../EXAMPLES.md), with the reports
it wrote. Each report folder holds two versions of the same report:

- `markdown/`: `report.md`, its SVG figures, `report.pdf` compiled by md2star;
- `latex/`: `report.tex`, its PDF figures, `report.pdf` compiled by `latexmk`.

| Script | What it does | Report |
|---|---|---|
| `01_rescore_published.py` | re-scores the 2014 airline winner | none |
| `02_describe_airline.py` | describes the 2014 airline winner | `02_airline_report/` |
| `03_search_airline.py` | two-level search on airline passengers | `03_search_report/` |
| `04_make_sales.py`, `04_sales_cli.sh` | synthetic sales, then the command line | `04_sales_report/` |
| `05_sales_pandas.py` | the same sales from pandas, with model checks | `05_sales_report/` |
| `06_cks_2013.py` | the 2013 configuration on airline passengers | `06_cks_2013_report/` |
| `07_autogp_airline.py` | an AutoGP posterior, one report per heavy structure | `07_autogp_report/` |
| `08_published_2014.py` | the thirteen published 2014 models | `08_published_2014/<series>/` |

Run them from the repository root: `python examples/02_describe_airline.py`.
