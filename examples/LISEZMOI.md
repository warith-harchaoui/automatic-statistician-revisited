# examples

[In English](README.md)

Un script par exemple d'[EXEMPLES.md](../EXEMPLES.md), avec les rapports
qu'il a écrits. Chaque dossier de rapport contient deux versions du même
rapport :

- `markdown/` : `report.md`, ses figures SVG, `report.pdf` compilé par md2star ;
- `latex/` : `report.tex`, ses figures PDF, `report.pdf` compilé par `latexmk`.

| Script | Ce qu'il fait | Rapport |
|---|---|---|
| `01_rescore_published.py` | recalcule le gagnant de 2014 sur les passagers aériens | aucun |
| `02_describe_airline.py` | décrit ce gagnant | `02_airline_report/` |
| `03_search_airline.py` | recherche sur deux niveaux, passagers aériens | `03_search_report/` |
| `04_make_sales.py`, `04_sales_cli.sh` | ventes synthétiques, puis la ligne de commande | `04_sales_report/` |
| `05_sales_pandas.py` | les mêmes ventes depuis pandas, avec vérifications | `05_sales_report/` |
| `06_cks_2013.py` | la configuration de 2013 sur les passagers aériens | `06_cks_2013_report/` |
| `07_autogp_airline.py` | une loi a posteriori AutoGP, un rapport par structure lourde | `07_autogp_report/` |
| `08_published_2014.py` | les treize modèles publiés en 2014 | `08_published_2014/<série>/` |

À lancer depuis la racine du dépôt : `python examples/02_describe_airline.py`.
