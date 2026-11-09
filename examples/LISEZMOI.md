# examples

[In English](README.md)

Chaque exemple d'[EXEMPLES.md](../EXEMPLES.md) a ici son script, à côté des
rapports qu'il a écrits. Un dossier de rapport contient deux fois le même
rapport :

- `markdown/` contient `report.md`, ses figures SVG et `report.pdf`, compilé
  par md2star ;
- `latex/` contient `report.tex`, ses figures PDF et `report.pdf`, compilé
  par `latexmk`.

| Script | Ce qu'il fait | Rapport |
|---|---|---|
| `01_rescore_published.py` | recalcule le gagnant de 2014 sur les passagers aériens et imprime la vraisemblance publiée à côté de la nôtre | aucun |
| `02_describe_airline.py` | décrit ce gagnant en anglais | `02_airline_report/` |
| `03_search_airline.py` | cherche une structure pour les passagers aériens en partant de zéro, jusqu'à ce que le BIC cesse de baisser, puis deux niveaux de plus | `03_search_report/` |
| `04_make_sales.py`, `04_sales_cli.sh` | fabrique huit ans de ventes mensuelles synthétiques, puis lance la ligne de commande dessus | `04_sales_report/` |
| `05_sales_pandas.py` | reprend les mêmes ventes depuis une série pandas, avec les vérifications du modèle | `05_sales_report/` |
| `06_cks_2013.py` | lance la configuration de 2013 sur les passagers aériens | `06_cks_2013_report/` |
| `07_autogp_airline.py` | ajuste une loi a posteriori AutoGP et écrit un rapport pour chacune de ses trois structures les plus lourdes | `07_autogp_report/` |
| `08_published_2014.py` | décrit les treize modèles que le système de 2014 a publiés | `08_published_2014/<série>/` |
| `redescribe.py` | réécrit les rapports des exemples 3 à 6 depuis leur `search.json`, sans nouvelle recherche | les mêmes dossiers |

Chaque script se lance depuis la racine du dépôt :

```bash
python examples/02_describe_airline.py
```

La recherche de l'exemple 3 a pris environ quatre heures et demie sur quatre
processus. Quand seule la façon d'écrire les rapports a changé,
`python examples/redescribe.py 3` réécrit ses rapports depuis la recherche
enregistrée et épargne ces heures.
