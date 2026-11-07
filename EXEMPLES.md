# Exemples

[In English](EXAMPLES.md)

Chaque sortie ci-dessous vient du code de ce dépôt, le 7 novembre 2026. Les
durées ont été mesurées sur un ordinateur portable partagé avec d'autres
calculs. Le code est identique à celui d'[EXAMPLES.md](EXAMPLES.md) ; seuls
les commentaires changent de langue.

## 1. Recalculer un modèle publié par le système de 2014

Le paquet livre les treize séries de référence de l'article d'AAAI 2014 et
les résultats de la recherche d'origine.

```python
import torch
from autostat import benchmarks
from autostat.model import negative_log_likelihood

print(benchmarks.names()[:3])
# ['01-airline', '02-solar', '03-mauna']

x, y = benchmarks.load("01-airline")
model = benchmarks.published("01-airline")[-1]   # le gagnant final de 2014
print(model.kernel.structure())
# SE + Lin + WN x Lin + SE x Per x Lin
print(round(model.nll, 3))                        # tel que publié en 2014
# 534.518

theta = torch.as_tensor(model.theta_vector())
with torch.no_grad():
    print(round(float(negative_log_likelihood(model, torch.as_tensor(x), torch.as_tensor(y), theta)), 3))
# 534.518
```

## 2. Décrire ce modèle en anglais

```python
from autostat import describe

report = describe(model, x, y, name="airline passengers", out_dir="airline_report")
for summary, _, _ in report.texts:
    print(summary)
# A linearly increasing function
# An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude
# A smooth function
# Uncorrelated noise with linearly increasing standard deviation
```

`airline_report/report.md` contient le rapport complet, cinq figures et le
tableau de vérification. Seize secondes, vérification comprise.

## 3. Chercher depuis zéro, brièvement

```python
from autostat import Experiment, search

result = search(x, y, Experiment.abcd_2014(max_depth=2, n_jobs=4))
print(result.best.kernel.structure(), round(result.best.bic, 1))
# WN + Per x SE + SE 1236.4
```

Deux niveaux trouvent déjà le cycle annuel. Les dix niveaux publiés
atteignent un BIC de 1123,7 ; les dix niveaux de ce portage atteignent
1108,9, en 4,1 heures avec six processus.

## 4. Votre propre CSV, en ligne de commande

```python
# make_sales.py : huit ans de ventes mensuelles, tendance + cycle annuel + bruit
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
months = pd.date_range("2016-01-01", periods=96, freq="MS")
k = np.arange(96)
units = 100 + 0.5 * k + 8 * np.sin(2 * np.pi * k / 12) + rng.normal(0, 2, 96)
pd.DataFrame({"month": months.strftime("%Y-%m-%d"), "units": units.round(2)}).to_csv("sales.csv", index=False)
```

```bash
autostat sales.csv --time month --value units --max-depth 3 --no-check --out sales_report
# WN + Lin x SE + Per x SE  (BIC 458.4)
#   1. A very smooth monotonically increasing function with linearly increasing marginal standard deviation
#   2. An approximately periodic function with a period of 1.0 years
#   3. Uncorrelated noise
# report: sales_report/report.md
```

Sept minutes. La progression part sur la sortie d'erreur ; les lignes
ci-dessus, sur la sortie standard. `--json` les remplace par une ligne
lisible par une machine.

## 5. Votre propre série pandas, en Python

```python
import pandas as pd
from autostat import Experiment, analyse

series = pd.read_csv("sales.csv", parse_dates=["month"]).set_index("month")["units"]
result, report = analyse(
    series,
    name="monthly sales",
    out_dir="sales_report",
    experiment=Experiment.abcd_2014(max_depth=3),
)
```

Un index de dates devient des années décimales : le rapport parle alors en
mois et en années. Une série de plus de 1 000 points est d'abord moyennée en
1 000 classes (`max_points=`), car le coût croît comme le cube de la
longueur.

## 6. La configuration de 2013

```python
result = search(x, y, Experiment.cks_2013(max_depth=3, n_jobs=4))
```

Ajouter, multiplier, remplacer, rien d'autre ; la quadratique rationnelle
parmi les noyaux de base ; le bruit dans la vraisemblance. En ligne de
commande : `--preset cks-2013`.
