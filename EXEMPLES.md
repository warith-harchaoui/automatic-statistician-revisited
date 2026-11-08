# Exemples

[In English](EXAMPLES.md)

Chaque sortie ci-dessous vient du code de ce dépôt, le 7 novembre 2026. Les
durées ont été mesurées sur un ordinateur portable partagé avec d'autres
calculs. Le code est identique à celui d'[EXAMPLES.md](EXAMPLES.md) ; seuls
les commentaires changent de langue.

Chaque exemple qui écrit un rapport a son script dans
[examples/](examples/) ; le rapport écrit l'accompagne deux fois :
`markdown/` contient `report.md`, ses figures SVG et `report.pdf` compilé
par md2star ; `latex/` contient `report.tex`, ses figures PDF et
`report.pdf` compilé par `latexmk`. Un rapport est décrit une fois, puis
écrit dans les deux formats par `report.write`.

| # | Script | Rapport |
|---|---|---|
| 1 | [01_rescore_published.py](examples/01_rescore_published.py) | aucun |
| 2 | [02_describe_airline.py](examples/02_describe_airline.py) | [markdown](examples/02_airline_report/markdown/report.md), [LaTeX](examples/02_airline_report/latex/report.tex) |
| 3 | [03_search_airline.py](examples/03_search_airline.py) | [markdown](examples/03_search_report/markdown/report.md), [LaTeX](examples/03_search_report/latex/report.tex) |
| 4 | [04_make_sales.py](examples/04_make_sales.py), [04_sales_cli.sh](examples/04_sales_cli.sh) | [markdown](examples/04_sales_report/markdown/report.md), [LaTeX](examples/04_sales_report/latex/report.tex) |
| 5 | [05_sales_pandas.py](examples/05_sales_pandas.py) | [markdown](examples/05_sales_report/markdown/report.md), [LaTeX](examples/05_sales_report/latex/report.tex) |
| 6 | [06_cks_2013.py](examples/06_cks_2013.py) | [markdown](examples/06_cks_2013_report/markdown/report.md), [LaTeX](examples/06_cks_2013_report/latex/report.tex) |
| 7 | [07_autogp_airline.py](examples/07_autogp_airline.py) | [markdown](examples/07_autogp_report/markdown/report.md), [LaTeX](examples/07_autogp_report/latex/report.tex) |
| 8 | [08_published_2014.py](examples/08_published_2014.py) | [un dossier par série](examples/08_published_2014/) |

Chaque script se lance depuis la racine du dépôt, par exemple
`python examples/02_describe_airline.py`.

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

## 2. Décrire ce modèle en anglais, en Markdown et en LaTeX

```python
from autostat import describe

report = describe(model, x, y, name="airline passengers", out_dir="airline_report/markdown")
report.write("airline_report/latex", "latex")
for summary, _, _ in report.texts:
    print(summary)
# A linearly increasing function
# An approximately periodic function with a period of 1.0 years and with linearly increasing amplitude
# A smooth function
# Uncorrelated noise with linearly increasing standard deviation
```

Chaque dossier contient le rapport complet, cinq figures et le tableau de
vérification, avec son PDF. Les deux ensemble : 51 secondes, vérification
et deux compilations comprises. `describe(..., report_format="latex")`
n'écrit que la version LaTeX ; `pdf=False` saute la compilation.

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
# examples/04_make_sales.py : huit ans de ventes mensuelles, tendance + cycle annuel + bruit
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
# pdf: sales_report/report.pdf
```

Deux minutes, sept sur un portable chargé. La progression part sur la sortie d'erreur ; les lignes
ci-dessus, sur la sortie standard. `--format latex` écrit `report.tex` à
la place, `--no-pdf` saute la compilation et `--json` imprime une ligne
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
    report_format="latex",
)
```

Un index de dates devient des années décimales : le rapport parle alors en
mois et en années. Une série de plus de 1 000 points est d'abord moyennée en
1 000 classes (`max_points=`), car le coût croît comme le cube de la
longueur.

## 6. La configuration de 2013

```python
result = search(x, y, Experiment.cks_2013(max_depth=3, n_jobs=4))
print(result.best.kernel.structure(), round(result.best.bic, 1))
# Lin x RQ + RQ 1370.4
```

Ajouter, multiplier, remplacer, rien d'autre ; la quadratique rationnelle
parmi les noyaux de base ; le bruit dans la vraisemblance. En ligne de
commande : `--preset cks-2013`.

## 7. Une loi a posteriori sur les structures avec AutoGP

```python
from autostat.smc import AutoGP, linear_schedule
from autostat.smc.report import describe_posterior

model = AutoGP(x.ravel(), y.ravel(), n_particles=8, seed=0)
model.fit_smc(linear_schedule(len(x), 0.10), n_mcmc=75, n_hmc=10)
for weight, formula in model.structures()[:3]:
    print(f"{weight:.3f}  {formula}")
# 0.543  (Lin * (((Lin * Per) + ((Lin * Lin) * (Lin + (Per + GE)))) + Lin))
# 0.276  ((Lin * (((Per + (GE * Lin)) + Lin) + Lin)) * Lin)
# 0.089  ((Lin * ((Per + (((Lin + GE) * Lin) * GE)) + Lin)) * Lin)

posterior = describe_posterior(model, name="airline passengers", out_dir="autogp_report")
```

La page de synthèse liste les structures avec leurs poids ; chacune des
plus lourdes a son propre rapport, avec la décomposition et les
vérifications de 2014. Le script entier a pris 21 minutes sur huit
particules, les deux formats de rapport compris.

## 8. Les treize modèles publiés en 2014

`examples/08_published_2014.py` décrit le modèle final de chaque série de
référence, des passagers aériens aux salaires, dans les deux formats. La
série solaire montre les fenêtres de changement : ses composantes
s'appliquent « until 1643 and from 1716 onwards », les années qui
entourent le minimum de Maunder.

## 9. L'interface graphique

```bash
autostat-gui
# autostat GUI: http://127.0.0.1:8765/
```

Le navigateur ouvre un formulaire généré depuis la ligne de commande par
sprezzature-cli-gui : un champ par option, valeurs par défaut remplies.
« Build command » affiche la ligne de commande équivalente ; « Lancer sur
cette machine » lance l'analyse par l'hôte local et liste les fichiers du
rapport. L'hôte n'écoute que 127.0.0.1.

## 10. L'API HTTP

```bash
uvicorn autostat.api:app
```

```python
import httpx

api = httpx.Client(base_url="http://127.0.0.1:8000", timeout=3600)
print(api.get("/v1/series").json()[:3])
# ['01-airline', '02-solar', '03-mauna']
body = api.post("/v1/published", json={"series": "02-solar", "report_format": "latex", "check": False}).json()
print(body["structure"])
# C + CW(SE + SE + WN x Lin + WN x Lin + SE x Per, WN + C)
print(body["components"][3])
# An approximately periodic function with a period of 10.8 years. This function applies until 1643 and from 1716 onwards
pdf = api.get(f"/v1/reports/{body['report_id']}/files/report.pdf").content
```

38 secondes sans vérification ni PDF. `POST /v1/analyses` prend `values`
(3 à 1 000), des `times` facultatifs, `max_depth` (1 à 6) et les options du
rapport. Les erreurs partagent un même corps, par exemple
`{"error": {"code": "invalid_input", "message": "values: List should have
at least 3 items after validation, not 2"}}`. Les rapports vont sous
`AUTOSTAT_REPORTS_DIR` (par défaut `./autostat_reports`).

## 11. Le serveur MCP

```bash
autostat-mcp            # routes HTTP et /mcp sur 127.0.0.1:8000
```

Déclarez `http://127.0.0.1:8000/mcp` dans l'hôte d'agents. Il publie
quatre outils : `list_series`, `describe_published`, `analyse_series` et
`get_report`, avec les noms d'arguments et les bornes de l'API.

## 12. La compétence d'agent

Copiez ou liez [skills/autostat](skills/autostat) dans le dossier des
compétences de l'environnement (`~/.claude/skills/` pour Claude Code).
L'agent oriente alors « que contient cette série », « décompose cette
série » ou « écris un rapport LaTeX de ce CSV » vers autostat, selon
[TRIGGERS.md](TRIGGERS.md).
