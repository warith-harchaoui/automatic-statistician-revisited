# Exemples

[In English](EXAMPLES.md)

Chaque sortie ci-dessous vient du code de ce dépôt, les 7 et 8 novembre 2026. Les
durées ont été mesurées sur un ordinateur portable partagé avec d'autres
calculs. Le code est identique à celui d'[EXAMPLES.md](EXAMPLES.md) ; seuls
les commentaires changent de langue.

La même analyse s'atteint de six façons.

| Surface | Lancement | Exemple |
|---|---|---|
| Bibliothèque Python | `from autostat import analyse` | sections 1 à 8, 13 |
| Ligne de commande (Click) | `autostat donnees.csv ...` | section 4 |
| Interface graphique (sprezzature-cli-gui) | `autostat-gui` | section 9 |
| API HTTP (FastAPI) | `uvicorn autostat.api:app` | section 10 |
| Serveur MCP (FastAPI-MCP) | `autostat-mcp` | section 11 |
| Compétence d'agent | `skills/autostat` avec `TRIGGERS.md` | section 12 |

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

Chaque dossier contient le rapport complet avec son PDF : les données
d'abord, le résumé, la formule trouvée, le chemin de recherche avec sa courbe
du BIC (ici les dix niveaux publiés en 2014), une section par composante avec
sa figure et le tableau de vérification. Les deux ensemble : environ une
minute, vérification et deux compilations comprises. `describe(..., report_format="latex")`
n'écrit que la version LaTeX ; `pdf=False` saute la compilation.

## 3. Chercher depuis zéro

```python
from autostat import Experiment, search

result = search(x, y, Experiment.abcd_2014(max_depth=10, n_jobs=6))
print(result.best.kernel.structure(), round(result.best.bic, 1))
# Lin x Lin + Lin x Lin x WN + Lin x Per x SE + SE 1108.9
```

Dix niveaux sur 144 points ont pris 4,1 heures avec six processus ; la recherche
publiée atteignait un BIC de 1123,7. Deux niveaux s'arrêtent à
`WN + Per x SE + SE` (BIC 1236,4) : le cycle annuel sans la croissance.
`Experiment.abcd_2014(max_depth=20, stopping_criteria=["no_improvement"],
patience=2)` cherche jusqu'à ce que le BIC n'ait plus baissé deux niveaux de
suite, comme le fait [03_search_airline.py](examples/03_search_airline.py).

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
autostat sales.csv --time month --value units --max-depth 20 --patience 2 --no-check --out sales_report
# Lin + WN + Per  (BIC 449.2)
#   1. A linearly increasing function
#   2. A periodic function with a period of 1.0 years
#   3. Uncorrelated noise
# report: sales_report/report.md
# pdf: sales_report/report.pdf
```

La recherche s'arrête deux niveaux après que le BIC a cessé de baisser : elle
trouve `Lin + WN + Per`, la formule dont les données sont tirées, au niveau 5 ;
les niveaux 6 et 7 ne trouvent rien de mieux. Avec `--max-depth 3`, elle
s'arrêtait à une approximation, `WN + Lin x SE + Per x SE` (BIC 458.4). Environ
quinze minutes. La progression part sur la sortie d'erreur ; les lignes
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
    experiment=Experiment.abcd_2014(
        max_depth=20, stopping_criteria=["no_improvement"], patience=2
    ),
    report_format="latex",
)
```

Un index de dates devient des années décimales : le rapport parle alors en
mois et en années. Une série de plus de 1 000 points est d'abord moyennée en
1 000 classes (`max_points=`), car le coût croît comme le cube de la
longueur.

## 6. La configuration de 2013

```python
exp = Experiment.cks_2013(max_depth=20, n_jobs=4, stopping_criteria=["no_improvement"], patience=2)
result = search(x, y, exp)
print(result.best.kernel.structure(), round(result.best.bic, 1))
# Lin x Per + (Lin + RQ) x (Lin + Lin x Per) 1112.4
```

Ajouter, multiplier, remplacer, rien d'autre ; la quadratique rationnelle parmi
les noyaux de base ; le bruit dans la vraisemblance. En ligne de commande :
`--preset cks-2013`. Le BIC baisse à chaque niveau jusqu'au dixième, puis deux
niveaux ne trouvent rien de mieux : douze niveaux, 23 minutes. À trois niveaux,
la recherche gardait `Lin x RQ + RQ` (BIC 1370,4), sans cycle annuel ; le cycle
apparaît au niveau 4. Le rédacteur de 2014 n'a pas de règle pour la quadratique
rationnelle : deux composantes portent la phrase « This simple AI is not capable
of describing the component Lin x RQ », comme dans l'original ; la section sur la
formule les écrit quand même.

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
« Build command » affiche la ligne de commande équivalente ; « Lancer sur cette
machine » lance l'analyse par l'hôte local et liste les fichiers du rapport.
L'hôte n'écoute que 127.0.0.1. La page ne charge rien d'un autre hôte : sa
première construction télécharge le CLI de Tailwind (environ 45 Mo) puis ne
compile que les classes qu'elle emploie.

## 10. L'API HTTP

```bash
uvicorn autostat.api:app
```

```python
import httpx

api = httpx.Client(base_url="http://127.0.0.1:8000", timeout=3600)
print(api.get("/v1/series").json()[:3])
# ['01-airline', '02-solar', '03-mauna']
body = api.post("/v1/published", json={"series": "12-births", "check": False, "pdf": False}).json()
print(sorted(body))
# ['bic', 'components', 'files', 'formula', 'report_format', 'report_id', 'search_path', 'structure']
print(body["structure"], round(body["bic"], 1))
# WN + SE + Per + Per 9398.3
print(body["search_path"]["settled_level"])
# 5
print(body["formula"]["explicit"][1][:70])
# f_{2}(t) \approx 38.53\cos\!\left(\frac{2\pi \cdot (t - 1977)}{0.01917}
```

La réponse contient la formule en chaînes LaTeX et la recherche niveau par
niveau. Les naissances ont un cycle hebdomadaire : leur période, 0,01917 an,
vaut sept jours. `POST /v1/analyses` prend `values` (3 à 1 000), des `times`
facultatifs, `max_depth` (1 à 6, ou jusqu'à 20 avec une `patience`), `patience`,
`symbolic` et les options du rapport. Les erreurs partagent un même corps, par
exemple `{"error": {"code": "invalid_input", "message": "values: List should
have at least 3 items after validation, not 2"}}`. Les rapports vont sous
`AUTOSTAT_REPORTS_DIR` (par défaut `./autostat_reports`).

## 11. Le serveur MCP

```bash
autostat-mcp            # routes HTTP et /mcp sur 127.0.0.1:8000
claude mcp add --transport http autostat http://127.0.0.1:8000/mcp
```

Une session sur le protocole, tirée de `tests/test_mcp.py` :

```text
tools/list   list_series, describe_published, analyse_series, get_report
tools/call   list_series -> ["01-airline", "02-solar", "03-mauna", ...]
tools/call   get_report {"report_id": "x"} -> isError: true,
             {"error":{"code":"not_found","message":"no report 'x'"}}
```

Quatre outils sont publiés, avec les noms d'arguments et les bornes de l'API.
`health` et le téléchargement binaire restent hors des outils de l'agent.

## 12. La compétence d'agent

```bash
python scripts/build_skill.py
# wrote dist/autostat-skill.zip
cp -r skills/autostat ~/.claude/skills/      # Claude Code
```

Téléversez le zip dans Claude, sous Settings > Capabilities > Skills. L'agent
oriente alors « que contient cette série », « quelle formule pour cette courbe »
ou "fit a formula to this noisy curve" vers autostat, selon
[TRIGGERS.md](TRIGGERS.md). La compétence vérifie son propre travail :

```bash
python skills/autostat/scripts/check_report.py examples/02_airline_report/markdown
# report OK
```

[evals/triggers.yaml](skills/autostat/evals/triggers.yaml) liste 14 requêtes qui
doivent charger la compétence et 7 qui ne le doivent pas, pour le test de
déclenchement du guide d'Anthropic. `tests/test_skill.py` tient la compétence aux
règles du guide.

## 13. Lire la formule et le chemin de recherche

Chaque rapport énonce la formule trouvée et étudie où la recherche devrait
s'arrêter. Sur les ventes des exemples 4 et 5 :

```python
import pandas as pd
from autostat import Experiment, analyse, findings

series = pd.read_csv("sales.csv", parse_dates=["month"]).set_index("month")["units"]
experiment = Experiment.abcd_2014(max_depth=20, stopping_criteria=["no_improvement"], patience=2)
result, report = analyse(series, name="monthly sales", out_dir="sales_report", experiment=experiment)

for closed_form, note in zip(report.formula().explicit, report.formula().notes):
    print(closed_form, "|", note)
# f_{1}(t) = 99.63 + 6.095\,(t - 2016) | exact polynomial of degree 1
# f_{2}(t) \approx 7.95\cos\!\left(\frac{2\pi \cdot (t - 2016)}{1} - 1.807\right) + 0.3961\cos\!\left(...\right) | Fourier series, 2 harmonics, largest gap 0.027 % of the component
# None | None

for sentence in findings(report.path, report.model.ndata):
    print(sentence)
# The search ran 7 levels (stopping after 2 level(s) without a BIC fall above 0.1, at most 20) and keeps the model of level 7, Lin + WN + Per, BIC 449.2.
# That structure was first reached at level 5; the levels after it only refitted its parameters, lowering BIC by 0.0 in all.
# The best different structure, Lin + WN + Per x SE (level 4), is 4.6 above it: positive evidence for the kept one on the Kass and Raftery scale.
# BIC stopped falling at level 5. The 2 level(s) searched past it kept its structure and lowered BIC by 0.00 in all, less than the tolerance of 0.1: going on buys nothing the data pay for.
# A rule stopping after 1 level(s) without a fall above 0.1 would have ended the search at level 6 and kept the same structure, less refitted.
# A rule stopping after 2 level(s) without a fall above 0.1 would have ended the search at level 7 and kept the same model.
# The kept expression has 6 parameters for 96 points: its BIC is a fit term 2 NLL = 421.9 plus a complexity term p ln n = 27.4, each extra parameter costing 4.56; it splits into 3 components, 2 of them with a closed form.
```

Les données sont tirées d'une pente de 6 par an depuis 100, d'un sinus
d'amplitude 8 et de période un an plus un bruit de variance 4. La formule donne
6,095, 7,95, un an et 3,578.
