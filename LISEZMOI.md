<p align="center"><img src="assets/logo-288.png" width="200" alt="Un statisticien à son bureau, l'index levé"></p>

# Automatic Statistician, revisité

[Read in English](README.md)

Donnez-lui une courbe bruitée. Il trouve la formule qui l'explique&nbsp;: une
tendance, des cycles, des variations lisses et du bruit, chacun avec ses
valeurs ajustées. Il décrit chaque part d'une phrase en anglais, confronte
chacune aux données et rédige l'analyse complète sous forme de rapport, en
Markdown ou en LaTeX.

Voici ce qu'il dit de douze années de passagers aériens mensuels, de 1949 à
1960 (le rapport l'écrit en anglais, nous traduisons)&nbsp;:

1. Une fonction qui croît linéairement, $f_1(t) = 92{,}68 + 31{,}58\,(t - 1949)$,
   en milliers de passagers par mois.
2. Une fonction à peu près périodique, de période 1,0 an, dont l'amplitude
   croît linéairement.
3. Une fonction lisse.
4. Un bruit sans corrélation, dont l'écart type croît linéairement.

Ces quatre phrases sont, mot pour mot, celles que l'Automatic Statistician
publiait pour la même série en 2014. Ce dépôt reprend ce système, dû à
Duvenaud, Lloyd, Grosse, Tenenbaum et Ghahramani (ICML 2013, AAAI 2014), qu'il
réécrit de son Python 2.7 et de son MATLAB d'origine vers Python 3 et `torch`.
Le code d'origine a reçu ses derniers commits en 2015 et 2017. Il réclame
MATLAB, la boîte à outils GPML (*Gaussian Processes for Machine Learning*) et
un Python en fin de vie depuis 2020. Celui-ci s'installe avec `pip` et
retrouve la vraisemblance de chaque modèle publié par le système de 2014.

```python
from autostat import analyse

result, report = analyse(series, name="ventes mensuelles", out_dir="rapport/")
print(result.best.kernel.structure())
# par exemple Lin + WN + Per
```

```bash
autostat ventes.csv --time mois --value unites --out rapport/
```

Le rapport s'ouvre sur les données, puis donne&nbsp;:

- un résumé, avec une phrase par composante&nbsp;;
- la formule trouvée, avec chaque paramètre ajusté et une forme explicite pour
  chaque composante qui en a une&nbsp;;
- le chemin de la recherche&nbsp;: la courbe du BIC, la structure retenue à
  chaque niveau et ce qu'un arrêt plus précoce aurait gardé&nbsp;;
- une section par composante, avec sa figure&nbsp;;
- les vérifications du modèle.

`rapport/report.pdf` est le même document, compilé par
[md2star](https://github.com/warith-harchaoui/md2star). Avec `--format latex`
(en Python, `report_format="latex"`), le rapport devient `report.tex`, compilé
par `latexmk`, comme l'écrivait le système de 2014. Les figures viennent de
[sprezzature-figures](https://github.com/warith-harchaoui/sprezzature-figures).
[EXEMPLES.md](EXEMPLES.md) déroule chaque exemple.
[examples/](examples/) contient chaque script avec le rapport qu'il a écrit,
dans les deux formats.

## Ce que fait la recherche

Un noyau est une formule qui dit à quel point deux points d'une série varient
ensemble. Deux mois voisins se ressemblent sous un noyau lisse. Deux mois de
juillet se ressemblent sous un noyau périodique. Sous un bruit, deux mesures ne
se doivent rien. La recherche bâtit ses formules à partir de quelques
noyaux de base&nbsp;:

| noyau de base | ce qu'il décrit |
|---|---|
| `SE`, exponentielle quadratique | une fonction lisse |
| `Per`, périodique | un motif qui se répète |
| `Lin`, linéaire | une tendance, ou une amplitude qui grandit |
| `C`, constante | un décalage |
| `WN`, bruit blanc | le bruit de mesure |
| `RQ`, quadratique rationnelle | des variations lisses à plusieurs échelles (préréglage 2013 seulement) |

Elle les combine par `+` (deux effets qui s'ajoutent), `x` (un effet qui en
module un autre), `CP` (un changement de régime&nbsp;: une structure, puis une
autre) et `CW` (une fenêtre de changement&nbsp;: une structure qui vaut un
temps). La recherche part d'un bruit seul et avance par niveaux. À chaque
niveau, elle réécrit la meilleure formule du niveau précédent de toutes les
façons que permet la grammaire, ajuste chaque candidate depuis neuf départs
tirés au hasard et garde celle dont le critère d'information bayésien (BIC)
est le plus bas. Le BIC ajoute à l'écart aux données un prix pour chaque
paramètre réglé&nbsp;: une formule qui colle aux données à force de paramètres
perd contre une formule plus simple qui colle presque aussi bien.

L'apprentissage automatique appelle ce travail la régression symbolique&nbsp;:
elle rend une formule là où d'autres méthodes rendent des poids.
Eureqa et PySR assemblent de l'arithmétique. Cette recherche assemble des
noyaux, si bien que le bruit et l'incertitude font partie de la formule au
lieu d'en être le reste. Le [rapport technique](doc/markdown/report.md) met
les deux face à face.

## Où la recherche s'arrête

Le système de 2014 cherchait toujours dix niveaux. Chaque rapport examine
désormais, pour sa série, si c'était trop ou trop peu. Il trace le BIC du
meilleur modèle à chaque niveau et marque le niveau où le BIC cesse de
baisser. Lancée avec `--patience 2`, la recherche s'arrête quand le BIC n'a
pas baissé deux niveaux de suite. Le rapport montre alors ce que ces deux niveaux
de plus ont trouvé.

Le rapport pèse aussi les structures entre elles, à la manière du *Bayesian
machine scientist* de Guimerà et de ses collègues (*Science Advances*, 2020)&nbsp;:
chaque structure rencontrée reçoit un poids a posteriori proportionnel à
`exp(-BIC/2)`. Sur la série publiée du centre d'appels, la structure retenue
pèse 0,713 et la suivante 0,281. Les données les départagent à peine&nbsp;: le
rapport le dit.

## Comment nous savons que le portage est fidèle

**Chaque modèle publié se retrouve.** Les fichiers de résultats d'origine,
pour les treize séries de référence, sont livrés avec ce paquet. Pour chaque
série et chacun des dix niveaux de recherche, `tests/test_published.py` prend
le modèle gagnant publié, calcule sa log-vraisemblance négative (NLL) aux
hyperparamètres imprimés et la compare au nombre publié. Les 130 concordent&nbsp;:
douze séries à 5e-5 nat près, la série solaire à 5e-3 nat près. Ses matrices
de covariance atteignent un conditionnement de 1e14, d'où la tolérance plus
large.

Ce test a trouvé un vrai défaut pendant le portage. Une petite gigue sur la
diagonale, relative à la diagonale moyenne, déplaçait la vraisemblance solaire
de 61 nats, parce qu'un noyau constant dominait cette diagonale. La première
factorisation n'ajoute plus de gigue, comme GPML.

**Le texte concorde.** Pour le modèle publié des passagers aériens, les quatre
phrases des composantes sont mot pour mot celles du système de 2014.

**Les gradients concordent.** La différentiation automatique remplace les
dérivées écrites à la main de GPML. Un test par différences finies les
tient à 1e-4 près.

**La recherche trouve des structures au moins aussi bonnes.** La recherche
tire ses départs au hasard et emploie un autre optimiseur (L-BFGS au lieu des
gradients conjugués de GPML), si bien qu'elle ne refait pas la recherche de
2014 chiffre pour chiffre. `scripts/replicate.py` relance la recherche
complète à dix niveaux avec les réglages publiés et compare le BIC final (plus
bas, c'est mieux)&nbsp;:

| série | BIC publié | ce portage | écart |
|---|---|---|---|
| 01-airline | 1123,7 | 1108,9 | -14,8 |
| 02-solar | -870,3 | -872,3 | -2,0 |

Les onze autres séries n'ont pas encore été relancées. À quatre heures environ
par série sur un ordinateur portable, elles attendent une machine libre.

**Ce qui n'a pas d'étalon publié.** Le préréglage de 2013 et les p-valeurs de
la vérification du modèle&nbsp;: les dépôts d'origine ne contiennent aucun
nombre auquel les comparer.

## AutoGP en Python&nbsp;: une loi a posteriori sur les structures

`autostat.smc` porte AutoGP.jl (Saad, Patton, Hoffman, Saurous et Mansinghka,
ICML 2023) vers Python et `torch`. Les recherches de 2013 et 2014 gardent la
seule structure au meilleur BIC. AutoGP garde un échantillon pondéré de
structures tiré de leur loi a posteriori, par Monte-Carlo séquentiel (SMC).
Les données entrent un dixième à la fois. Les particules sont repondérées
selon qu'elles ont bien prédit chaque lot, rééchantillonnées, puis déplacées
par des mouvements de structure (remplacement de sous-arbre, détachement et
rattachement) et par Monte-Carlo hamiltonien (HMC) sur les paramètres. Les
prévisions font la moyenne des particules, si bien que les intervalles
s'élargissent là où les données ne départagent pas les structures. De
nouvelles observations s'ajoutent plus tard sans tout réajuster.

```python
from autostat.smc import AutoGP, linear_schedule
from autostat.smc.report import describe_posterior

model = AutoGP(t, y, n_particles=8, seed=0)
model.fit_smc(linear_schedule(len(t), 0.10), n_mcmc=75, n_hmc=10)
model.structures()[:3]          # [(poids, formule), ...]
mean, bounds = model.predict(t_futur, quantiles=(0.05, 0.95))
describe_posterior(model, name="passagers", out_dir="rapport/")
```

`describe_posterior` traduit chaque structure lourde dans les noyaux de 2014
et écrit un rapport par structure, plus une page de synthèse avec les poids et
une figure du BIC de chaque structure.

**Vérifié contre l'original, nombre par nombre.** `tests/test_smc_julia.py`
charge AutoGP.jl v0.1.19 par `juliacall` et compare chaque quantité
déterministe sur la série des passagers aériens&nbsp;:

| quantité | cas | plus grand écart relatif |
|---|---|---|
| densité jointe d'une trace (a priori, bruit, vraisemblance) | 30 structures tirées par Julia | 1e-14 |
| matrice de covariance | les mêmes 30 | 1e-15 |
| densités aller et retour d'un mouvement de structure | 400 mouvements tirés par Julia (218 remplacements, 165 rattachements, 17 détachements) | 1e-15 |
| rapport de densités du mouvement | les mêmes 400 | 4e-13 |
| moyenne et covariance prédictives | 20 structures, 15 points futurs | 3e-12 |

**Ce que la comparaison a trouvé dans AutoGP.jl.** Le gradient que Gen donne
pour chaque paramètre de noyau vaut `-z`, le gradient de son a priori normal
standard, sur les 359 paramètres testés. `covariance_prior` n'est pas déclaré
différentiable (`Gen.accepts_output_grad` vaut faux), donc la vraisemblance
n'atteint jamais les pas de saute-mouton. Le test d'acceptation emploie la
densité complète, donc l'échantillonneur reste correct, mais son HMC déplace
les paramètres de noyau comme une marche aléatoire conduite par l'a priori.
Le portage reproduit ce comportement par défaut (`HMCConfig(gradient="prior")`)
et propose le comportement voulu avec `gradient="full"`. Le bruit reçoit le
gradient complet dans les deux cas, comme en Julia.

**Comparé en loi.** Les tirages aléatoires diffèrent d'un langage à l'autre,
donc des exécutions complètes ne concordent qu'en loi.
`scripts/compare_autogp.py` ajuste AutoGP.jl, le portage et le portage à
gradient complet sur les mêmes séries de référence, avec plusieurs graines
aléatoires chacun, puis note les trois avec le même code. Sur 87 exécutions,
aucune différence n'est significative. Le portage tourne environ six fois
plus vite.

## La prévision face aux méthodes de 2026

L'article de 2014 jugeait aussi son système sur l'extrapolation&nbsp;: apprendre
sur les premiers 90&nbsp;% de chacune des treize séries, prévoir les 10&nbsp;%
restants. Le dossier [benchmark](benchmark/README.md) rejoue l'épreuve. Il
recalcule d'abord le tableau de l'article à partir des prévisions publiées en
2014, à 2e-10 près, puis ajoute les méthodes de 2026 famille par famille&nbsp;:

- modèles classiques&nbsp;: ETS, ARIMA, Theta&nbsp;;
- arbres de décision boostés&nbsp;: LightGBM&nbsp;;
- réseaux profonds&nbsp;: N-HiTS, N-BEATS, PatchTST&nbsp;;
- modèles de fondation&nbsp;: Chronos-2, Chronos-Bolt, TimesFM 2.5, TabPFN-TS,
  NVIDIA KumoTabular&nbsp;;
- régression symbolique&nbsp;: Eureqa (les prévisions publiées en 2014), PySR,
  gplearn&nbsp;;
- les portages de ce dépôt.

Les méthodes à hyperparamètres libres sont réglées par validation croisée à
origine glissante sur la seule partie d'apprentissage, avec skrub et skore.
Les prévisions de 2014 finissent cinquièmes sur vingt-quatre méthodes et ne
s'effondrent jamais. Eureqa se trompe 590 fois plus que la meilleure méthode
sur une série. PySR, sur la même découpe, reste borné sur cette série (4,56).
Le tableau se trouve dans [benchmark/RESULTS.md](benchmark/RESULTS.md).

## La famille de l'Automatic Statistician

Chaque projet que nous avons trouvé, des articles d'origine à leurs
successeurs de 2026, avec ce que ce dépôt reprend de chacun. Chaque dépôt a
été vérifié sur GitHub le 7 novembre 2026.

| projet | année | ce que c'est | ici |
|---|---|---|---|
| [gp-structure-search](https://github.com/jamesrobertlloyd/gp-structure-search) | 2013 | Recherche compositionnelle de noyaux (CKS), ICML 2013. MATLAB et Python 2.7, licence MIT. | le préréglage `Experiment.cks_2013()` |
| [gpss-research](https://github.com/jamesrobertlloyd/gpss-research) | 2014 | *Automatic Bayesian Covariance Discovery* (ABCD), AAAI 2014&nbsp;: changements de régime, rapports en langue naturelle. Licence MIT. | porté en entier |
| [kmc-research](https://github.com/jamesrobertlloyd/kmc-research) | 2015 | Critique de modèle par l'écart moyen maximal (MMD), NIPS 2015. Licence MIT. | la section de vérification du modèle |
| [comp-kern](https://github.com/jamesrobertlloyd/comp-kern) | 2014 | «&nbsp;Interface simplifiée pour l'apprentissage compositionnel de noyaux&nbsp;». Sans licence. | lu, pas copié |
| [auto-stat-basic](https://github.com/jamesrobertlloyd/auto-stat-basic) | 2015 | «&nbsp;Premiers essais d'architecture de statisticien automatique&nbsp;». Sans licence. | lu, pas copié |
| [autogpc](https://github.com/charles92/autogpc) | 2016 | Recherche de noyaux pour la classification par GP (thèse de Qiurui He). Sans licence. | cité |
| [Relational Automatic Statistician](https://github.com/OpenXAIProject/Relational-Automatic-Statistician) | 2019 | Une structure partagée entre plusieurs séries (Hwang, Tong et Choi, ICML 2016). Apache 2.0. | cité |
| [autostat (sutoiku)](https://github.com/sutoiku/autostat) | 2022 | Une réimplémentation d'ABCD en Mathematica. Licence MIT. | cité |
| [autostat_gpflow](https://github.com/DavidJanz/autostat_gpflow) | 2018 | Une implémentation élémentaire sur GPflow. Licence MIT. | cité |
| [GPy-ABCD](https://github.com/T-Flet/GPy-ABCD) | 2021 | ABCD configurable sur GPy (Fletcher, Bundy et Nuamah, atelier AutoML d'ICML 2021). BSD 3 clauses. | cité |
| [AutoGP.jl](https://github.com/probsys/AutoGP.jl) | 2023 | Monte-Carlo séquentiel (SMC) sur les structures de noyaux (Saad et al., ICML 2023, arXiv 2307.09607). Julia, Apache 2.0, maintenu. | porté dans `autostat.smc`, vérifié contre lui par `juliacall` |
| [gallifrey](https://github.com/ChrisBoettner/gallifrey) | 2025 | Une réimplémentation d'AutoGP.jl en JAX pour l'astronomie (Boettner, A&A 2025, arXiv 2505.20394). Apache 2.0. | cité |
| [AutoBNN](https://github.com/tensorflow/probability/tree/main/spinoffs/autobnn) | 2024 | Réseaux de neurones bayésiens compositionnels en JAX (Google). Apache 2.0. | cité |

Trois travaux voisins n'ont aucun code public que nous ayons trouvé&nbsp;:

- BoxLM, où un grand modèle de langue (LLM) propose des programmes
  probabilistes (Li, Fox et Goodman, ICML 2024, arXiv 2402.17879)&nbsp;;
- une chaîne où un modèle de vision et de langue juge les ajustements de
  noyaux sur leurs graphiques (Lee et al., arXiv 2509.25946, 2025)&nbsp;;
- la composition de noyaux à grande échelle (Kim et Teh, AISTATS 2018, arXiv
  1706.02524).

L'adresse automaticstatistician.com sert aujourd'hui un site sans rapport avec
le projet. Le site d'origine, en ligne depuis juillet 2014, survit dans les
archives du Web.

## Installation

Python 3.11 ou plus récent. La recherche, les sources des rapports et leurs
figures ne demandent que des paquets Python. Compiler le PDF demande, pour un
rapport Markdown, [Pandoc](https://pandoc.org/installing.html) et
[LibreOffice](https://www.libreoffice.org/download/) (`md2star doctor` vérifie
les deux). Pour un rapport LaTeX, il faut une distribution TeX avec `latexmk`
(MacTeX, TeX Live ou MiKTeX). Sans eux, la source est écrite et un
avertissement nomme l'outil manquant. `--no-pdf` saute l'étape.

```bash
pip install git+https://github.com/warith-harchaoui/automatic-statistician-revisited
# ou, depuis un clone, pour développer :
pip install -r requirements-dev.txt
make check
```

Lire des fichiers Parquet demande
`pip install "automatic-statistician-revisited[parquet]"`. Les formes
explicites des composantes lisses demandent
`pip install "automatic-statistician-revisited[symbolic]"`, qui installe PySR
et, à travers lui, Julia. Sous Windows sans `make`, lancez à la main les
quatre commandes du `Makefile`.

## Utilisation

La même analyse s'ouvre de six façons. Chacune appelle `autostat.service`, si
bien que les six vérifient leurs entrées et se comportent de la même manière.

| surface | pour la lancer | pour qui |
|---|---|---|
| Bibliothèque Python | `from autostat import analyse` | un carnet de notes, un autre programme |
| Ligne de commande (Click) | `autostat data.csv ...` | un terminal, un script |
| Interface graphique (sprezzature-cli-gui) | `autostat-gui` | un formulaire dans le navigateur, sur cette machine seulement |
| API HTTP (FastAPI) | `uvicorn autostat.api:app` | un autre programme, avec OpenAPI sur `/docs` |
| Serveur MCP (FastAPI-MCP) | `autostat-mcp` | un hôte d'agents, au point d'accès `/mcp` |
| Skill d'agent | [skills/autostat/SKILL.md](skills/autostat/SKILL.md) | Claude Code ou OpenCode, aiguillés par [TRIGGERS.md](TRIGGERS.md) |

La ligne de commande lit un fichier CSV ou Parquet&nbsp;:

```bash
autostat data.csv --time date --value y --out rapport/
autostat data.csv --time date --value y --format latex --out rapport/
autostat data.csv --time date --value y --preset cks-2013 --max-depth 4
autostat data.csv --time date --value y --max-depth 20 --patience 2
autostat data.csv --time date --value y --symbolic --out rapport/
autostat data.csv --time date --value y --no-check --no-pdf --json
```

La configuration publiée cherche `--max-depth` niveaux, pas un de moins.
`--patience 2` arrête la recherche quand le BIC n'a pas baissé deux niveaux de
suite. `--symbolic` demande à PySR une formule pour chaque composante lisse,
environ une minute chacune.

L'interface Python expose chaque étape&nbsp;: `prepare` (dates en années
décimales, longues séries regroupées par paquets), `search` avec une
`Experiment`, `describe` pour le rapport, `fit` et `posterior` pour un modèle
seul.

L'API et le serveur MCP partagent une même application&nbsp;: `autostat-mcp`
sert les routes HTTP et `/mcp` sur le port 8000. Les outils MCP s'appellent
`list_series`, `describe_published`, `analyse_series` et `get_report`. L'API
borne une recherche à 1&nbsp;000 points et à six niveaux, vingt avec une patience.
Les deux écoutent sur 127.0.0.1 par défaut et n'ont aucune authentification&nbsp;:
placez-en une devant elles avant de les exposer.

## Limites

- **La taille.** L'inférence exacte par GP coûte un temps cubique en le
  nombre de points. `prepare` regroupe toute série de plus de 1&nbsp;000 points
  en 1&nbsp;000 paquets moyennés, la borne de l'original, si bien qu'une
  structure plus fine qu'un paquet passe inaperçue. La recherche complète à
  dix niveaux sur les 144 points des passagers aériens a pris 4,1 heures avec
  six processus sur un ordinateur portable partagé. `--max-depth 3` donne un
  premier aperçu en quelques minutes.
- **Une seule dimension d'entrée dans le rapport.** Les noyaux et la recherche
  acceptent plusieurs dimensions d'entrée, mais la description en anglais
  suppose une seule dimension.
- **La régression seulement.** La variante pour la classification (autogpc)
  n'est pas portée.
- **Le préréglage de 2013 est une approximation.** Il emploie la grammaire et
  les réglages de 2013 avec l'initialisation des paramètres de 2014. Ses deux
  départs aléatoires d'écart 4 manquent souvent une période&nbsp;: sur la série
  tendance plus cycle de `tests/test_search.py`, il s'arrête à `SE` là où le
  préréglage de 2014 trouve le cycle.
- **La grammaire d'AutoGP lui appartient.** `autostat.smc` garde les noyaux de
  base d'AutoGP.jl (linéaire, gamma-exponentiel, périodique) et son changement
  de régime, qui diffèrent de la grammaire de 2014. Ses rapports traduisent
  chaque structure dans les noyaux de 2014.
- **La vérification du modèle est un crible.** Une petite p-valeur dit que les
  données contredisent l'a priori d'une composante sur une statistique. Elle
  ne dit pas quel modèle a raison.

## Citer

Citez les travaux d'origine, puis ce portage si vous l'employez&nbsp;:

```bibtex
@inproceedings{duvenaud2013structure,
  title     = {Structure Discovery in Nonparametric Regression through Compositional Kernel Search},
  author    = {Duvenaud, David and Lloyd, James Robert and Grosse, Roger and Tenenbaum, Joshua B. and Ghahramani, Zoubin},
  booktitle = {Proceedings of the 30th International Conference on Machine Learning},
  year      = {2013},
  eprint    = {1302.4922},
  archivePrefix = {arXiv}
}

@inproceedings{lloyd2014automatic,
  title     = {Automatic Construction and Natural-Language Description of Nonparametric Regression Models},
  author    = {Lloyd, James Robert and Duvenaud, David and Grosse, Roger and Tenenbaum, Joshua B. and Ghahramani, Zoubin},
  booktitle = {Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence},
  year      = {2014},
  eprint    = {1402.4304},
  archivePrefix = {arXiv}
}

@inproceedings{lloyd2015statistical,
  title     = {Statistical Model Criticism using Kernel Two Sample Tests},
  author    = {Lloyd, James Robert and Ghahramani, Zoubin},
  booktitle = {Advances in Neural Information Processing Systems 28},
  year      = {2015}
}
```

[CITATION.cff](CITATION.cff) décrit ce dépôt pour les outils de citation.

## Licence et crédits

Le logiciel est de Warith Harchaoui, sous licence MIT ([LICENSE](LICENSE)).
La conception portée ici, les données de référence et les résultats publiés
sont de James Robert Lloyd, David Duvenaud et Roger Grosse, sous la licence
MIT reproduite dans [LICENSE-ORIGINAL](LICENSE-ORIGINAL). `autostat.smc` porte
AutoGP.jl, (c) 2023 Google LLC, sous la licence Apache 2.0 reproduite dans
[LICENSE-AUTOGP](LICENSE-AUTOGP). La plupart des séries de référence viennent
de la *Time Series Data Library* de Rob Hyndman. [PORTING.md](PORTING.md)
liste ce qui a changé dans le portage et les défauts trouvés en chemin dans
l'original. Le logo et ses variantes se trouvent dans [assets/](assets/).

Les rapports qu'écrit le logiciel sont signés «&nbsp;Some Math empowered by some
AI guided by some researchers&nbsp;».
