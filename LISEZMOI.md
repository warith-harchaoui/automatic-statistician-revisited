<p align="center"><img src="assets/logo-288.png" width="200" alt="Un statisticien à son bureau, l'index levé"></p>

# Automatic Statistician, revisité

[Read in English](README.md)

On lui donne une série temporelle. Il cherche, dans une grammaire de noyaux
de processus gaussiens (GP, *Gaussian process*), la structure qui explique le
mieux la série. Il découpe cette structure en composantes additives, décrit
chacune en anglais, confronte chacune à ses propres hypothèses et rédige le
tout sous forme de rapport.

C'est l'Automatic Statistician de Duvenaud, Lloyd, Grosse, Tenenbaum et
Ghahramani (ICML 2013, AAAI 2014), porté de son code d'origine en Python 2.7
et MATLAB vers Python 3 et `torch`. Le code d'origine date de 2015 et 2017 :
il réclame MATLAB, la boîte à outils GPML (*Gaussian Processes for Machine
Learning*) et un Python en fin de vie depuis 2020. Ce dépôt s'installe avec
`pip install` et retrouve la vraisemblance de chaque modèle publié par le
système de 2014.

```python
from autostat import analyse

result, report = analyse(series, name="ventes mensuelles", out_dir="rapport/")
print(result.best.kernel.structure())
# par exemple WN + SE + Lin x Per x SE
```

```bash
autostat ventes.csv --time mois --value unites --out rapport/
```

Le fichier `rapport/report.md` contient alors le résumé, une section par
composante avec sa figure, la formule trouvée par la recherche (l'expression
du modèle avec ses paramètres ajustés plus une forme explicite pour chaque
composante polynomiale), puis le tableau de vérification du modèle ;
`rapport/report.pdf` est le même document compilé par
[md2star](https://github.com/warith-harchaoui/md2star). Avec `--format latex`
(en Python : `report_format="latex"`), le rapport devient `report.tex`,
compilé par `latexmk`, comme l'écrivait le système de 2014. Les figures sont
tracées par [sprezzature-figures](https://github.com/warith-harchaoui/sprezzature-figures).
L'exemple de la série des passagers aériens, dans [EXEMPLES.md](EXEMPLES.md),
montre le résultat ; [examples/](examples/) contient le script de chaque
exemple et le rapport qu'il a écrit, dans les deux formats.

## Ce que fait la recherche

Un noyau est une formule qui dit à quel point deux points d'une série bougent
ensemble. La recherche assemble des noyaux à partir de quelques briques :

| noyau de base | ce qu'il décrit |
|---|---|
| `SE`, exponentielle quadratique | une fonction lisse |
| `Per`, périodique | un motif qui se répète |
| `Lin`, linéaire | une tendance, ou une amplitude qui grandit |
| `C`, constante | un décalage |
| `WN`, bruit blanc | le bruit de mesure |
| `RQ`, quadratique rationnelle | des variations lisses à plusieurs échelles (préréglage 2013) |

Elle les combine par `+` (deux effets qui s'ajoutent), `x` (un effet qui en
module un autre), `CP` (un point de rupture : une structure, puis une autre)
et `CW` (une fenêtre : une structure qui vaut un temps). À chaque niveau, elle
essaie toutes les expressions à une réécriture de la meilleure courante, ajuste
chacune depuis neuf départs aléatoires et garde celle dont le critère
d'information bayésien (BIC, *Bayesian information criterion*) est le plus
bas. Dix niveaux plus tard, la série des passagers aériens donne
`SE + Lin + WN x Lin + SE x Per x Lin`, que le rapport lit ainsi :

1. Une fonction linéairement croissante.
2. Une fonction approximativement périodique, de période 1,0 an, dont
   l'amplitude croît linéairement.
3. Une fonction lisse.
4. Un bruit non corrélé dont l'écart type croît linéairement.

Le rapport reste en anglais : ses phrases sont celles du système d'origine,
reprises mot pour mot.

## Comment nous savons que le portage est fidèle

**Chaque modèle publié se retrouve.** Les fichiers de résultats d'origine,
pour les treize séries de référence, sont livrés avec le paquet. Pour chaque
série et chacun des dix niveaux de la recherche publiée,
`tests/test_published.py` prend le modèle gagnant, calcule sa
log-vraisemblance négative (NLL) aux hyperparamètres imprimés et la compare
au chiffre publié. Les 130 concordent : douze séries à 5e-5 nat près, la
série solaire (dont les matrices de covariance atteignent un conditionnement
de 1e14) à 5e-3 nat près.

Ce test a trouvé un vrai défaut pendant le portage. Un petit terme ajouté à
la diagonale, proportionnel à la diagonale moyenne, déplaçait la
vraisemblance solaire de 61 nats : un noyau constant dominait cette
diagonale. La première factorisation n'ajoute désormais rien, comme GPML.

**Le texte concorde.** Pour le modèle publié des passagers aériens, les
quatre résumés de composantes sont mot pour mot ceux du système de 2014.

**Les gradients concordent.** La différentiation automatique remplace les
dérivées écrites à la main de GPML ; un test par différences finies les tient
à 1e-4.

**La recherche trouve des structures au moins aussi bonnes.** Elle emploie
des départs aléatoires et un autre optimiseur (L-BFGS au lieu des gradients
conjugués de GPML) : elle ne refait donc pas la recherche de 2014 chiffre
pour chiffre. `scripts/replicate.py` relance la recherche complète à dix
niveaux avec les réglages publiés et compare le BIC final (plus bas vaut
mieux) :

| série | BIC publié | ce portage | écart |
|---|---|---|---|
| 01-airline | 1123,7 | 1108,9 | -14,8 |
| 02-solar | -870,3 | -872,3 | -2,0 |

Les onze autres séries tournaient encore le 7 novembre 2026 ; ce tableau
s'allonge à mesure qu'elles finissent.

**Ce qui n'a pas d'étalon publié.** Le préréglage 2013 et les p-valeurs de
vérification : les dépôts d'origine ne contiennent aucun chiffre auquel les
comparer.

## AutoGP en Python : une loi a posteriori sur les structures

`autostat.smc` porte AutoGP.jl (Saad, Patton, Hoffman, Saurous et
Mansinghka, ICML 2023) en Python et `torch`. Les recherches de 2013 et de
2014 gardent la seule structure de meilleur BIC ; celle-ci garde un
échantillon pondéré de structures tiré de leur loi a posteriori, par
Monte-Carlo séquentiel (SMC) : les données entrent par dixièmes, les
particules sont repondérées selon leur prévision de chaque lot,
rééchantillonnées, puis déplacées par des mouvements de structure
(remplacement de sous-arbre, détachement et rattachement) et par Monte-Carlo
hamiltonien (HMC) sur les paramètres. Les prévisions moyennent les
particules : les intervalles s'élargissent là où les données ne départagent
pas les structures. On peut ajouter des observations plus tard sans tout
réajuster. Les deux recherches précédentes restent telles quelles.

```python
from autostat.smc import AutoGP, linear_schedule

model = AutoGP(t, y, n_particles=8, seed=0)
model.fit_smc(linear_schedule(len(t), 0.10), n_mcmc=75, n_hmc=10)
model.structures()[:3]          # [(poids, formule), ...]
mean, bounds = model.predict(t_future, quantiles=(0.05, 0.95))
model.add_data(t_new, y_new)    # puis model.mcmc_structure(...)
```

**Vérifié contre l'original, chiffre pour chiffre.** `tests/test_smc_julia.py`
charge AutoGP.jl v0.1.19 par `juliacall` et compare toutes les quantités
déterministes sur la série des passagers aériens :

| quantité | cas | plus grand écart relatif |
|---|---|---|
| log-densité jointe d'une trace (a priori, bruit, vraisemblance) | 30 structures tirées par Julia | 1e-14 |
| matrice de covariance | les mêmes 30 | 1e-15 |
| densités aller et retour d'un mouvement de structure | 400 mouvements tirés par Julia (218 remplacements, 165 rattachements, 17 détachements) | 1e-15 |
| rapport des densités du modèle pour ce mouvement | les mêmes 400 | 4e-13 |
| moyenne et covariance prédictives | 20 structures, 15 points futurs | 3e-12 |

**Ce que la comparaison a révélé dans AutoGP.jl.** Le gradient que Gen
calcule pour chaque paramètre de noyau vaut `-z`, le gradient de sa loi a
priori normale, sur les 359 paramètres testés : `covariance_prior` n'est pas
déclarée dérivable (`Gen.accepts_output_grad` vaut faux), si bien que la
vraisemblance n'atteint jamais les pas de saute-mouton. Le test
d'acceptation utilise la densité complète, donc l'échantillonneur reste
exact, mais son HMC déplace les paramètres de noyau comme une marche
aléatoire guidée par l'a priori. Le portage reproduit ce comportement par
défaut (`HMCConfig(gradient="prior")`) et propose le comportement voulu avec
`gradient="full"`. Le bruit reçoit le gradient complet dans les deux cas,
comme en Julia.

**Comparé en loi.** Les tirages aléatoires diffèrent entre les deux
langages : des exécutions complètes ne peuvent concorder que
statistiquement. `scripts/compare_autogp.py` ajuste AutoGP.jl, le portage et
le portage à gradient complet sur les mêmes séries, avec plusieurs graines
chacun ; le même code note les trois ; les résultats vont dans
`results/autogp_comparison.parquet`.

## Prévoir face aux méthodes de 2026

L'article de 2014 jugeait aussi son système en extrapolation : apprendre sur
les premiers 90 % de chacune des treize séries, prévoir les 10 % restants.
Le dossier [benchmark](benchmark/README.md) refait cette comparaison. Il
recalcule d'abord le tableau de l'article à partir des prévisions publiées
en 2014 (à 2e-10 près), puis ajoute les méthodes de 2026, famille par
famille : classiques (ETS, ARIMA, Theta), LightGBM, réseaux profonds
(N-HiTS, N-BEATS, PatchTST), modèles de fondation (Chronos-2, Chronos-Bolt,
TimesFM 2.5, TabPFN-TS, NVIDIA KumoTabular) et les deux portages de ce
dépôt. Les méthodes à hyper-paramètres libres sont réglées par validation
croisée à origine glissante sur la seule partie d'apprentissage, avec skrub
et skore. Le tableau du moment est dans
[benchmark/RESULTS.md](benchmark/RESULTS.md).

## La famille Automatic Statistician

Tous les projets que nous avons trouvés, des articles d'origine aux
successeurs de 2026, avec ce que ce dépôt reprend de chacun. Chaque dépôt a
été vérifié sur GitHub le 7 novembre 2026.

| projet | année | ce que c'est | ici |
|---|---|---|---|
| [gp-structure-search](https://github.com/jamesrobertlloyd/gp-structure-search) | 2013 | Recherche compositionnelle de noyaux (CKS, *compositional kernel search*), ICML 2013. MATLAB et Python 2.7, licence MIT. | préréglage `Experiment.cks_2013()` |
| [gpss-research](https://github.com/jamesrobertlloyd/gpss-research) | 2014 | Découverte automatique de covariance bayésienne (ABCD, *Automatic Bayesian Covariance Discovery*), AAAI 2014 : ruptures, rapports en langue naturelle. Licence MIT. | porté en entier |
| [kmc-research](https://github.com/jamesrobertlloyd/kmc-research) | 2015 | Critique de modèle par l'écart moyen maximal (MMD, *maximum mean discrepancy*), NIPS 2015. Licence MIT. | la section de vérification |
| [comp-kern](https://github.com/jamesrobertlloyd/comp-kern) | 2014 | « Interface simplifiée à l'apprentissage compositionnel de noyaux ». Sans licence. | lu, rien copié |
| [auto-stat-basic](https://github.com/jamesrobertlloyd/auto-stat-basic) | 2015 | « Premiers essais d'architecture d'Automatic Statistician ». Sans licence. | lu, rien copié |
| [autogpc](https://github.com/charles92/autogpc) | 2016 | Recherche de noyaux pour la classification par GP (thèse de Qiurui He). Sans licence. | cité |
| [Relational Automatic Statistician](https://github.com/OpenXAIProject/Relational-Automatic-Statistician) | 2019 | Structure partagée entre plusieurs séries (Hwang, Tong et Choi, arXiv 1511.08343). Apache 2.0. | cité |
| [autostat (sutoiku)](https://github.com/sutoiku/autostat) | 2022 | Réimplémentation d'ABCD en Mathematica. Licence MIT. | cité |
| [autostat_gpflow](https://github.com/DavidJanz/autostat_gpflow) | 2018 | Implémentation élémentaire sur GPflow. Licence MIT. | cité |
| [GPy-ABCD](https://github.com/T-Flet/GPy-ABCD) | 2021 | ABCD configurable sur GPy (Fletcher, Bundy et Nuamah, atelier AutoML d'ICML 2021). BSD 3 clauses. | cité |
| [AutoGP.jl](https://github.com/probsys/AutoGP.jl) | 2023 | Monte-Carlo séquentiel (SMC, *sequential Monte Carlo*) sur les structures de noyaux (Saad et al., ICML 2023, arXiv 2307.09607). Julia, Apache 2.0, maintenu. | porté : `autostat.smc`, vérifié contre l'original par `juliacall` |
| [gallifrey](https://github.com/ChrisBoettner/gallifrey) | 2025 | Réimplémentation JAX d'AutoGP.jl pour l'astronomie (Boettner, A&A 2025, arXiv 2505.20394). Apache 2.0. | cité |
| [AutoBNN](https://github.com/tensorflow/probability/tree/main/spinoffs/autobnn) | 2024 | Réseaux de neurones bayésiens compositionnels en JAX (Google). Apache 2.0. | cité |

Trois travaux voisins n'ont pas de code public à notre connaissance : BoxLM,
où un grand modèle de langue (LLM, *large language model*) propose des
programmes probabilistes (Li, Fox et Goodman, ICML 2024, arXiv 2402.17879) ;
une chaîne fondée sur un modèle vision-langage qui juge les ajustements sur
leurs graphiques (Lee et al., arXiv 2509.25946, 2025) ; la composition de
noyaux à grande échelle (Kim et Teh, AISTATS 2018, arXiv 1706.02524). Le site
automaticstatistician.com est tenu indépendamment des auteurs d'origine et
ne propose aucun code.

## Installation

Python 3.11 ou plus récent. La recherche, les sources des rapports et
leurs figures ne demandent que des roues Python. Compiler le PDF demande,
pour un rapport Markdown, [Pandoc](https://pandoc.org/installing.html) et
[LibreOffice](https://www.libreoffice.org/download/) (`md2star doctor` vérifie
les deux) ; pour un rapport LaTeX, une distribution TeX avec `latexmk`
(MacTeX, TeX Live ou MiKTeX). Sans eux, la source est écrite et un
avertissement nomme l'outil manquant ; `--no-pdf` saute l'étape.

```bash
pip install git+https://github.com/warith-harchaoui/automatic-statistician-revisited
# ou, depuis un clone, pour contribuer :
pip install -r requirements-dev.txt
make check
```

La lecture des fichiers Parquet demande
`pip install "automatic-statistician-revisited[parquet]"`. Sous Windows sans
`make`, on lance à la main les quatre commandes du `Makefile`.

## Utilisation

La même analyse s'atteint de six façons. Chacune appelle `autostat.service` :
toutes valident et se comportent de même.

| Surface | Lancement | Pour |
|---|---|---|
| Bibliothèque Python | `from autostat import analyse` | carnets et autre code |
| Ligne de commande (Click) | `autostat donnees.csv ...` | un terminal, un script |
| Interface graphique (sprezzature-cli-gui) | `autostat-gui` | un formulaire dans le navigateur, sur cette machine seulement |
| API HTTP (FastAPI) | `uvicorn autostat.api:app` | un autre programme ; OpenAPI sur `/docs` |
| Serveur MCP (FastAPI-MCP) | `autostat-mcp` | un hôte d'agents ; point d'accès `/mcp` |
| Compétence d'agent | [skills/autostat/SKILL.md](skills/autostat/SKILL.md) | Claude Code, OpenCode ; aiguillage dans [TRIGGERS.md](TRIGGERS.md) |

La ligne de commande lit un fichier CSV ou Parquet :

```bash
autostat donnees.csv --time date --value y --out rapport/
autostat donnees.csv --time date --value y --format latex --out rapport/
autostat donnees.csv --time date --value y --preset cks-2013 --max-depth 4
autostat donnees.csv --time date --value y --max-depth 20 --patience 2
autostat donnees.csv --time date --value y --no-check --no-pdf --json
```

La configuration publiée cherche toujours `--max-depth` niveaux.
`--patience 2` s'arrête quand le BIC n'a pas baissé deux niveaux de suite ;
le rapport étudie ces deux niveaux : chaque rapport a une section sur le
chemin de recherche avec la courbe du BIC, la structure de chaque niveau et
ce qu'un arrêt plus précoce aurait gardé.

L'interface Python donne accès à chaque étape : `prepare` (dates en années
décimales, séries longues regroupées), `search` avec une `Experiment`,
`describe` pour le rapport, `fit` et `posterior` pour un modèle seul.
[EXEMPLES.md](EXEMPLES.md) rassemble des recettes prêtes à lancer pour
chaque surface.

L'API et le serveur MCP partagent une application : `autostat-mcp` sert les
routes HTTP et `/mcp` sur le port 8000. Les outils MCP sont `list_series`,
`describe_published`, `analyse_series` et `get_report` ; l'API borne une
recherche à six niveaux et 1 000 points. Les deux écoutent 127.0.0.1 par
défaut, sans authentification : placez-les derrière une avant de les
exposer.

## Limites

- **La taille.** L'inférence exacte par GP coûte un temps cubique en nombre
  de points. `prepare` regroupe toute série de plus de 1 000 points en 1 000
  classes, la borne du système d'origine. Une structure plus fine qu'une
  classe passe inaperçue. La recherche complète à dix niveaux sur les 144
  points des passagers aériens a pris 4,1 heures avec six processus sur un
  ordinateur portable partagé. `--max-depth 3` donne un premier aperçu en
  quelques minutes.
- **Une seule dimension d'entrée dans le rapport.** Les noyaux et la
  recherche acceptent plusieurs dimensions ; la description en anglais en
  suppose une.
- **Régression seulement.** La variante de classification (autogpc) n'est pas
  portée.
- **Le préréglage 2013 est une approximation.** Il reprend la grammaire et
  les réglages de 2013 avec l'initialisation des paramètres de 2014. Ses deux
  départs aléatoires, d'écart 4, manquent souvent une période : sur la série
  tendance plus cycle de `tests/test_search.py`, il s'arrête sur `SE` là où le
  préréglage 2014 trouve le cycle.
- **AutoGP a sa propre grammaire.** `autostat.smc` garde les noyaux de base
  d'AutoGP.jl (linéaire, gamma-exponentiel, périodique) et sa rupture, qui
  diffèrent de la grammaire de 2014 ; son rapport en anglais n'est pas encore
  écrit.
- **La vérification est un crible.** Une p-valeur faible dit que les données
  contredisent l'hypothèse d'une composante sur une statistique. Elle ne dit
  pas quel modèle est le bon.

## Citer

Citez les travaux d'origine, puis ce portage si vous l'employez. Les entrées
BibTeX figurent dans [README.md](README.md#citing) ;
[CITATION.cff](CITATION.cff) décrit ce dépôt pour les outils de citation.

## Licence et crédits

Licence MIT ([LICENSE](LICENSE)). La conception portée ici, les données de
référence et les résultats publiés sont de James Robert Lloyd, David Duvenaud
et Roger Grosse, sous la licence MIT reproduite dans
[LICENSE-ORIGINAL](LICENSE-ORIGINAL). `autostat.smc` porte AutoGP.jl, (c) 2023
Google LLC, sous la licence Apache 2.0 reproduite dans
[LICENSE-AUTOGP](LICENSE-AUTOGP). La plupart des séries de référence
viennent de la *Time Series Data Library* de Rob Hyndman.
[PORTING.md](PORTING.md) recense ce qui a changé au portage et les défauts du
code d'origine relevés en chemin.
