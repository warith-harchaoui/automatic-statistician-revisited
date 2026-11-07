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
composante avec sa figure, puis le tableau de vérification du modèle.
L'exemple de la série des passagers aériens, dans [EXEMPLES.md](EXEMPLES.md),
montre le résultat.

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
| [AutoGP.jl](https://github.com/probsys/AutoGP.jl) | 2023 | Monte-Carlo séquentiel (SMC, *sequential Monte Carlo*) sur les structures de noyaux (Saad et al., ICML 2023, arXiv 2307.09607). Julia, Apache 2.0, maintenu. | cité |
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

Python 3.11 ou plus récent. Les dépendances sont des roues Python
disponibles sur toutes les plateformes : aucun paquet système n'est requis
sous macOS, Ubuntu ou Windows.

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

La ligne de commande lit un fichier CSV ou Parquet :

```bash
autostat donnees.csv --time date --value y --out rapport/
autostat donnees.csv --time date --value y --preset cks-2013 --max-depth 4
autostat donnees.csv --time date --value y --no-check --json
```

L'interface Python donne accès à chaque étape : `prepare` (dates en années
décimales, séries longues regroupées), `search` avec une `Experiment`,
`describe` pour le rapport, `fit` et `posterior` pour un modèle seul.
[EXEMPLES.md](EXEMPLES.md) rassemble des recettes prêtes à lancer.

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
[LICENSE-ORIGINAL](LICENSE-ORIGINAL). La plupart des séries de référence
viennent de la *Time Series Data Library* de Rob Hyndman.
[PORTING.md](PORTING.md) recense ce qui a changé au portage et les défauts du
code d'origine relevés en chemin.
