# Rapport du modèle machine learning amélioré

## Résumé

La première baseline était effectivement trop faible : ROC-AUC `0,602`, F1
`0,292` et régression incapable de battre la médiane. Une analyse d'ablation a
montré que le principal problème n'était pas l'échantillon, mais l'absence de
contexte opérationnel récent et la dérive très forte entre les mois.

La version améliorée utilise CatBoost et 68 features historiques sans fuite.
Sur le même test chronologique, elle atteint une ROC-AUC de `0,657`, un F1 de
`0,362` et un rappel de `0,614`. Le gain est mesurable, mais le modèle reste
insuffisant pour une décision opérationnelle à forts enjeux.

La régression gagne seulement 0,57 minute de MAE par rapport à la médiane. Elle
reste expérimentale et ne doit pas être présentée comme une estimation précise.

## Objectifs

Le pipeline répond à trois questions avant le départ prévu :

1. le vol aura-t-il au moins 15 minutes de retard à l'arrivée ?
2. s'il est en retard, combien de minutes peut-on estimer ?
3. s'il est en retard, quelles causes officielles sont les plus plausibles ?

Les résultats sont produits avec :

```bash
uv run train-flight-models --sample-fraction 0.1
```

## Données et découpage

Le CSV complet de 7 079 081 lignes est parcouru avec Polars, jamais avec Spark.
Les vols annulés, déroutés ou sans retard d'arrivée connu sont exclus des cibles.
L'échantillonnage déterministe retient 696 596 vols achevés.

| Ensemble | Mois | Vols | Retards ≥ 15 min | Taux de retard |
|---|---|---:|---:|---:|
| Entraînement | janvier à août | 462 156 | 107 561 | 23,27 % |
| Validation | septembre à octobre | 118 740 | 17 013 | 14,33 % |
| Test | novembre à décembre | 115 700 | 20 868 | 18,04 % |

Le test n'est utilisé par le code ni pour ajuster CatBoost, ni pour choisir le
seuil. Les performances de plusieurs configurations ont cependant été consultées
sur ce benchmark pendant l'audit. Novembre-décembre permet donc une comparaison
cohérente avec l'ancienne baseline, mais ne constitue plus un holdout humainement
vierge. Une année 2025 séparée sera nécessaire pour l'estimation finale.

## Pourquoi la première baseline échouait

### 1. Dérive temporelle

Le taux de retard du CSV complet passe de 13,20 % en octobre à 14,67 % en
novembre puis 21,15 % en décembre. Le seuil optimal de l'ancienne baseline était
`0,158` sur septembre-octobre mais environ `0,115` a posteriori sur le test.

L'échantillon de 10 % reproduit correctement ces taux. Le problème ne venait
donc pas d'un échantillonnage biaisé.

### 2. Features trop statiques

La compagnie, la route et l'horaire expliquent un risque moyen, mais ne disent
pas si le réseau ou un aéroport traverse actuellement une période perturbée.
CatBoost appliqué aux seules données statiques n'a fait progresser la ROC-AUC
que de `0,602` à `0,608`.

### 3. Modèle linéaire trop limité

Le one-hot linéaire ne représentait pas correctement les interactions comme
compagnie × route × heure ou aéroport × heure. CatBoost traite directement les
catégories, le numéro de vol et leurs combinaisons.

### 4. Régression très bruitée

Une fois le seuil de 15 minutes franchi, la distribution est très asymétrique et
contient des retards extrêmes. Les seules données de planning expliquent peu leur
amplitude. La médiane de 43 minutes est donc difficile à battre.

### 5. Causes rares ou indisponibles avant le vol

La météo exacte et l'état de la rotation précédente ne figurent pas parmi les
entrées pré-départ. `security` ne possède que 86 cas positifs dans le test, ce qui
rend sa précision instable malgré une ROC-AUC supérieure à 0,5.

## Features ajoutées

### Planning

- jour du mois, jour de l'année, semaine et week-end ;
- encodages cycliques du mois, du jour et des horaires ;
- horaires prévus bruts, durée prévue et distance ;
- compagnie, numéro de vol, origine, destination, États et route ;
- compagnie-route, heure de départ/arrivée et aéroport-heure.

### Historiques sans fuite

Les fenêtres utilisent uniquement les dates strictement antérieures au jour du
vol. Elles sont calculées avant l'échantillonnage sur tous les vols disponibles.

| Niveau | Fenêtres |
|---|---|
| Réseau complet | 1, 3, 7 et 28 jours |
| Compagnie | 1, 7 et 28 jours |
| Aéroport d'origine | 1, 3, 14 et 28 jours |
| Aéroport de destination | 1, 3, 14 et 28 jours |
| Route | 7 et 28 jours |

Chaque fenêtre fournit le taux de retard, le taux de perturbation incluant les
annulations et déroutements, le volume de vols et la gravité moyenne des retards.
Cela représente 68 features historiques.

Le test automatisé vérifie par exemple que les features du 18 avril sont égales
aux résultats du 17 avril et que le 1er janvier ne possède aucun historique.

Pendant le backtest, un vol de décembre peut utiliser les jours de test déjà
terminés, exactement comme un système actualisé quotidiennement. Il ne peut
jamais utiliser son propre jour ou une date future. En production, ces profils
devront donc être rafraîchis quotidiennement.

## Ablation des améliorations

Toutes les lignes utilisent le même échantillon et le même benchmark
novembre-décembre. Le seuil de chaque configuration est choisi uniquement sur
septembre-octobre. Cette table est une analyse rétrospective, pas une nouvelle
estimation indépendante de généralisation.

| Configuration | ROC-AUC | Average precision | Précision | Rappel | F1 |
|---|---:|---:|---:|---:|---:|
| SGD one-hot statique | 0,602 | 0,237 | 0,250 | 0,350 | 0,292 |
| CatBoost avec planning enrichi | 0,608 | 0,240 | 0,236 | 0,536 | 0,328 |
| + historiques réseau, compagnie et aéroports | 0,650 | 0,285 | 0,253 | 0,614 | 0,358 |
| + historique de route, configuration finale | **0,657** | **0,292** | **0,256** | **0,614** | **0,362** |

Les historiques expliquent donc l'essentiel du gain. Le changement d'algorithme
seul ne suffisait pas.

Les historiques de gravité réduisaient la ROC-AUC du classifieur principal. Ils
ont été retirés de ce modèle, mais conservés pour la régression et les causes où
ils améliorent les métriques.

## Classification finale

| Métrique | Validation | Test |
|---|---:|---:|
| ROC-AUC | 0,660 | 0,657 |
| Average precision | 0,239 | 0,292 |
| Balanced accuracy | 0,617 | 0,611 |
| Précision | 0,214 | 0,256 |
| Rappel | 0,603 | 0,614 |
| F1 | 0,316 | 0,362 |
| Log loss | 0,397 | 0,449 |
| Brier score | 0,120 | 0,141 |

Le seuil sélectionné est `0.194229807`. La matrice de confusion du test est :

| | Prédit sans retard | Prédit en retard |
|---|---:|---:|
| Réel sans retard | 57 698 | 37 134 |
| Réel en retard | 8 059 | 12 809 |

Le seuil favorise le rappel : 61,4 % des retards sont détectés, mais seulement
25,6 % des alertes correspondent réellement à un retard. Le modèle signale
43,2 % des vols du test.

Une stratégie toujours positive aurait un F1 de `0,306`. Une stratégie toujours
négative aurait 82 % d'exactitude mais un F1 nul. L'exactitude brute serait donc
trompeuse pour comparer ces modèles.

## Régression conditionnelle

L'évaluation porte sur les 20 868 vols retardés du test.

| Modèle | MAE | RMSE | R² |
|---|---:|---:|---:|
| CatBoost avec gravité historique | 43,47 min | 105,45 min | -0,074 |
| Médiane d'entraînement de 43 min | 44,04 min | 104,89 min | -0,062 |

Le gain de MAE est de 0,57 minute, soit environ 1,3 %. La RMSE et le R² sont
moins bons que la médiane. Il n'est donc pas honnête de présenter ce modèle comme
une bonne estimation individuelle du nombre de minutes.

## Causes conditionnelles

| Cause | Support | Précision | Rappel | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `carrier` | 11 528 | 0,573 | 0,952 | 0,715 | 0,640 |
| `weather` | 1 198 | 0,205 | 0,331 | 0,253 | 0,733 |
| `nas` | 10 312 | 0,526 | 0,927 | 0,671 | 0,658 |
| `security` | 86 | 0,019 | 0,163 | 0,035 | 0,703 |
| `late_aircraft` | 9 871 | 0,613 | 0,878 | 0,722 | 0,775 |

Les probabilités sont conditionnelles au fait que le vol soit retardé. Plusieurs
causes peuvent être proposées simultanément. `security` doit rester accompagné
d'un avertissement ou être masqué dans l'interface à cause de sa précision.

## Features les plus importantes

Les premières importances CatBoost du classifieur sont :

1. `arrival_hour` ;
2. `op_unique_carrier` ;
3. `departure_time_sin` ;
4. `month_category` ;
5. `day_of_week_category` ;
6. `departure_hour` ;
7. `origin_disruption_rate_1d` ;
8. `carrier_disruption_rate_1d` ;
9. `dest_disruption_rate_1d` ;
10. `route_delay_rate_7d`.

Cette hiérarchie confirme que l'état récent des opérations apporte un signal que
le planning seul ne contenait pas. Une importance ne démontre toutefois pas une
relation causale.

## Artefacts

```text
models/
├── flight_delay_models.joblib
├── training_metrics.json
└── feature_importance.csv
```

L'artefact contient les modèles CatBoost et les profils historiques les plus
récents pour une future prédiction. Ces fichiers sont ignorés par Git et doivent
être régénérés. Un fichier Joblib ne doit être chargé que s'il provient d'une
source fiable.

## Limites et prochaine amélioration utile

Le modèle amélioré est meilleur, mais une ROC-AUC de 0,657 et une précision de
0,256 restent modestes. Le CSV atteint maintenant sa limite informative.

Les prochains gains sérieux nécessitent :

- des prévisions météo par aéroport connues au moment de la prédiction ;
- l'état des opérations du jour même deux ou trois heures avant le départ ;
- la rotation précédente de l'appareil, absente faute de numéro de queue ;
- une actualisation quotidienne des profils historiques ;
- davantage de données temporelles que la seule année 2024 ;
- une période séparée pour régler les hyperparamètres et les seuils.
- un véritable holdout 2025 jamais consulté pendant le développement.

Sans ces informations, augmenter seulement la complexité du modèle risque surtout
d'améliorer l'entraînement sans généraliser aux mois futurs.
