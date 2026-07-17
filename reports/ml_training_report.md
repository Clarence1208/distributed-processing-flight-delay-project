# Rapport de la baseline machine learning

## Objectif

La prédiction est effectuée avant le départ prévu. Le pipeline répond à trois
questions distinctes :

1. le vol aura-t-il au moins 15 minutes de retard à l'arrivée ?
2. s'il est en retard, combien de minutes peut-on estimer ?
3. s'il est en retard, quelles causes officielles sont les plus plausibles ?

Les résultats présentés ici proviennent de la commande suivante :

```bash
uv run train-flight-models --sample-fraction 0.1
```

## Préparation des données

Le fichier complet de 7 079 081 lignes est parcouru avec Polars en mode
streaming. Spark n'intervient jamais dans cette étape. L'échantillonnage est
déterministe avec la graine `42`.

Les vols annulés, déroutés ou sans retard d'arrivée connu sont exclus. Après
filtrage, 696 596 vols sont retenus.

| Ensemble | Mois | Vols | Retards ≥ 15 min | Taux de retard |
|---|---|---:|---:|---:|
| Entraînement | janvier à août | 462 156 | 107 561 | 23,27 % |
| Validation | septembre à octobre | 118 740 | 17 013 | 14,33 % |
| Test | novembre à décembre | 115 700 | 20 868 | 18,04 % |

Le découpage est chronologique. Le test représente donc une période future et
n'est utilisé ni pour ajuster les modèles, ni pour calibrer les probabilités, ni
pour choisir les seuils.

## Prévention des fuites de données

Les modèles utilisent uniquement des informations disponibles avant le départ :

- calendrier et encodages cycliques du mois et du jour ;
- horaires de départ et d'arrivée prévus ;
- durée prévue et distance ;
- compagnie, aéroports, États et route.

Les informations suivantes sont explicitement interdites comme features :
horaires réels, `dep_delay`, `arr_delay`, roulage, temps de vol, durée réelle et
colonnes officielles de causes. `arr_delay` et les causes servent uniquement à
construire les cibles d'apprentissage.

## Modèles

Les nombres manquants sont remplacés par la médiane puis standardisés. Les
catégories sont imputées et encodées en one-hot, avec regroupement des modalités
rares et gestion des catégories inconnues.

La classification utilise une régression logistique entraînée par descente de
gradient stochastique avec pondération des classes. Une calibration sigmoïde est
ajustée sur la validation. Le seuil de décision maximise le F1 de validation.

La régression est entraînée uniquement sur les vols retardés. Elle prédit le
logarithme du nombre de minutes, puis la sortie est reconvertie et bornée entre
15 et 1 440 minutes.

Les causes sont cinq problèmes binaires indépendants entraînés uniquement sur
les vols retardés. Elles sont donc conditionnelles à l'existence d'un retard et
plusieurs causes peuvent être sélectionnées pour le même vol.

## Classification du retard

| Métrique | Validation | Test |
|---|---:|---:|
| ROC-AUC | 0,638 | 0,602 |
| Average precision | 0,219 | 0,237 |
| Exactitude | 0,640 | 0,693 |
| Balanced accuracy | 0,601 | 0,560 |
| Précision | 0,210 | 0,250 |
| Rappel | 0,546 | 0,350 |
| F1 | 0,303 | 0,292 |
| Log loss | 0,396 | 0,478 |
| Brier score | 0,119 | 0,148 |

Le seuil calibré est `0.158380801`. Sur le test, la matrice de confusion est :

| | Prédit sans retard | Prédit en retard |
|---|---:|---:|
| Réel sans retard | 72 897 | 21 935 |
| Réel en retard | 13 556 | 7 312 |

La ROC-AUC supérieure à 0,5 montre que le planning contient un signal, mais le
classement reste modeste. Le faible F1 s'explique notamment par le déséquilibre
des classes et par l'absence de météo, d'état du trafic et de retard accumulé par
l'appareil avant le vol.

## Estimation des minutes

L'évaluation porte sur les 20 868 vols retardés du test.

| Modèle | MAE | RMSE | R² |
|---|---:|---:|---:|
| Régression conditionnelle | 44,04 min | 105,08 min | -0,066 |
| Baseline médiane de 43 min | 44,04 min | 104,89 min | -0,062 |

La régression ne surpasse pas réellement la baseline médiane. L'estimation en
minutes est disponible pour préparer l'intégration Streamlit, mais elle doit être
affichée comme expérimentale. Le résultat indique surtout que les seules données
de planning ne suffisent pas à prévoir l'amplitude d'un retard.

## Estimation des causes

Les métriques suivantes sont calculées uniquement parmi les vols retardés du
test. `Support` désigne le nombre de vols pour lesquels la cause est présente.

| Cause | Support | Précision | Rappel | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `carrier` | 11 528 | 0,573 | 0,941 | 0,712 | 0,613 |
| `weather` | 1 198 | 0,198 | 0,194 | 0,196 | 0,706 |
| `nas` | 10 312 | 0,519 | 0,930 | 0,666 | 0,637 |
| `security` | 86 | 0,008 | 0,174 | 0,015 | 0,677 |
| `late_aircraft` | 9 871 | 0,605 | 0,834 | 0,701 | 0,744 |

`late_aircraft` est la cause la mieux classée par ROC-AUC. Le résultat de
`security` n'est pas exploitable comme décision : seulement 86 cas positifs sont
présents dans le test et la précision est presque nulle.

## Coefficients principaux

Les associations absolues les plus fortes du classifieur de retard concernent
notamment l'origine Hawaï, la destination SFO, la compagnie YX, la distance, la
durée prévue et plusieurs routes liées à SFO ou ASE. Le détail complet est généré
localement dans `models/feature_importance.csv`.

Un coefficient ne prouve pas qu'une feature cause le retard. Pour les nombres
standardisés et les catégories one-hot, son signe indique seulement le sens de
l'association appris par cette baseline, toutes les autres features du modèle
étant conservées.

## Artefacts produits

```text
models/
├── flight_delay_models.joblib
├── training_metrics.json
└── feature_importance.csv
```

Ces fichiers sont ignorés par Git parce qu'ils sont générés et liés aux versions
des bibliothèques. Chaque membre de l'équipe les régénère avec la commande
d'entraînement. Un artefact Joblib ne doit être chargé que s'il provient d'une
source fiable.

## Prochaines améliorations

- construire des taux historiques glissants par compagnie, aéroport et route,
  sans utiliser d'informations futures ;
- ajouter les prévisions météo connues avant le départ ;
- intégrer l'état récent du trafic et la rotation précédente de l'appareil ;
- comparer des modèles d'arbres et régler leurs hyperparamètres ;
- entraîner la meilleure configuration sur davantage que 10 % du CSV ;
- réserver deux périodes distinctes pour la calibration et le choix des seuils ;
- ajouter une explication locale adaptée au formulaire Streamlit.
