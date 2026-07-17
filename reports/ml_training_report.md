# Rapport du modèle machine learning — protocole métier v5

## Résumé

Le modèle réduit désormais fortement la sur-alerte, mais il n'atteint pas le
niveau de fiabilité demandé. Sur les 115 700 vols de novembre et décembre, il
signale 11 433 vols, soit 9,9 %. Parmi ces alertes, 34,2 % correspondent à un
retard réel d'au moins 15 minutes et 18,7 % des retards sont détectés.

Le contrat exige une précision d'au moins 50 %, un rappel d'au moins 20 % et une
couverture comprise entre 5 et 10 %. Les bornes basses à 95 % doivent elles aussi
dépasser les objectifs et au moins 500 alertes sont requises. Le résultat est
donc sans ambiguïté : **business gate échoué, modèle non publiable**.

Cette conclusion est volontairement plus stricte qu'une simple accuracy. Sur le
test, répondre « à l'heure » pour tous les vols aurait 82 % d'exactitude, tout en
ne détectant aucun retard.

## Questions traitées

Le pipeline tente de répondre, avant le départ prévu, à trois questions :

1. le vol aura-t-il au moins 15 minutes de retard à l'arrivée ?
2. s'il est retardé, combien de minutes peut-on estimer ?
3. quelles causes sont les plus plausibles parmi `carrier`, `weather`, `nas` et
   `security` ?

`late_aircraft_delay` est exclue dès le parsing. Elle n'est ni une feature, ni
une cible, ni une explication retournée.

## Données et protocole temporel

Le CSV complet contient 7 079 081 vols. Polars le parcourt pour construire les
historiques et le planning, puis l'échantillonnage déterministe à 10 % conserve
696 596 vols achevés comme cibles ML.

| Sous-ensemble | Rôle | Mois | Vols | Retards | Taux |
|---|---|---|---:|---:|---:|
| Entraînement | apprentissage initial | janvier à juillet | 401 778 | 93 438 | 23,26 % |
| Réglage | early stopping | août | 60 378 | 14 123 | 23,39 % |
| Validation | choix du seuil uniquement | septembre à octobre | 118 740 | 17 013 | 14,33 % |
| Test | mesure hors échantillon | novembre à décembre | 115 700 | 20 868 | 18,04 % |

Août choisit le nombre d'arbres. Le modèle final est ensuite réentraîné sur
janvier à août avec ce nombre fixé. Septembre-octobre ne sert qu'au seuil ; le
test n'est jamais consulté par l'algorithme de sélection.

Le benchmark novembre-décembre a toutefois été observé lors des itérations
humaines précédentes du projet. Il reste utile pour la comparaison, mais un
holdout 2025 jamais consulté sera indispensable avant tout déploiement.

## Features disponibles avant le départ

### Planning statique

- date, jour, semaine, week-end et encodages cycliques ;
- heures prévues, durée prévue, distance et numéro de vol ;
- compagnie, aéroports, États, route et interactions aéroport-heure.

### Congestion planifiée

Six volumes sont calculés sur le planning complet avant échantillonnage :

- `origin_scheduled_departures_hour` ;
- `origin_scheduled_departures_3h` ;
- `dest_scheduled_arrivals_hour` ;
- `dest_scheduled_arrivals_3h` ;
- `route_scheduled_flights_day` ;
- `carrier_origin_scheduled_flights_day`.

Ils comptent les vols prévus, y compris ceux qui seront ensuite annulés ou
déroutés. Le jour d'arrivée est dérivé des heures et de `crs_elapsed_time` pour
éviter de confondre un décalage de fuseau avec un passage à J+1.

Pour un vol isolé, l'artefact peut fournir un profil médian par mois, jour et
créneau. Ce profil sert uniquement au diagnostic : une alerte publiable exige
les six volumes exacts du planning journalier.

### Historiques sans fuite

Les 68 features historiques utilisent uniquement les jours strictement
antérieurs au vol cible.

| Niveau | Fenêtres |
|---|---|
| Réseau | 1, 3, 7 et 28 jours |
| Compagnie | 1, 7 et 28 jours |
| Origine | 1, 3, 14 et 28 jours |
| Destination | 1, 3, 14 et 28 jours |
| Route | 7 et 28 jours |

Chaque fenêtre fournit taux de retard, taux de perturbation, volume et gravité
moyenne. Les horaires réels, `dep_delay`, les durées réelles et les causes
officielles ne sont jamais des features pré-départ.

## Contrat métier et seuil

Le modèle est publiable seulement si toutes les conditions suivantes passent
sur la validation, le test global et chaque mois du test :

- borne basse de Wilson à 95 % de la précision ≥ 50 % ;
- borne basse de Wilson à 95 % du rappel ≥ 20 % ;
- couverture des alertes entre 5 et 10 % ;
- au moins 500 alertes.

Aucun seuil ne respecte ce contrat sur septembre-octobre. Le seuil de repli
`0,337121` cible donc le centre de la plage, soit 7,5 % d'alertes sur la
validation. Un seuil de repli ne peut jamais rendre le gate positif.

## Classification principale

| Métrique | Validation | Test |
|---|---:|---:|
| Précision | 31,08 % | 34,18 % |
| Intervalle de précision à 95 % | [30,13 % ; 32,05 %] | [33,32 % ; 35,06 %] |
| Rappel | 16,27 % | 18,73 % |
| Intervalle de rappel à 95 % | [15,72 % ; 16,83 %] | [18,20 % ; 19,26 %] |
| Couverture | 7,50 % | 9,88 % |
| Alertes | 8 906 | 11 433 |
| Vrais positifs | 2 768 | 3 908 |
| Faux positifs | 6 138 | 7 525 |
| F1 | 0,214 | 0,242 |
| ROC-AUC | 0,666 | 0,648 |
| Average precision | 0,244 | 0,280 |
| Gate | Échoué | Échoué |

La matrice de confusion du test est :

| | Prédit à l'heure | Alerte expérimentale |
|---|---:|---:|
| Réel à l'heure | 87 307 | 7 525 |
| Réel en retard | 16 960 | 3 908 |

### Stabilité mensuelle

| Mois | Précision | Rappel | Couverture | Alertes | Gate |
|---|---:|---:|---:|---:|---|
| Novembre | 30,14 % | 18,51 % | 9,04 % | 5 170 | Échoué |
| Décembre | 37,52 % | 18,87 % | 10,70 % | 6 263 | Échoué |

Décembre dépasse la limite de couverture. Cette dérive confirme qu'un seuil
global calibré sur septembre-octobre n'est pas stable toute l'année.

### Pourquoi ne plus maximiser F1

Sur le même modèle, le seuil F1 `0,214254` signalerait 46 388 vols du test,
soit 40,1 %. Sa précision serait seulement de 25,8 %, pour 57,4 % de rappel.
Ce comportement est mathématiquement défendable pour F1 mais inacceptable pour
un site utilisateur : près de trois alertes sur quatre seraient fausses et près
de la moitié des vols seraient signalés.

## Apport des variables de congestion

Les variables de congestion sont utilisées, mais leur signal reste secondaire :

| Feature | Rang d'importance | Importance CatBoost |
|---|---:|---:|
| `dest_scheduled_arrivals_3h` | 42 | 0,958 |
| `origin_scheduled_departures_3h` | 50 | 0,666 |
| `dest_scheduled_arrivals_hour` | 52 | 0,601 |
| `carrier_origin_scheduled_flights_day` | 56 | 0,550 |
| `route_scheduled_flights_day` | 77 | 0,272 |
| `origin_scheduled_departures_hour` | 78 | 0,240 |

Les principales features restent les heures, la compagnie, le mois et les
perturbations historiques récentes. Le planning améliore le contexte disponible
avant le départ, mais ne remplace ni la météo prévisionnelle ni l'état réel des
opérations du jour.

## Régression conditionnelle

L'évaluation porte sur les 20 868 vols réellement retardés du test.

| Modèle | MAE | RMSE | R² |
|---|---:|---:|---:|
| CatBoost | 43,46 min | 105,25 min | -0,070 |
| Médiane de 43 min | 44,04 min | 104,89 min | -0,062 |

Le gain de MAE est inférieur à une minute et les deux autres métriques sont
moins bonnes. L'estimation individuelle reste expérimentale.

## Causes conditionnelles

| Cause | Cas positifs | Précision | Rappel | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `carrier` | 11 528 | 57,2 % | 95,0 % | 0,714 | 0,632 |
| `weather` | 1 198 | 18,4 % | 34,6 % | 0,240 | 0,714 |
| `nas` | 10 312 | 52,1 % | 94,9 % | 0,672 | 0,656 |
| `security` | 86 | 0,8 % | 1,2 % | 0,009 | 0,628 |

Les causes sont conditionnelles à un retard. Elles ne sont jamais affichées
comme décisions publiables lorsque le classifieur principal échoue au gate.
`security` doit rester masquée ou accompagnée d'un avertissement explicite.

## Contrat de prédiction

L'artefact Joblib v5 contient les modèles, les profils historiques, les profils
de planning et le résultat du business gate. `predict_flight` distingue :

- `diagnostic_*` : résultat expérimental consultable dans le notebook ;
- `published_delay_alert` : alerte exposable, sinon `None` ;
- `prediction_publishable` : vrai uniquement si le gate passe et si le planning
  journalier exact est fourni ;
- `publication_blockers` : raisons empêchant la publication.

Avec le résultat actuel, toute prédiction reste diagnostique.

## Limites et prochaines données nécessaires

Le modèle atteint la limite informative du CSV 2024 pour l'objectif demandé.
Les prochaines améliorations utiles sont :

1. prévisions météo réellement disponibles à l'instant de prédiction, par
   aéroport et horizon ;
2. état des opérations du jour avant le départ : retards observés, files et
   capacité aéroportuaire à un cutoff défini ;
3. avis FAA/NAS connus à ce même cutoff ;
4. connexion automatique au planning quotidien exact ;
5. validation finale sur 2025, jamais consultée pendant le développement ;
6. modèle séparé pour les annulations, actuellement exclues.

Augmenter uniquement la complexité de CatBoost ou abaisser le seuil ne peut pas
transformer les faux positifs actuels en information fiable.

## Reproduire le run

```bash
uv sync --extra dev --extra notebook --python 3.11
uv run train-flight-models \
  --input data/flight_data_2024.csv \
  --sample-fraction 0.1 \
  --seed 42
```

Les artefacts sont ignorés par Git et doivent être régénérés localement. Leur
construction parcourt le CSV complet et peut utiliser environ 4 Go de mémoire.
Un fichier Joblib ne doit être chargé que s'il provient d'une source fiable.
