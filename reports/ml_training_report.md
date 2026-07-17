# Rapport du classifieur de retard — protocole métier v6

## Résumé

Le machine learning répond désormais à une seule question : **le vol arrivera-t-il
en retard ?** La cible `is_delayed` vaut 1 dès que `arr_delay > 0`. Un retard
d'une minute est donc positif ; une arrivée exactement à l'heure ou en avance est
négative.

Les modèles de causes et la régression du nombre de minutes ont été supprimés.
Les colonnes `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay` et
`late_aircraft_delay` ne sont plus lues par le pipeline ML. Le code compagnie
`op_unique_carrier` reste une feature disponible avant le départ.

Sur les 115 700 vols du test novembre-décembre, le modèle signale 19 629 vols,
soit 17,0 %. Sa précision atteint **50,9 %**, son rappel **26,4 %** et sa borne
basse de précision à 95 % **50,2 %**. Le test global respecte donc les objectifs.

Le modèle reste néanmoins expérimental : la validation atteint seulement 49,1 %
de précision et les résultats diffèrent fortement entre novembre et décembre.
Le business gate complet reste donc échoué.

## Données et protocole temporel

Le CSV complet contient 7 079 081 vols. Polars le parcourt pour construire les
historiques et les volumes de planning, puis l'échantillonnage déterministe à
10 % conserve 696 596 vols achevés.

| Sous-ensemble | Rôle | Mois | Vols | Retards `> 0` | Taux |
|---|---|---|---:|---:|---:|
| Entraînement | apprentissage initial | janvier à juillet | 401 778 | 157 369 | 39,17 % |
| Réglage | early stopping | août | 60 378 | 23 189 | 38,41 % |
| Validation | choix du seuil | septembre à octobre | 118 740 | 34 063 | 28,69 % |
| Test | mesure hors échantillon | novembre à décembre | 115 700 | 37 896 | 32,75 % |

Août fixe le nombre d'arbres. Le classifieur final est ensuite réentraîné sur
janvier-août. Septembre-octobre choisit le seuil ; novembre-décembre n'est pas
consulté par l'algorithme de sélection.

Le benchmark 2024 a été regardé pendant le développement humain. Une validation
finale sur 2025, jamais consultée, reste indispensable avant un déploiement réel.

## Features disponibles avant le départ

Le modèle utilise uniquement des informations disponibles avant le départ :

- date, mois, jour, semaine, week-end et encodages cycliques ;
- horaires prévus, durée prévue, distance et numéro de vol ;
- compagnie, aéroports, États, route et interactions aéroport-heure ;
- six volumes de congestion construits sur le planning complet ;
- historiques glissants du réseau, de la compagnie, des aéroports et de la
  route, arrêtés avant le jour du vol.

Les historiques fournissent taux de retard, taux de perturbation et volume sur
des fenêtres de 1 à 28 jours. Les anciennes moyennes de minutes, inutilisées par
le classifieur, ont été retirées.

Les six features de planning sont :

- `origin_scheduled_departures_hour` ;
- `origin_scheduled_departures_3h` ;
- `dest_scheduled_arrivals_hour` ;
- `dest_scheduled_arrivals_3h` ;
- `route_scheduled_flights_day` ;
- `carrier_origin_scheduled_flights_day`.

`dep_delay`, `arr_delay`, les horaires réels, les durées réelles et toutes les
causes officielles sont interdites comme features.

## Contrat métier

Le plafond d'alertes a été adapté à la nouvelle prévalence de la cible. Avec
32,8 % de retards dans le test, limiter les alertes à 10 % tout en demandant
20 % de rappel imposerait mécaniquement une précision supérieure à 65 %.

Le contrat v6 exige donc :

- borne basse de Wilson à 95 % de la précision ≥ 50 % ;
- borne basse de Wilson à 95 % du rappel ≥ 20 % ;
- couverture des alertes entre 5 et 20 % ;
- au moins 500 alertes ;
- conditions respectées sur la validation, le test global et chaque mois du
  test.

Aucun seuil ne respecte toutes les contraintes sur septembre-octobre. Le seuil
de repli `0,448117` vise le centre de la plage de couverture, soit 12,5 % sur la
validation. Un seuil de repli ne peut pas rendre le gate positif.

## Résultats de classification

| Métrique | Validation | Test |
|---|---:|---:|
| Précision | 49,08 % | **50,89 %** |
| Intervalle de précision à 95 % | [48,28 % ; 49,88 %] | [50,19 % ; 51,59 %] |
| Rappel | 21,39 % | **26,36 %** |
| Intervalle de rappel à 95 % | [20,95 % ; 21,83 %] | [25,92 % ; 26,81 %] |
| Couverture | 12,50 % | **16,97 %** |
| Alertes | 14 843 | 19 629 |
| Vrais positifs | 7 285 | 9 990 |
| Faux positifs | 7 558 | 9 639 |
| F1 | 0,298 | 0,347 |
| ROC-AUC | 0,657 | 0,643 |
| Average precision | 0,423 | 0,459 |
| Gate local | Échoué | Réussi |

Matrice de confusion du test :

| | Prédit à l'heure | Signalé en retard |
|---|---:|---:|
| Réel à l'heure | 68 165 | 9 639 |
| Réel en retard | 27 906 | 9 990 |

## Stabilité mensuelle

| Mois | Précision | Rappel | Couverture | Alertes | Gate |
|---|---:|---:|---:|---:|---|
| Novembre | 49,14 % | 16,25 % | 9,55 % | 5 460 | Échoué |
| Décembre | 51,57 % | 34,17 % | 24,22 % | 14 169 | Échoué |

Novembre manque les objectifs de précision et de rappel. Décembre dépasse le
plafond de couverture de 20 %. Le même seuil n'est donc pas stable entre les
deux mois.

## Pourquoi le seuil F1 est rejeté

Le seuil maximisant F1 (`0,270839`) donnerait sur le test :

- précision : 38,66 % ;
- rappel : 78,71 % ;
- couverture : 66,68 %, soit 77 153 vols signalés ;
- F1 : 0,519.

Il obtiendrait un meilleur F1 en signalant les deux tiers des vols. Ce
comportement est incompatible avec l'objectif utilisateur, même si la métrique
statistique augmente.

## Features les plus importantes

| Feature | Importance |
|---|---:|
| `arrival_hour` | 5,718 |
| `departure_time_sin` | 5,465 |
| `dest_hour` | 5,303 |
| `dest_delay_rate_1d` | 5,283 |
| `departure_hour` | 4,965 |
| `scheduled_departure_minutes` | 4,935 |
| `route_delay_rate_7d` | 4,884 |
| `route_disruption_rate_7d` | 4,852 |
| `day_of_week_category` | 4,772 |
| `op_unique_carrier` | 4,569 |

Le code compagnie contribue bien au modèle, mais il n'est pas suffisant seul.
Les heures, la destination, la route et les historiques récents apportent
également du signal.

## Contrat de prédiction

L'artefact Joblib v6 contient un seul `CatBoostClassifier`, les profils
historiques, les profils de planning et le résultat du business gate.

`predict_flight` retourne notamment :

- `delay_probability` : probabilité diagnostique ;
- `diagnostic_is_delayed_prediction` : classe interne expérimentale ;
- `published_delay_alert` : décision publiable, sinon `None` ;
- `prediction_publishable` : vrai uniquement si le gate passe et si les six
  volumes exacts du planning journalier sont fournis ;
- `publication_blockers` : raisons empêchant la publication.

Avec le run actuel, toute prédiction reste diagnostique.

## Limites et prochaines améliorations

Le test global atteint le seuil demandé, mais pas la validation ni chaque mois.
Les améliorations prioritaires sont :

1. ajouter des informations réellement disponibles le jour du vol : état des
   opérations, capacité aéroportuaire et avis NAS connus à un cutoff défini ;
2. connecter automatiquement le planning journalier exact ;
3. calibrer et contrôler la dérive saisonnière sans consulter le test final ;
4. valider sur une année 2025 entièrement tenue à l'écart ;
5. éventuellement traiter les annulations dans un modèle séparé.

La météo et la sécurité ne doivent pas être réintroduites sans données
prévisionnelles ou opérationnelles correspondantes.

## Reproduire le run

```bash
uv sync --extra dev --extra notebook --python 3.11
uv run train-flight-models \
  --input data/flight_data_2024.csv \
  --sample-fraction 0.1 \
  --seed 42
```

L'entraînement produit l'artefact v6 et ses métriques dans `models/official/`.
L'artefact officiel de 4 Mo est versionné pour rendre le formulaire Streamlit
utilisable après clonage ; les métriques et importances restent reproductibles
localement.
