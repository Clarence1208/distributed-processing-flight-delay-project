# Analyse PySpark des retards de vols

## Périmètre

L'analyse porte sur l'échantillon reproductible de 10 000 vols obtenu avec la
graine `42`. Un vol achevé est considéré en retard lorsque `arr_delay > 0`.

Les cinq colonnes de causes postérieures au vol sont exclues dès le parsing
propre. Elles ne participent plus aux statistiques ni aux corrélations.

## Résultats globaux

| Indicateur | Résultat |
|---|---:|
| Vols analysés | 10 000 |
| Vols achevés | 9 836 |
| Vols arrivés en retard | 3 578 |
| Taux de retard | 36,38 % |
| Vols annulés | 122 (1,22 %) |
| Vols déroutés | 42 (0,42 %) |
| Retard moyen à l'arrivée | 7,55 minutes |
| Retard médian à l'arrivée | −6 minutes |
| 90e percentile | 47 minutes |
| Retard maximal | 2 014 minutes |

La médiane négative montre qu'au moins la moitié des vols arrivent en avance.
Une minorité de retards extrêmes tire néanmoins la moyenne vers le haut.

## Qualité des données

Les 164 valeurs manquantes de `arr_delay`, `actual_elapsed_time` et `air_time`
correspondent aux 122 annulations et aux 42 déroutements. Les variables prévues,
la distance, la compagnie et les aéroports sont complètes dans l'échantillon.

Les 10 000 lignes passent les règles de parsing. Le Parquet propre ne contient
plus `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay` ni
`late_aircraft_delay`.

## Corrélations avec `arr_delay`

Les corrélations de Pearson utilisent 9 836 vols achevés.

| Variable | Corrélation | Disponible avant le départ ? |
|---|---:|:---:|
| `dep_delay` | 0,966 | Non |
| `taxi_out` | 0,184 | Non |
| `taxi_in` | 0,130 | Non |
| `scheduled_departure_minutes` | 0,087 | Oui |
| `scheduled_arrival_minutes` | 0,075 | Oui |
| `actual_elapsed_time` | 0,048 | Non |
| `day_of_week` | 0,037 | Oui |
| `month` | −0,022 | Oui |
| `air_time` | 0,011 | Non |
| `crs_elapsed_time` | −0,008 | Oui |
| `distance` | −0,008 | Oui |

`dep_delay` est presque directement lié au retard d'arrivée, mais il est connu
après le départ. Il constituerait une fuite de données. Les variables numériques
connues à l'avance ont des relations linéaires faibles ; CatBoost exploite aussi
les non-linéarités et les interactions catégorielles.

## Variations mensuelles

| Mois | Taux de retard |
|---|---:|
| Janvier | 39,75 % |
| Février | 29,76 % |
| Mars | 39,13 % |
| Avril | 36,09 % |
| Mai | 41,43 % |
| Juin | 40,27 % |
| Juillet | 48,19 % |
| Août | 37,92 % |
| Septembre | 28,88 % |
| Octobre | 25,19 % |
| Novembre | 32,62 % |
| Décembre | 37,36 % |

Juillet est le mois le plus retardé de l'échantillon ; octobre est le moins
retardé. Ces observations restent indicatives sur seulement 10 000 vols.

## Compagnies et aéroports principaux

Parmi les cinq compagnies les plus représentées :

| Compagnie | Vols | Taux de retard |
|---|---:|---:|
| `WN` | 2 022 | 38,43 % |
| `DL` | 1 459 | 32,71 % |
| `AA` | 1 385 | 42,87 % |
| `OO` | 1 080 | 32,80 % |
| `UA` | 1 033 | 34,19 % |

Parmi les cinq aéroports de départ les plus représentés :

| Aéroport | Vols | Taux de retard |
|---|---:|---:|
| `ATL` | 496 | 33,88 % |
| `DFW` | 438 | 42,29 % |
| `DEN` | 412 | 43,63 % |
| `ORD` | 391 | 39,02 % |
| `CLT` | 319 | 41,08 % |

Ces taux ne démontrent pas une causalité propre à la compagnie ou à l'aéroport.
Ils mélangent notamment routes, horaires, saison et conditions opérationnelles.

## Limites

- L'échantillon Spark sert à comprendre les données, pas à entraîner le modèle.
- Une corrélation ne démontre pas une causalité.
- Pearson ne mesure que les relations linéaires numériques.
- Les valeurs extrêmes influencent les moyennes et les corrélations.
- Le dataset ML complet reste traité exclusivement en Python.
