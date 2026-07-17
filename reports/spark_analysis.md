# Analyse PySpark des retards de vols

## Périmètre et méthode

Cette analyse porte uniquement sur l'échantillon reproductible de 10 000 vols
préparé pour Spark avec la graine `42`. Un vol est considéré en retard lorsque
son retard à l'arrivée (`arr_delay`) est supérieur ou égal à 15 minutes.

La cause `late_aircraft_delay` est volontairement exclue dès le parsing propre.
Elle ne participe ni aux statistiques, ni aux corrélations, ni aux futures
explications du modèle.

Les corrélations sont des corrélations linéaires de Pearson calculées sur 9 836
vols achevés, non annulés, non déroutés et sans valeur manquante dans les
variables étudiées.

## Résultats globaux

| Indicateur | Résultat |
|---|---:|
| Vols analysés | 10 000 |
| Vols achevés | 9 836 |
| Retards d'au moins 15 minutes | 2 119 |
| Taux de retard parmi les vols achevés | 21,5 % |
| Vols annulés | 122 (1,22 %) |
| Vols déroutés | 42 (0,42 %) |
| Retard moyen à l'arrivée | 7,5 minutes |
| Retard médian à l'arrivée | -6 minutes |
| 90e percentile du retard | 47 minutes |
| Retard maximal | 2 014 minutes |

La médiane négative indique qu'au moins la moitié des vols arrivent en avance.
La moyenne positive est tirée vers le haut par une minorité de retards très
importants. Le 99e percentile est de 217 minutes, tandis qu'un vol extrême atteint
2 014 minutes de retard. La médiane et les percentiles décrivent donc mieux un
vol habituel que la moyenne seule.

## Qualité et valeurs manquantes

Les 164 valeurs manquantes de `arr_delay`, `actual_elapsed_time` et `air_time`
correspondent aux 122 annulations et aux 42 déroutements. Elles sont attendues et
ne constituent pas une erreur de parsing.

`cancellation_code` est vide pour 9 878 vols, car cette information n'est
renseignée que pour les 122 vols annulés. Les variables prévues avant le départ,
la distance, la compagnie et les aéroports ne comportent aucune valeur manquante.

## Corrélations avec le retard à l'arrivée

| Variable | Corrélation de Pearson | Disponible avant le départ ? |
|---|---:|:---:|
| `dep_delay` | 0,966 | Non |
| `carrier_delay` | 0,658 | Non |
| `nas_delay` | 0,355 | Non |
| `weather_delay` | 0,317 | Non |
| `taxi_out` | 0,184 | Non |
| `taxi_in` | 0,130 | Non |
| `scheduled_departure_minutes` | 0,087 | Oui |
| `scheduled_arrival_minutes` | 0,075 | Oui |
| `actual_elapsed_time` | 0,048 | Non |
| `day_of_week` | 0,037 | Oui |
| `month` | -0,022 | Oui |
| `air_time` | 0,011 | Non |
| `crs_elapsed_time` | -0,008 | Oui |
| `distance` | -0,008 | Oui |
| `security_delay` | 0,006 | Non |
| `day_of_month` | 0,001 | Oui |

`dep_delay` explique presque directement `arr_delay`, mais n'est connu qu'après
le départ. Les quatre colonnes de causes retenues sont attribuées après le vol.
Ces variables seraient donc des fuites de données dans un modèle qui prédit
avant le départ.

Les variables numériques réellement connues à l'avance ont ici des corrélations
linéaires faibles. Cela ne signifie pas qu'elles sont inutiles : les relations
peuvent être non linéaires et dépendre d'interactions avec la compagnie,
l'aéroport, la route et l'horaire. Les variables catégorielles ne sont par
ailleurs pas représentées dans une corrélation de Pearson classique.

## Causes enregistrées des retards

| Cause | Minutes | Vols affectés | Part des minutes attribuées |
|---|---:|---:|---:|
| Compagnie (`carrier_delay`) | 48 709 | 1 136 | 54,2 % |
| Système aérien national (`nas_delay`) | 30 271 | 1 056 | 33,7 % |
| Météo (`weather_delay`) | 10 831 | 141 | 12,0 % |
| Sécurité (`security_delay`) | 88 | 6 | 0,1 % |

Dans cet échantillon, la compagnie représente 54,2 % des minutes attribuées aux
quatre causes retenues. Ces colonnes décrivent toutefois les causes constatées
après le vol ; elles peuvent servir de cibles explicatives, mais jamais de
features d'entrée avant le départ.

## Variations temporelles et opérationnelles

Les mois de juillet (31,9 %), juin (26,4 %) et janvier (25,2 %) ont les taux de
retard les plus élevés. Octobre (11,9 %), février (14,8 %) et septembre (15,4 %)
ont les taux les plus faibles. Chaque mois contient entre 716 et 946 vols : ces
écarts sont indicatifs et devront être confirmés sur le dataset ML complet.

Parmi les cinq compagnies les plus représentées, les taux observés sont de
28,1 % pour `AA` (1 385 vols), 22,3 % pour `UA` (1 033), 21,3 % pour `WN`
(2 022), 19,7 % pour `OO` (1 080) et 17,5 % pour `DL` (1 459).

Parmi les cinq aéroports de départ les plus présents, `DEN` atteint 27,9 % de
retards sur 412 vols, `CLT` 27,7 % sur 319, `ORD` 26,4 % sur 391, `DFW` 25,9 %
sur 438 et `ATL` 19,7 % sur 496. Ces valeurs ne mesurent pas un effet causal
propre à l'aéroport : elles mélangent notamment les routes, compagnies, horaires
et conditions rencontrées.

## Limites et conséquences pour le ML

- L'échantillon de 10 000 lignes sert à comprendre les données, pas à tirer des
  conclusions définitives sur tous les vols américains.
- Une corrélation ne démontre pas une causalité.
- Pearson mesure seulement les relations linéaires entre variables numériques.
- Les valeurs extrêmes influencent fortement les moyennes et les corrélations.
- Le futur modèle devra exclure les informations connues après le départ et
  traiter les variables catégorielles avec un encodage adapté.

Le dataset ML complet restera traité exclusivement en Python lors de l'étape 3.
