# Analyse PySpark des retards de vols

## Périmètre et méthode

Cette analyse porte uniquement sur l'échantillon reproductible de 2 000 vols
préparé pour Spark avec la graine `42`. Un vol est considéré en retard lorsque
son retard à l'arrivée (`arr_delay`) est supérieur ou égal à 15 minutes.

La cause `late_aircraft_delay` est volontairement exclue dès le parsing propre.
Elle ne participe ni aux statistiques, ni aux corrélations, ni aux futures
explications du modèle.

Les corrélations sont des corrélations linéaires de Pearson calculées sur 1 971
vols achevés, non annulés, non déroutés et sans valeur manquante dans les
variables étudiées.

## Résultats globaux

| Indicateur | Résultat |
|---|---:|
| Vols analysés | 2 000 |
| Vols achevés | 1 971 |
| Retards d'au moins 15 minutes | 420 |
| Taux de retard parmi les vols achevés | 21,3 % |
| Vols annulés | 25 (1,25 %) |
| Vols déroutés | 4 (0,20 %) |
| Retard moyen à l'arrivée | 9,9 minutes |
| Retard médian à l'arrivée | -6 minutes |
| 90e percentile du retard | 49 minutes |
| Retard maximal | 2 014 minutes |

La médiane négative indique qu'au moins la moitié des vols arrivent en avance.
La moyenne positive est tirée vers le haut par une minorité de retards très
importants. Le 99e percentile est de 221 minutes, tandis qu'un vol extrême atteint
2 014 minutes de retard. La médiane et les percentiles décrivent donc mieux un
vol habituel que la moyenne seule.

## Qualité et valeurs manquantes

Les 29 valeurs manquantes de `arr_delay`, `actual_elapsed_time` et `air_time`
correspondent aux 25 annulations et aux 4 déroutements. Elles sont attendues et
ne constituent pas une erreur de parsing.

`cancellation_code` est vide pour 1 975 vols, car cette information n'est
renseignée que pour les 25 vols annulés. Les variables prévues avant le départ,
la distance, la compagnie et les aéroports ne comportent aucune valeur manquante.

## Corrélations avec le retard à l'arrivée

| Variable | Corrélation de Pearson | Disponible avant le départ ? |
|---|---:|:---:|
| `dep_delay` | 0,983 | Non |
| `carrier_delay` | 0,764 | Non |
| `weather_delay` | 0,298 | Non |
| `nas_delay` | 0,282 | Non |
| `taxi_out` | 0,155 | Non |
| `scheduled_arrival_minutes` | 0,070 | Oui |
| `scheduled_departure_minutes` | 0,063 | Oui |
| `month` | -0,050 | Oui |
| `crs_elapsed_time` | 0,041 | Oui |
| `distance` | 0,036 | Oui |
| `day_of_week` | 0,008 | Oui |

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
| Compagnie (`carrier_delay`) | 11 964 | 212 | 58,4 % |
| Système aérien national (`nas_delay`) | 5 967 | 222 | 29,1 % |
| Météo (`weather_delay`) | 2 531 | 29 | 12,3 % |
| Sécurité (`security_delay`) | 41 | 2 | 0,2 % |

Dans cet échantillon, la compagnie représente 58,4 % des minutes attribuées aux
quatre causes retenues. Ces colonnes décrivent toutefois les causes constatées
après le vol ; elles peuvent servir de cibles explicatives, mais jamais de
features d'entrée avant le départ.

## Variations temporelles et opérationnelles

Les mois de juin (31,4 %), juillet (30,5 %) et mars (27,2 %) ont les taux de
retard les plus élevés. Octobre (10,4 %), septembre (12,8 %) et février (14,1 %)
ont les taux les plus faibles. Chaque mois contient entre 136 et 194 vols : ces
écarts sont indicatifs et devront être confirmés sur le dataset ML complet.

Parmi les compagnies les plus représentées, les taux observés sont de 26,3 %
pour `AA` (283 vols), 21,1 % pour `UA` (218), 20,9 % pour `WN` (373), 18,9 %
pour `DL` (291) et 17,0 % pour `OO` (225). `B6` atteint 34,7 %, mais sur seulement
72 vols.

Parmi les aéroports de départ les plus présents, `CLT` atteint 38,0 % de retards
sur 73 vols, `DEN` 31,6 % sur 77 vols et `DFW` 25,3 % sur 82 vols. Ces valeurs ne
mesurent pas un effet causal propre à l'aéroport : elles mélangent notamment les
routes, compagnies, horaires et conditions rencontrées.

## Limites et conséquences pour le ML

- L'échantillon de 2 000 lignes sert à comprendre les données, pas à tirer des
  conclusions définitives sur tous les vols américains.
- Une corrélation ne démontre pas une causalité.
- Pearson mesure seulement les relations linéaires entre variables numériques.
- Les valeurs extrêmes influencent fortement les moyennes et les corrélations.
- Le futur modèle devra exclure les informations connues après le départ et
  traiter les variables catégorielles avec un encodage adapté.

Le dataset ML complet restera traité exclusivement en Python lors de l'étape 3.
