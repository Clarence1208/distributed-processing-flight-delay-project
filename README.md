# Prédiction des retards de vols

Projet étudiant de traitement distribué consacré aux vols domestiques américains
de 2024. L'objectif final est de prédire si un vol aura au moins 15 minutes de
retard, d'estimer le nombre de minutes et de présenter les facteurs possibles
dans une interface Streamlit.

Le code et les noms techniques sont en anglais. Les commentaires, les messages,
les rapports et la documentation sont en français.

## État du projet

| Étape | État | Contenu |
|---|---|---|
| 1. Parsing PySpark | Terminé | Échantillon reproductible de 2 000 vols, typage, contrôles qualité et Parquet |
| 2. Analyse PySpark | Terminé | Statistiques, valeurs manquantes, agrégations, causes et corrélations de Pearson |
| 3. Machine learning Python | Baseline terminée | Chargement Python, classification calibrée, régression, causes, évaluation et prédiction |
| 4. Streamlit | À faire | Tableau de bord, graphiques et formulaire de prédiction |

Les résultats sont interprétés dans le
[rapport Spark](reports/spark_analysis.md) et le
[rapport du machine learning](reports/ml_training_report.md).

## Fonctionnalités actuellement implémentées

### Préparation Spark

- lecture des 35 colonnes CSV avec un schéma explicite, sans inférence ;
- tirage aléatoire exact et reproductible de 2 000 lignes avec la graine `42` ;
- conversion contrôlée des dates, entiers, nombres réels et textes ;
- acceptation des entiers CSV écrits sous la forme `12` ou `12.0` ;
- validation des champs obligatoires, dates, heures `HHMM`, indicateurs, durées
  et distances ;
- séparation entre lignes valides et quarantaine avec motifs détaillés ;
- écriture Parquet partitionnée par `year` et `month` ;
- rapport qualité global au format JSON ;
- commande reproductible `prepare-spark-data`.

### Analyse Spark

- taux de retard, d'annulation et de déroutement ;
- moyenne, médiane, percentiles et maximum du retard ;
- valeurs manquantes par colonne ;
- indicateurs par mois, compagnie et aéroport de départ ;
- répartition des minutes entre les cinq causes enregistrées ;
- statistiques descriptives des variables numériques ;
- matrice de corrélation de Pearson sur les vols achevés ;
- classement des corrélations avec `arr_delay` ;
- indicateur `available_before_departure` pour repérer les fuites de données ;
- exports CSV/JSON et commande `analyze-spark-data`.

### Machine learning Python

- parcours du CSV complet avec Polars en mode streaming, sans Spark ;
- échantillonnage déterministe configurable, fixé à 10 % pour la baseline ;
- exclusion des vols annulés, déroutés ou sans retard d'arrivée connu ;
- découpage chronologique : janvier-août pour l'entraînement,
  septembre-octobre pour la validation et novembre-décembre pour le test ;
- classification d'un retard d'au moins 15 minutes avec `SGDClassifier` ;
- calibration sigmoïde des probabilités sur la validation ;
- régression conditionnelle du nombre de minutes avec `SGDRegressor` ;
- cinq classifications conditionnelles pour les causes `carrier`, `weather`,
  `nas`, `security` et `late_aircraft` ;
- seuils de décision choisis sur la validation en maximisant le score F1 ;
- métriques de classification, régression, confusion et coefficients ;
- sauvegarde locale du préprocesseur, des modèles et des métadonnées avec Joblib ;
- commandes reproductibles `train-flight-models` et `predict-flight`.

## Colonnes et features ajoutées

### Spark

| Colonne | Rôle | Persistée dans le Parquet propre |
|---|---|:---:|
| `scheduled_departure_minutes` | Heure de départ prévue convertie en minutes depuis minuit | Oui |
| `scheduled_arrival_minutes` | Heure d'arrivée prévue convertie en minutes depuis minuit | Oui |
| `validation_errors` | Liste des anomalies d'une ligne rejetée | Quarantaine uniquement |
| `is_valid_row` | Résultat des contrôles qualité | Quarantaine uniquement |
| `flight_status` | Statut exclusif utilisé pour l'analyse | Non, calculé pendant l'analyse |

### Machine learning

Les features numériques utilisées avant le départ sont `day_of_month`,
`is_weekend`, les encodages cycliques du mois, du jour de la semaine et des deux
horaires prévus, `crs_elapsed_time` et `distance`.

Les features catégorielles sont `op_unique_carrier`, `origin`,
`origin_state_nm`, `dest`, `dest_state_nm` et `route`. Les catégories manquantes
sont imputées puis encodées en one-hot. Les catégories rares sont regroupées et
les catégories inconnues restent acceptées lors d'une future prédiction.

Les cibles ajoutées sont `is_delayed_15`, `delay_minutes`, `reason_carrier`,
`reason_weather`, `reason_nas`, `reason_security` et `reason_late_aircraft`.

`dep_delay`, les horaires réels, les durées réelles et les cinq colonnes de
causes sont connus pendant ou après le vol. Ils ne sont jamais utilisés comme
features du modèle pré-départ. Les causes officielles servent uniquement de
cibles d'entraînement.

## Structure du dépôt

```text
.
├── data/
│   ├── flight_data_2024_sample.csv           # échantillon de 10 000 lignes versionné
│   ├── flight_data_2024_data_dictionary.csv  # description des colonnes versionnée
│   ├── flight_data_2024.csv                  # dataset ML complet, ignoré par Git
│   ├── processed/                            # sorties du parsing, ignorées
│   └── analysis/                             # sorties de l'analyse, ignorées
├── examples/
│   └── flight.json                           # futur vol d'exemple
├── reports/
│   ├── spark_analysis.md                     # rapport Spark versionné
│   └── ml_training_report.md                 # rapport ML versionné
├── src/flight_delays/
│   ├── parsing.py                            # étape 1, PySpark
│   ├── analysis.py                           # étape 2, PySpark
│   ├── ml_data.py                            # chargement et features Python
│   ├── training.py                           # entraînement et évaluation
│   └── prediction.py                         # prédiction d'un futur vol
├── tests/
├── pyproject.toml
└── uv.lock
```

Les fichiers générés et le CSV complet de 1,2 Go ne sont pas envoyés sur GitHub.
Ils sont soit régénérés par les commandes du projet, soit récupérés séparément.

## Installation sur un nouveau PC

### 1. Cloner le dépôt

```bash
git clone https://github.com/Clarence1208/distributed-processing-flight-delay-project.git
cd distributed-processing-flight-delay-project
```

### 2. Installer les prérequis

- Python 3.11 ;
- Java 17 ;
- Git ;
- `uv` pour créer l'environnement Python reproductible.

Vérifier Java et Python :

```bash
java -version
python3 --version
```

Si `uv` n'est pas déjà disponible :

```bash
python3 -m pip install uv
```

### 3. Définir `JAVA_HOME`

Sur macOS :

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

Sur Linux, adapter le chemin au JDK installé :

```bash
export JAVA_HOME=/chemin/vers/jdk-17
```

Sur Windows PowerShell :

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-17"
```

### 4. Créer l'environnement

```bash
uv sync --extra dev --python 3.11
```

`uv.lock` verrouille notamment PySpark, NumPy, Polars, pandas, scikit-learn,
Joblib et Pytest afin que toute l'équipe utilise le même environnement.

## Données disponibles

Le dépôt contient :

- `data/flight_data_2024_sample.csv` : 10 000 lignes pour tester Spark et le ML ;
- `data/flight_data_2024_data_dictionary.csv` : types et exemples.

Le fichier complet `data/flight_data_2024.csv` contient 7 079 081 vols et pèse
environ 1,2 Go. GitHub ne peut pas accueillir ce fichier ; chaque membre de
l'équipe doit le récupérer depuis la source commune du projet et le placer sous
ce nom exact. Il sert à reproduire les métriques ML mais ne passe jamais par
Spark.

## Exécution rapide

```bash
# 1. Vérifier les tests Spark et Python
uv run pytest

# 2. Préparer les 2 000 vols Spark
uv run prepare-spark-data --mode overwrite

# 3. Exécuter l'analyse Spark
uv run analyze-spark-data --mode overwrite

# 4a. Tester l'entraînement ML avec le petit CSV versionné
uv run train-flight-models \
  --input data/flight_data_2024_sample.csv \
  --sample-fraction 1

# 4b. Entraîner la baseline sur 10 % du CSV complet
uv run train-flight-models --sample-fraction 0.1

# 5. Prédire le vol d'exemple avec le modèle local
uv run predict-flight --flight-json examples/flight.json
```

Si nécessaire, définir `JAVA_HOME` dans le même terminal avant les commandes
Spark et les tests complets. Le ML Python n'a pas besoin de Java.

Les sorties reproductibles sont créées dans :

```text
data/processed/spark/flights_2000/
├── flights/
├── rejects/
└── quality_report/

data/analysis/spark/
├── overview/
├── flight_status/
├── missing_values/
├── monthly_metrics/
├── carrier_metrics/
├── origin_metrics/
├── delay_causes/
├── numeric_summary/
└── correlations/

models/
├── flight_delay_models.joblib
├── training_metrics.json
└── feature_importance.csv
```

Ces dossiers sont ignorés par Git et peuvent être supprimés puis régénérés.

Pour consulter toutes les options :

```bash
uv run prepare-spark-data --help
uv run analyze-spark-data --help
uv run train-flight-models --help
uv run predict-flight --help
```

## Résultats Spark actuels

Sur les 2 000 vols :

- 420 vols achevés ont au moins 15 minutes de retard, soit 21,3 % ;
- 25 vols sont annulés et 4 sont déroutés ;
- la médiane du retard à l'arrivée est de -6 minutes ;
- `late_aircraft_delay` et `carrier_delay` représentent ensemble environ 75,4 %
  des minutes de causes enregistrées ;
- `dep_delay` est très corrélé à `arr_delay` (`r = 0,983`), mais constitue une
  fuite de données pour un modèle pré-départ.

Ces résultats décrivent uniquement l'échantillon Spark de 2 000 lignes.

## Résultats ML actuels

La baseline a parcouru le CSV complet et retenu de façon déterministe 696 596
vols achevés, soit environ 10 %. Le test final contient 115 700 vols de novembre
et décembre, jamais utilisés pour ajuster les modèles ni les seuils.

Pour la classification d'un retard d'au moins 15 minutes :

- ROC-AUC : 0,602 ;
- précision : 0,250 ;
- rappel : 0,350 ;
- F1 : 0,292 ;
- exactitude : 0,693 ;
- seuil calibré : 0,158.

Ces scores indiquent un signal prédictif encore modeste avec les seules
informations de planning. Parmi les causes conditionnelles, `late_aircraft`
obtient la meilleure ROC-AUC avec 0,744. `security` est beaucoup trop rare pour
produire une décision fiable.

La régression conditionnelle obtient une MAE de 44,04 minutes, presque identique
à la baseline médiane de 44,04 minutes, et un R² négatif. L'interface devra donc
présenter cette estimation comme expérimentale tant que des données météo et des
agrégats historiques ne sont pas ajoutés.

## Travail restant

### Améliorations de l'étape 3

- créer des agrégats historiques sans utiliser le futur ;
- ajouter des données météo connues avant le départ et des indicateurs de trafic ;
- comparer la baseline linéaire à des modèles d'arbres et régler les paramètres ;
- entraîner la meilleure configuration sur davantage que 10 % des lignes ;
- séparer validation de calibration et validation de choix des seuils ;
- améliorer l'explication locale des prédictions, par exemple avec SHAP ;
- étudier un modèle spécifique aux annulations, actuellement exclues.

### Étape 4 — Streamlit

- tableau de bord des statistiques et corrélations ;
- filtres par date, compagnie, aéroport et route ;
- graphiques des retards et de leurs causes ;
- formulaire de saisie d'un futur vol ;
- probabilité de retard et estimation en minutes ;
- facteurs contribuant à la prédiction ;
- chargement des modèles sauvegardés ;
- tests de l'interface et instructions de lancement.

## Limites connues

- L'analyse Spark repose volontairement sur seulement 2 000 lignes.
- Une corrélation ne démontre pas une causalité et Pearson ne mesure que les
  relations linéaires numériques.
- Les causes officielles sont connues après le vol et servent uniquement de
  cibles d'entraînement.
- Les probabilités de causes sont conditionnelles au fait que le vol soit en
  retard et plusieurs causes peuvent être proposées en même temps.
- La régression actuelle n'améliore pas réellement la médiane d'entraînement.
- La calibration et les seuils utilisent le même ensemble de validation.
- Les fichiers Joblib ne doivent être chargés que s'ils proviennent d'une source
  fiable et doivent être régénérés après une mise à jour des dépendances.
- L'interface Streamlit n'est pas encore implémentée.

## Résolution de problèmes

- `JAVA_GATEWAY_EXITED` ou Java introuvable : vérifier Java 17 et `JAVA_HOME`.
- Sortie Spark déjà existante : ajouter `--mode overwrite`.
- Mémoire Spark insuffisante : conserver le master par défaut `local[2]`.
- CSV complet absent : utiliser le petit CSV versionné pour tester le ML.
- Mémoire limitée pendant le ML : réduire `--sample-fraction`, par exemple à
  `0.02`.
- Modèle absent lors de la prédiction : exécuter d'abord
  `uv run train-flight-models`.
