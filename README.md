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
| 3. Machine learning Python | À faire | Classification, régression, explication des facteurs et évaluation |
| 4. Streamlit | À faire | Tableau de bord, graphiques et formulaire de prédiction |

Le [rapport de l'analyse Spark](reports/spark_analysis.md) contient les résultats
interprétés et les précautions concernant les corrélations.

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
- indicateur `available_before_departure` pour repérer les futures fuites de
  données ;
- exports CSV/JSON et commande `analyze-spark-data`.

### Colonnes et features ajoutées

| Colonne | Rôle | Persistée dans le Parquet propre |
|---|---|:---:|
| `scheduled_departure_minutes` | Heure de départ prévue convertie en minutes depuis minuit | Oui |
| `scheduled_arrival_minutes` | Heure d'arrivée prévue convertie en minutes depuis minuit | Oui |
| `validation_errors` | Liste des anomalies d'une ligne rejetée | Quarantaine uniquement |
| `is_valid_row` | Résultat des contrôles qualité | Quarantaine uniquement |
| `flight_status` | Statut exclusif utilisé pour l'analyse | Non, calculé pendant l'analyse |

Pour une prédiction effectuée avant le départ, les variables numériques déjà
identifiées comme disponibles sont notamment `month`, `day_of_month`,
`day_of_week`, les deux horaires prévus en minutes, `crs_elapsed_time` et
`distance`. Les variables catégorielles `op_unique_carrier`, `origin` et `dest`
seront également importantes après encodage.

`dep_delay`, les durées réelles et les cinq colonnes de causes sont connues
pendant ou après le vol. Elles ne devront pas être utilisées comme entrées du
modèle pré-départ.

## Structure du dépôt

```text
.
├── data/
│   ├── flight_data_2024_sample.csv           # échantillon de 10 000 lignes versionné
│   ├── flight_data_2024_data_dictionary.csv  # description des colonnes versionnée
│   ├── flight_data_2024.csv                  # dataset ML complet, ignoré par Git
│   ├── processed/                            # sorties du parsing, ignorées
│   └── analysis/                             # sorties de l'analyse, ignorées
├── reports/
│   └── spark_analysis.md                     # rapport interprété versionné
├── src/flight_delays/
│   ├── parsing.py
│   └── analysis.py
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

Sur Windows PowerShell, adapter également le chemin :

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-17"
```

### 4. Créer l'environnement

```bash
uv sync --extra dev --python 3.11
```

`uv.lock` verrouille les versions de PySpark, NumPy, Pytest et leurs dépendances
afin que toute l'équipe utilise le même environnement.

## Données disponibles

Le dépôt contient tout ce qui est nécessaire pour les étapes Spark :

- `data/flight_data_2024_sample.csv` : 10 000 lignes ;
- `data/flight_data_2024_data_dictionary.csv` : types et exemples.

Le fichier complet `data/flight_data_2024.csv` contient 7 079 081 vols et pèse
environ 1,2 Go. GitHub ne peut pas accueillir ce fichier ; chaque membre de
l'équipe doit le récupérer depuis la source commune du projet et le placer sous
ce nom exact. Il ne sera nécessaire qu'à partir de l'étape 3 et ne passera pas
par Spark.

## Exécution rapide

Les commandes suivantes suffisent pour reproduire les étapes terminées :

```bash
# 1. Vérifier le projet
uv run pytest

# 2. Préparer les 2 000 vols Spark
uv run prepare-spark-data --mode overwrite

# 3. Exécuter l'analyse Spark
uv run analyze-spark-data --mode overwrite
```

Si nécessaire, définir `JAVA_HOME` dans le même terminal avant ces commandes.

Les sorties sont créées dans :

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
```

Ces dossiers sont ignorés par Git et peuvent être supprimés puis régénérés à
tout moment.

Pour consulter toutes les options :

```bash
uv run prepare-spark-data --help
uv run analyze-spark-data --help
```

## Résultats Spark actuels

Sur les 2 000 vols :

- 420 vols achevés ont au moins 15 minutes de retard, soit 21,3 % ;
- 25 vols sont annulés et 4 sont déroutés ;
- la médiane du retard à l'arrivée est de -6 minutes ;
- `late_aircraft_delay` et `carrier_delay` représentent ensemble environ 75,4 %
  des minutes de causes enregistrées ;
- `dep_delay` est très corrélé à `arr_delay` (`r = 0,983`), mais n'est pas connu
  avant le départ et constitue donc une fuite de données pour le futur modèle.

Ces résultats décrivent seulement l'échantillon Spark. Ils devront être vérifiés
sur le dataset complet pendant l'étape ML.

## Travail restant

### Étape 3 — machine learning en Python

- charger et nettoyer le CSV complet sans Spark ;
- définir trois objectifs distincts :
  - classification du retard à l'arrivée supérieur ou égal à 15 minutes ;
  - régression du nombre de minutes de retard ;
  - estimation des raisons probables du retard ;
- construire uniquement des features disponibles au moment de la prédiction ;
- encoder les compagnies, aéroports et routes ;
- créer des agrégats historiques sans utiliser le futur ;
- effectuer un découpage chronologique entraînement/validation/test ;
- établir des modèles de référence puis comparer plusieurs algorithmes ;
- traiter le déséquilibre entre vols à l'heure et vols en retard ;
- évaluer la classification avec précision, rappel, F1, ROC-AUC et PR-AUC ;
- évaluer la régression avec MAE, RMSE et éventuellement `R²` ;
- expliquer les prédictions avec importance des variables ou valeurs SHAP ;
- sauvegarder le préprocesseur, les modèles et leurs métadonnées.

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
- Une corrélation ne démontre pas une causalité.
- Pearson ne mesure que les relations linéaires entre variables numériques.
- Les causes officielles de retard sont connues après le vol et ne sont pas des
  features pré-départ.
- Le modèle et l'interface ne sont pas encore implémentés.

## Résolution de problèmes

- `JAVA_GATEWAY_EXITED` ou Java introuvable : vérifier Java 17 et `JAVA_HOME`.
- Sortie déjà existante : ajouter `--mode overwrite`.
- Mémoire insuffisante : conserver le master par défaut `local[2]`.
- CSV complet absent : normal pour les étapes 1 et 2 ; il n'est requis que pour
  le machine learning.
