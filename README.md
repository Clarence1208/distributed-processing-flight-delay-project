# Prédiction des retards de vols

Projet étudiant de traitement distribué consacré aux vols domestiques américains
de 2024. L'objectif est de prédire avant le départ si un vol arrivera en retard.

La définition retenue est volontairement directe :

```text
is_delayed = 1 si arr_delay > 0, sinon 0
```

Un retard d'une minute compte donc comme un retard. Les causes et l'estimation
du nombre de minutes ont été retirées afin de concentrer le projet sur cette
classification binaire.

Le code et les noms techniques sont en anglais. Les commentaires, les messages,
les rapports, le notebook et l'interface sont en français.

## État du projet

| Étape | État | Contenu |
|---|---|---|
| 1. Parsing PySpark | Terminé | Typage, qualité, quarantaine et Parquet sur 10 000 vols |
| 2. Analyse PySpark | Terminé | Statistiques, valeurs manquantes, agrégations et corrélations |
| 3. Machine learning Python | Expérimental | Classifieur CatBoost v6, historiques, planning et business gate |
| 4. Streamlit | Terminé | Dashboard, explorateur, performances et formulaire protégé |

Le parcours pédagogique complet se trouve dans le
[notebook principal](notebooks/flight_delay_pipeline.ipynb). Les interprétations
détaillées sont disponibles dans le [rapport Spark](reports/spark_analysis.md)
et le [rapport du modèle](reports/ml_training_report.md).

## Résultats actuels

Le CSV complet contient 7 079 081 vols. Le ML parcourt ce fichier pour calculer
les historiques et le planning, puis conserve 696 596 vols achevés avec un
échantillonnage déterministe à 10 %.

Sur le test novembre-décembre de 115 700 vols :

| Métrique | Résultat |
|---|---:|
| Précision | **50,89 %** |
| Borne basse de précision à 95 % | **50,19 %** |
| Rappel | **26,36 %** |
| Couverture des alertes | **16,97 %** |
| Alertes | 19 629 |
| ROC-AUC | 0,643 |

Le test global dépasse donc 50 % de précision. Le modèle reste néanmoins non
publiable : la validation n'atteint que 49,08 % et les performances sont
instables entre novembre et décembre.

Le seuil maximisant F1 a été rejeté, car il signalerait 66,7 % des vols avec
seulement 38,7 % de précision.

## Fonctionnalités

### Parsing PySpark

- lecture CSV avec schéma textuel explicite, sans inférence silencieuse ;
- échantillon exact et reproductible de 10 000 lignes, graine `42` ;
- conversion contrôlée des dates, heures, entiers, nombres et textes ;
- validation des champs obligatoires, plages, heures HHMM et indicateurs ;
- séparation entre lignes propres et quarantaine avec motifs d'erreur ;
- écriture Parquet partitionnée par année et mois ;
- export d'un rapport qualité JSON ;
- commande reproductible `prepare-spark-data`.

Le schéma source conserve les 35 colonnes afin de lire correctement le CSV, mais
les cinq causes postérieures au vol sont exclues du Parquet propre :
`carrier_delay`, `weather_delay`, `nas_delay`, `security_delay` et
`late_aircraft_delay`.

### Analyse PySpark

- taux de retard pour `arr_delay > 0` ;
- annulations, déroutements, moyenne, médiane, percentiles et maximum ;
- valeurs manquantes ;
- indicateurs par mois, compagnie et aéroport de départ ;
- statistiques descriptives numériques ;
- matrice de corrélation de Pearson ;
- indication des variables réellement disponibles avant le départ ;
- exports CSV/JSON et commande `analyze-spark-data`.

Sur l'échantillon Spark : 9 836 vols sont achevés et 3 578 arrivent en retard,
soit 36,38 %.

### Machine learning Python

- lecture du CSV complet avec Polars en mode streaming, sans Spark ;
- exclusion des vols annulés, déroutés ou sans retard d'arrivée connu ;
- cible unique `is_delayed` dérivée de `arr_delay > 0` ;
- découpage chronologique sans mélange du futur dans le passé ;
- classifieur `CatBoostClassifier`, sans one-hot encoding ;
- six volumes de congestion calculés avant échantillonnage ;
- historiques glissants construits uniquement sur les jours précédents ;
- seuil choisi uniquement sur septembre-octobre ;
- contrôle séparé du test novembre-décembre et de chaque mois ;
- intervalles de Wilson à 95 % ;
- artefact Joblib v6 contenant un seul modèle ;
- commandes `train-flight-models` et `predict-flight`.

Le contrat métier demande :

- précision basse à 95 % ≥ 50 % ;
- rappel bas à 95 % ≥ 20 % ;
- entre 5 et 20 % des vols signalés ;
- au moins 500 alertes ;
- stabilité sur la validation, le test et chaque mois de test.

### Features du classifieur

Les features statiques comprennent la date, le jour, les encodages cycliques,
les horaires prévus, la durée prévue, la distance, le numéro de vol, la
compagnie, les aéroports, les États et la route.

Les six features de congestion planifiée sont :

- `origin_scheduled_departures_hour` ;
- `origin_scheduled_departures_3h` ;
- `dest_scheduled_arrivals_hour` ;
- `dest_scheduled_arrivals_3h` ;
- `route_scheduled_flights_day` ;
- `carrier_origin_scheduled_flights_day`.

Les historiques utilisent des fenêtres de 1 à 28 jours pour le réseau, la
compagnie, l'origine, la destination et la route. Chaque fenêtre fournit taux
de retard, taux de perturbation et volume de vols.

Le code compagnie `op_unique_carrier` reste bien utilisé. Les horaires réels,
`dep_delay`, `arr_delay`, les durées réelles et les causes sont interdits comme
features pré-départ.

### Interface Streamlit

L'application [streamlit_app.py](streamlit_app.py) contient quatre pages :

1. **Vue d'ensemble** : chiffres Spark, saisonnalité et résultat du modèle ;
2. **Explorer les vols** : filtres par mois, compagnie et aéroport, graphiques et
   table des 10 000 vols ;
3. **Performance du modèle** : gate métier, validation/test, confusion,
   stabilité mensuelle, comparaison avec F1 et importance des features ;
4. **Diagnostic d'un vol** : formulaire pré-départ, probabilité, seuil et
   explication des blocages.

L'application ne présente jamais une classe expérimentale comme une décision
publique. `published_delay_alert` reste `None` tant que le business gate échoue
ou que le planning journalier exact manque.

L'artefact officiel v6 de 4 Mo est versionné pour que le formulaire fonctionne
dès le clonage. Si cet artefact est supprimé ou incompatible, les graphiques
restent disponibles grâce à un instantané compact des métriques v6 et
l'interface explique comment réentraîner le modèle au lieu de planter.

## Structure du dépôt

```text
.
├── data/
│   ├── flight_data_2024_sample.csv
│   ├── flight_data_2024_data_dictionary.csv
│   ├── flight_data_2024.csv                 # complet, ignoré par Git
│   ├── processed/                           # sorties Spark, ignorées
│   └── analysis/                            # agrégats Spark, ignorés
├── examples/
│   └── flight.json
├── notebooks/
│   └── flight_delay_pipeline.ipynb
├── reports/
│   ├── spark_analysis.md
│   └── ml_training_report.md
├── models/official/
│   └── flight_delay_models.joblib          # artefact v6 utilisable par Streamlit
├── scripts/
│   └── generate_notebook.py
├── src/flight_delays/
│   ├── parsing.py
│   ├── analysis.py
│   ├── ml_data.py
│   ├── historical.py
│   ├── schedule.py
│   ├── training.py
│   ├── prediction.py
│   └── dashboard.py
├── tests/
├── streamlit_app.py
├── pyproject.toml
└── uv.lock
```

## Installation sur un nouveau PC

Prérequis : Python 3.11, Java 17, Git et `uv`.

```bash
git clone https://github.com/Clarence1208/distributed-processing-flight-delay-project.git
cd distributed-processing-flight-delay-project
python3 -m pip install uv
uv sync --extra dev --extra notebook --python 3.11
```

Configurer Java 17.

Sur macOS :

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

Sur Linux :

```bash
export JAVA_HOME=/chemin/vers/jdk-17
```

Sur Windows PowerShell :

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-17"
```

Le dépôt contient l'échantillon de 10 000 lignes. Pour reproduire
l'entraînement officiel, placer le CSV complet sous le nom exact :

```text
data/flight_data_2024.csv
```

Ce fichier pèse environ 1,2 Go et n'est pas envoyé sur GitHub.

## Lancer le projet

### Notebook complet

```bash
uv run --extra notebook jupyter lab notebooks/flight_delay_pipeline.ipynb
```

Puis utiliser **Run → Run All Cells**.

### Interface Streamlit

```bash
uv run streamlit run streamlit_app.py
```

Streamlit affiche ensuite l'adresse locale, généralement
`http://localhost:8501`.

### Parsing et analyse Spark

```bash
uv run prepare-spark-data --mode overwrite
uv run analyze-spark-data --mode overwrite
```

### Réentraînement officiel sur 10 %

```bash
uv run train-flight-models \
  --input data/flight_data_2024.csv \
  --sample-fraction 0.1 \
  --seed 42
```

Les sorties sont écrites dans `models/official/`. Seul l'artefact officiel v6
est versionné ; les métriques et importances régénérées restent locales.

### Prédiction en ligne de commande

```bash
uv run predict-flight --flight-json examples/flight.json
```

## Tests

```bash
uv run pytest
```

Les tests couvrent le parsing et l'analyse Spark, les features temporelles, le
planning, l'entraînement, le contrat de prédiction, le notebook et les quatre
pages Streamlit.

## Limites et prochaines étapes

- Le test global dépasse 50 % de précision, mais la validation et chaque mois ne
  passent pas encore ensemble le business gate.
- Les historiques de l'artefact sont arrêtés au 31 décembre 2024.
- Le planning exact doit être connecté automatiquement pour une vraie alerte.
- L'état opérationnel du jour et les avis NAS connus avant le départ pourraient
  améliorer la stabilité.
- Une année 2025 jamais consultée est nécessaire pour la validation finale.
- Les annulations devraient être traitées par un modèle séparé.

La météo et la sécurité ne seront réintroduites que si des données réellement
disponibles avant le départ sont ajoutées au projet.
