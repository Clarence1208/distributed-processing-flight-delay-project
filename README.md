# Prédiction des retards de vols

Projet étudiant de traitement distribué consacré aux vols domestiques américains
de 2024. L'objectif final est de prédire si un vol aura au moins 15 minutes de
retard, d'estimer le nombre de minutes et de présenter les facteurs possibles
dans un notebook Jupyter interactif.

Le code et les noms techniques sont en anglais. Les commentaires, les messages,
les rapports et la documentation sont en français.

## État du projet

| Étape | État | Contenu |
|---|---|---|
| 1. Parsing PySpark | Terminé | Échantillon reproductible de 10 000 vols, typage, contrôles qualité et Parquet |
| 2. Analyse PySpark | Terminé | Statistiques, valeurs manquantes, agrégations, causes et corrélations de Pearson |
| 3. Machine learning Python | Prototype non publiable | Congestion planifiée, historiques sans fuite, CatBoost et business gate strict |
| 4. Visualisation Jupyter | Terminé | Graphiques, explorateur, métriques métier et diagnostics bloqués si le gate échoue |

Le parcours complet se trouve dans le
[notebook principal](notebooks/flight_delay_pipeline.ipynb). Les résultats sont
également interprétés dans le
[rapport Spark](reports/spark_analysis.md) et le
[rapport du machine learning](reports/ml_training_report.md).

## Fonctionnalités actuellement implémentées

### Préparation Spark

- lecture des 35 colonnes CSV brutes avec un schéma explicite, sans inférence ;
- suppression immédiate de `late_aircraft_delay`, conservée uniquement dans le
  fichier source pour respecter son format ;
- tirage aléatoire exact et reproductible de 10 000 lignes avec la graine `42` ;
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
- répartition des minutes entre les quatre causes retenues ;
- statistiques descriptives des variables numériques ;
- matrice de corrélation de Pearson sur les vols achevés ;
- classement des corrélations avec `arr_delay` ;
- indicateur `available_before_departure` pour repérer les fuites de données ;
- exports CSV/JSON et commande `analyze-spark-data`.

### Machine learning Python

- parcours du CSV complet avec Polars en mode streaming, sans Spark ;
- échantillonnage déterministe configurable, fixé à 10 % pour la baseline ;
- exclusion des vols annulés, déroutés ou sans retard d'arrivée connu ;
- découpage chronologique : janvier-juillet pour l'entraînement, août pour
  l'early stopping, septembre-octobre pour le seuil et novembre-décembre pour
  le test ;
- six volumes de congestion calculés sur le planning complet avant
  échantillonnage, sans résultat réel du vol ;
- 68 features historiques calculées uniquement sur les jours antérieurs ;
- historiques glissants du réseau, de la compagnie, des aéroports et de la route ;
- fréquence, volume et gravité récente des perturbations ;
- classification d'un retard d'au moins 15 minutes avec `CatBoostClassifier` ;
- traitement natif des catégories et de leurs interactions, sans one-hot ;
- régression conditionnelle du nombre de minutes avec `CatBoostRegressor` ;
- quatre classifications conditionnelles pour les causes `carrier`, `weather`,
  `nas` et `security` ;
- seuil principal choisi uniquement sur septembre-octobre avec trois objectifs :
  précision d'au moins 50 %, rappel d'au moins 20 % et 5 à 10 % d'alertes ;
- bornes basses de Wilson à 95 %, support minimal de 500 alertes et contrôle
  séparé de novembre et décembre ;
- seuil de repli conservateur si le contrat est impossible, sans jamais valider
  le modèle dans ce cas ;
- métriques de classification, régression, confusion, baselines naïves et
  importance des features ;
- sauvegarde locale des modèles, historiques récents et métadonnées avec Joblib ;
- commandes reproductibles `train-flight-models` et `predict-flight`.

### Notebook Jupyter

- parcours guidé des quatre étapes dans un seul fichier ;
- aperçu du CSV brut et du Parquet propre ;
- tableaux des indicateurs, valeurs manquantes, causes et corrélations ;
- graphiques sur les mois, compagnies, aéroports, statuts et retards ;
- explorateur interactif des 10 000 vols avec filtres ;
- entraînement ML reproductible sur 10 % du CSV complet, avec mode démonstration
  configurable et rechargement d'un artefact v5 compatible ;
- métriques métier, stabilité mensuelle, matrice de confusion et importance des
  features ;
- formulaire modifiable pour produire un diagnostic de futur vol ; aucune alerte
  n'est publiée si le gate ou le contexte du planning exact manque.

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

Les features numériques de planning comprennent le jour, la semaine, les
encodages cycliques, les horaires prévus bruts et cycliques, la durée prévue et
la distance.

Les features catégorielles comprennent la compagnie, le numéro de vol, les
aéroports et États, la route, la combinaison compagnie-route, les heures et les
combinaisons aéroport-heure. CatBoost les traite directement.

Les six features de congestion planifiée sont :

- `origin_scheduled_departures_hour` et
  `origin_scheduled_departures_3h` ;
- `dest_scheduled_arrivals_hour` et `dest_scheduled_arrivals_3h` ;
- `route_scheduled_flights_day` ;
- `carrier_origin_scheduled_flights_day`.

Elles comptent tous les vols prévus, y compris les vols ensuite annulés ou
déroutés, car le planning est connu avant le départ. Pour un diagnostic isolé,
l'artefact fournit des profils typiques par saison, jour et créneau. Ceux-ci ne
suffisent toutefois pas à publier une alerte : l'application finale devra fournir
automatiquement les six volumes exacts du planning du jour.

Les 68 features historiques décrivent les 1, 3, 7, 14 ou 28 jours précédents,
selon le groupe : taux de retard, taux de perturbation, nombre de vols et durée
moyenne des retards. La fenêtre est fermée avant le jour cible. Une valeur du
jour courant ou du futur ne peut donc pas entrer dans ces calculs.

Les cibles ajoutées sont `is_delayed_15`, `delay_minutes`, `reason_carrier`,
`reason_weather`, `reason_nas` et `reason_security`.

`dep_delay`, les horaires réels, les durées réelles et les quatre colonnes de
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
├── notebooks/
│   └── flight_delay_pipeline.ipynb            # parcours complet en quatre étapes
├── reports/
│   ├── spark_analysis.md                     # rapport Spark versionné
│   └── ml_training_report.md                 # rapport ML versionné
├── src/flight_delays/
│   ├── parsing.py                            # étape 1, PySpark
│   ├── analysis.py                           # étape 2, PySpark
│   ├── ml_data.py                            # chargement et features Python
│   ├── schedule.py                           # congestion du planning et profils
│   ├── historical.py                         # historiques temporels sans fuite
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
uv sync --extra dev --extra notebook --python 3.11
```

`uv.lock` verrouille notamment PySpark, NumPy, Polars, pandas, scikit-learn,
CatBoost, JupyterLab, Seaborn, les widgets et Pytest afin que toute l'équipe
utilise le même environnement.

## Données disponibles

Le dépôt contient :

- `data/flight_data_2024_sample.csv` : 10 000 lignes pour tester Spark et le ML ;
- `data/flight_data_2024_data_dictionary.csv` : types et exemples.

Le fichier complet `data/flight_data_2024.csv` contient 7 079 081 vols et pèse
environ 1,2 Go. GitHub ne peut pas accueillir ce fichier ; chaque membre de
l'équipe doit le récupérer depuis la source commune du projet et le placer sous
ce nom exact. Il sert à reproduire les métriques ML mais ne passe jamais par
Spark.

## Ouvrir le notebook

```bash
uv run --extra notebook jupyter lab notebooks/flight_delay_pipeline.ipynb
```

Dans JupyterLab, utiliser **Run → Run All Cells**. Les étapes Spark utilisent les
10 000 lignes versionnées et l'étape ML utilise actuellement 10 % du CSV complet,
soit 696 596 vols achevés. Le grand CSV doit donc être placé dans `data/`. Pour
un essai rapide sans ce fichier, définir `USE_FULL_ML_DATA = False`. Le notebook
versionné contient déjà les derniers tableaux et graphiques exécutés ; les
widgets nécessitent toutefois JupyterLab.

## Exécution en ligne de commande

```bash
# 1. Vérifier les tests Spark et Python
uv run pytest

# 2. Préparer les 10 000 vols Spark
uv run prepare-spark-data --mode overwrite

# 3. Exécuter l'analyse Spark
uv run analyze-spark-data --mode overwrite

# 4a. Tester l'entraînement ML avec le petit CSV versionné
uv run train-flight-models \
  --input data/flight_data_2024_sample.csv \
  --sample-fraction 1

# 4b. Entraîner le modèle évalué sur 10 % du CSV complet
uv run train-flight-models --sample-fraction 0.1

# 5. Prédire le vol d'exemple avec le modèle local
uv run predict-flight --flight-json examples/flight.json
```

Après tout changement du schéma des features ou de la version d'artefact, il faut
relancer l'entraînement. La commande de prédiction d'exemple retourne un
diagnostic : elle ne publie pas d'alerte tant que le modèle n'a pas réussi le
business gate et que le planning journalier exact n'est pas fourni par une
application.

Si nécessaire, définir `JAVA_HOME` dans le même terminal avant les commandes
Spark et les tests complets. Le ML Python n'a pas besoin de Java.

Les sorties reproductibles sont créées dans :

```text
data/processed/spark/flights_10000/
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
├── feature_importance.csv
├── official/                    # modèle notebook entraîné sur 10 % du CSV complet
└── notebook_demo/               # modèle notebook entraîné sur le petit CSV
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

Sur les 10 000 vols :

- 2 119 vols achevés ont au moins 15 minutes de retard, soit 21,5 % ;
- 122 vols sont annulés et 42 sont déroutés ;
- la médiane du retard à l'arrivée est de -6 minutes ;
- parmi les quatre causes retenues, `carrier_delay` représente 54,2 % des
  minutes, `nas_delay` 33,7 %, `weather_delay` 12,0 % et `security_delay` 0,1 % ;
- `dep_delay` est très corrélé à `arr_delay` (`r = 0,966`), mais constitue une
  fuite de données pour un modèle pré-départ.

Ces résultats décrivent uniquement l'échantillon Spark de 10 000 lignes.

## Résultats ML actuels

Le pipeline parcourt les 7 079 081 lignes pour construire les historiques et la
congestion planifiée, puis retient de façon déterministe 696 596 vols achevés
comme cibles ML. Le protocole v5 sépare quatre périodes :

| Usage | Mois | Vols |
|---|---|---:|
| Apprentissage initial | janvier à juillet | 401 778 |
| Early stopping, puis réapprentissage | août | 60 378 |
| Choix du seuil | septembre à octobre | 118 740 |
| Test temporel | novembre à décembre | 115 700 |

Le modèle est publiable uniquement si les bornes basses à 95 % de la précision
et du rappel atteignent respectivement 50 % et 20 %, avec 5 à 10 % d'alertes et
au moins 500 alertes. Aucun seuil ne satisfait ces contraintes sur la validation.
Le seuil de repli `0,337` vise donc 7,5 % d'alertes sur cette validation, sans
changer le résultat du gate : **le modèle reste non publiable**.

| Ensemble | Précision | Borne basse 95 % | Rappel | Borne basse 95 % | Couverture | Alertes | TP | FP | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Validation | 31,1 % | 30,1 % | 16,3 % | 15,7 % | 7,5 % | 8 906 | 2 768 | 6 138 | Échoué |
| Test | 34,2 % | 33,3 % | 18,7 % | 18,2 % | 9,9 % | 11 433 | 3 908 | 7 525 | Échoué |

La stabilité mensuelle échoue également : novembre atteint 30,1 % de précision,
18,5 % de rappel et 9,0 % de couverture ; décembre atteint 37,5 %, 18,9 % et
10,7 %. La couverture de décembre dépasse donc la limite de 10 %.

La ROC-AUC du test vaut `0,648` et l'average precision `0,280`. Maximiser le F1
donnerait encore 40,1 % d'alertes, dont seulement 25,8 % correctes. Le nouveau
seuil réduit donc fortement la sur-alerte, mais il ne crée pas le signal manquant
nécessaire pour atteindre 50 % de précision.

La régression conditionnelle obtient une MAE de 43,46 minutes, contre 44,04 pour
la médiane, mais sa RMSE et son R² restent moins bons. `weather` atteint une
ROC-AUC de 0,714 ; `security` reste inexploitable avec seulement 86 cas et 0,8 %
de précision.

Le [rapport ML](reports/ml_training_report.md) détaille le protocole, les
variables, les résultats et les limites de déploiement.

## Travail restant

### Améliorations de l'étape 3

- ajouter des prévisions météo réellement connues avant le départ ;
- intégrer des indicateurs de trafic et de perturbation du jour même ;
- connecter automatiquement le planning complet du jour à l'interface ;
- actualiser quotidiennement les profils historiques utilisés en production ;
- entraîner la meilleure configuration sur davantage que 10 % des lignes ;
- valider une dernière fois sur un holdout 2025 jamais consulté ;
- améliorer l'explication locale des prédictions, par exemple avec SHAP ;
- étudier un modèle spécifique aux annulations, actuellement exclues.

### Évolution vers Streamlit

Streamlit n'est pas encore implémenté. La future application devra réutiliser le
business gate du pipeline, afficher un blocage rouge si le modèle n'est pas prêt
et ne jamais demander à l'utilisateur de saisir manuellement les six volumes de
planning.

## Limites connues

- L'analyse Spark repose volontairement sur seulement 10 000 lignes.
- Une corrélation ne démontre pas une causalité et Pearson ne mesure que les
  relations linéaires numériques.
- Les causes officielles sont connues après le vol et servent uniquement de
  cibles d'entraînement.
- Les probabilités de causes sont conditionnelles au fait que le vol soit en
  retard et plusieurs causes peuvent être proposées en même temps.
- La régression actuelle n'améliore que d'environ 0,59 minute la MAE de la médiane et
  dégrade les autres métriques.
- Les profils historiques doivent être actualisés pour rester pertinents après
  la fin de 2024 ; l'artefact local utilise les derniers profils disponibles.
- Le modèle ne possède ni météo prévisionnelle ni état opérationnel du jour même.
- Les profils typiques de planning permettent un diagnostic isolé, pas une alerte
  publiable ; celle-ci exige le planning exact du jour.
- Le benchmark novembre-décembre a servi à l'analyse d'ablation ; il ne constitue
  plus un holdout totalement vierge pour estimer la généralisation future.
- Les fichiers Joblib ne doivent être chargés que s'ils proviennent d'une source
  fiable et doivent être régénérés après une mise à jour des dépendances.
- Les widgets du notebook nécessitent l'exécution locale de JupyterLab et ne sont
  pas interactifs dans l'aperçu statique de GitHub.

## Résolution de problèmes

- `JAVA_GATEWAY_EXITED` ou Java introuvable : vérifier Java 17 et `JAVA_HOME`.
- Sortie Spark déjà existante : ajouter `--mode overwrite`.
- Mémoire Spark insuffisante : conserver le master par défaut `local[2]`.
- CSV complet absent : utiliser le petit CSV versionné pour tester le ML.
- Mémoire limitée pendant le ML : réduire `--sample-fraction`, par exemple à
  `0.02`. Le calcul des historiques et du planning parcourt malgré tout le CSV
  complet ; prévoir environ 4 Go de mémoire disponible pour le run officiel.
- Modèle absent lors de la prédiction : exécuter d'abord
  `uv run train-flight-models`.
- Notebook sans kernel Python : relancer
  `uv sync --extra dev --extra notebook --python 3.11`.
