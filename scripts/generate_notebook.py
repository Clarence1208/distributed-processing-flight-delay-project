"""Génère le notebook pédagogique principal du projet."""

from pathlib import Path

import nbformat as nbf


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# Prédiction des retards de vols domestiques US

Ce notebook rassemble le projet en quatre étapes reproductibles. La cible métier est maintenant simple : **un vol est en retard lorsque `arr_delay > 0`**. Les causes de retard et l'estimation des minutes ne font plus partie du machine learning.

Le code et les noms techniques restent en anglais ; les explications et les graphiques sont en français."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SPARK_SAMPLE_SIZE = 10_000
SPARK_SAMPLE_SEED = 42
ML_SAMPLE_FRACTION = 0.1
RETRAIN_MODEL = False

RAW_SAMPLE_PATH = PROJECT_ROOT / "data/flight_data_2024_sample.csv"
FULL_DATA_PATH = PROJECT_ROOT / "data/flight_data_2024.csv"
SPARK_OUTPUT = PROJECT_ROOT / "data/processed/spark/flights_10000"
ANALYSIS_OUTPUT = PROJECT_ROOT / "data/analysis/spark"
MODEL_DIRECTORY = PROJECT_ROOT / "models/official"

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 120)

display(Markdown(f"**Racine du projet :** `{PROJECT_ROOT}`"))
display(Markdown(f"**CSV complet disponible :** {'oui' if FULL_DATA_PATH.exists() else 'non'}"))"""
    )
)

cells.append(nbf.v4.new_markdown_cell("""# Étape 1 — Parsing avec PySpark

Spark lit les 10 000 lignes avec un schéma textuel explicite, convertit les types, contrôle les règles de qualité puis écrit les lignes valides en Parquet. Les cinq colonnes de causes postérieures au vol sont exclues du dataset propre."""))

cells.append(
    nbf.v4.new_code_cell(
        """from flight_delays.parsing import (
    create_spark_session,
    parse_and_validate,
    read_flights_csv,
    sample_rows,
    split_rows,
    write_results,
)

spark = create_spark_session(
    application_name="Notebook des retards de vols",
    master="local[2]",
)
spark.sparkContext.setLogLevel("WARN")

raw_spark = read_flights_csv(spark, str(RAW_SAMPLE_PATH))
sampled_spark = sample_rows(raw_spark, SPARK_SAMPLE_SIZE, SPARK_SAMPLE_SEED)
validated_spark = parse_and_validate(sampled_spark).cache()
valid_spark, rejected_spark = split_rows(validated_spark)

valid_count = valid_spark.count()
rejected_count = rejected_spark.count()
write_results(validated_spark, str(SPARK_OUTPUT), mode="overwrite")

print(f"Lignes valides : {valid_count:,}")
print(f"Lignes en quarantaine : {rejected_count:,}")"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """removed_cause_columns = {
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
}
raw_preview = pd.read_csv(RAW_SAMPLE_PATH, nrows=8).drop(
    columns=list(removed_cause_columns),
    errors="ignore",
)
clean_spark = spark.read.parquet(str(SPARK_OUTPUT / "flights"))
clean_preview = clean_spark.limit(8).toPandas()

display(Markdown("### Aperçu du CSV brut sans les causes exclues"), raw_preview)
display(Markdown("### Aperçu du Parquet propre"), clean_preview)

print("Causes absentes du Parquet :", removed_cause_columns.isdisjoint(clean_spark.columns))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Le CSV brut conserve son format d'origine, mais les causes officielles ne sont plus propagées dans le Parquet. Elles sont constatées après le vol et ne répondent pas à l'objectif binaire retenu."""
    )
)

cells.append(nbf.v4.new_markdown_cell("""# Étape 2 — Analyse avec PySpark

L'analyse utilise désormais la même définition que le modèle : tout retard d'arrivée strictement positif. Elle décrit les statuts, les variations mensuelles, les compagnies, les aéroports et les corrélations, sans analyser les causes."""))

cells.append(
    nbf.v4.new_code_cell(
        """from flight_delays.analysis import (
    build_group_metrics,
    build_missing_values,
    build_status_distribution,
    write_analysis,
)

overview_spark, correlations_spark, correlation_row_count = write_analysis(
    clean_spark,
    str(ANALYSIS_OUTPUT),
    mode="overwrite",
)
overview = overview_spark.first().asDict()
status_metrics = build_status_distribution(clean_spark).toPandas()
monthly_metrics = build_group_metrics(clean_spark, "month").orderBy("month").toPandas()
carrier_metrics = build_group_metrics(clean_spark, "op_unique_carrier").toPandas()
origin_metrics = build_group_metrics(clean_spark, "origin").toPandas()
missing_values = build_missing_values(clean_spark).toPandas()
arrival_correlations = correlations_spark.toPandas()

overview_table = pd.DataFrame(
    {
        "Indicateur": [
            "Vols analysés",
            "Vols achevés",
            "Vols arrivés en retard",
            "Taux de retard",
            "Vols annulés",
            "Vols déroutés",
            "Retard moyen",
            "Retard médian",
        ],
        "Valeur": [
            f"{overview['flight_count']:,}",
            f"{overview['completed_flight_count']:,}",
            f"{overview['delayed_flight_count']:,}",
            f"{overview['delayed_flight_percentage']:.1f} %",
            f"{overview['cancelled_flight_count']:,}",
            f"{overview['diverted_flight_count']:,}",
            f"{overview['average_arrival_delay_minutes']:.1f} min",
            f"{overview['median_arrival_delay_minutes']:.0f} min",
        ],
    }
)
display(overview_table)
print(f"Vols utilisés pour les corrélations : {correlation_row_count:,}")"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(2, 2, figsize=(16, 11))

sns.lineplot(
    data=monthly_metrics,
    x="month",
    y="delayed_flight_percentage",
    marker="o",
    ax=axes[0, 0],
)
axes[0, 0].set(title="Taux de retard par mois", xlabel="Mois", ylabel="Taux (%)")

top_carriers = carrier_metrics.head(10).sort_values("delayed_flight_percentage")
sns.barplot(
    data=top_carriers,
    y="op_unique_carrier",
    x="delayed_flight_percentage",
    ax=axes[0, 1],
)
axes[0, 1].set(title="Compagnies les plus représentées", xlabel="Taux (%)", ylabel="Compagnie")

top_origins = origin_metrics.head(10).sort_values("delayed_flight_percentage")
sns.barplot(
    data=top_origins,
    y="origin",
    x="delayed_flight_percentage",
    ax=axes[1, 0],
)
axes[1, 0].set(title="Principaux aéroports de départ", xlabel="Taux (%)", ylabel="Aéroport")

correlation_plot = arrival_correlations.dropna().head(10).sort_values("correlation")
sns.barplot(
    data=correlation_plot,
    y="variable",
    x="correlation",
    hue="available_before_departure",
    ax=axes[1, 1],
)
axes[1, 1].set(title="Corrélations avec arr_delay", xlabel="Pearson", ylabel="Variable")
axes[1, 1].legend(title="Disponible avant départ")

plt.tight_layout()
plt.show()

display(Markdown("### Valeurs manquantes les plus fréquentes"), missing_values.head(12))
display(Markdown("### Corrélations avec le retard d'arrivée"), arrival_correlations)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """La corrélation de `dep_delay` avec `arr_delay` est très forte, mais `dep_delay` n'est connu qu'après le départ. Les variables réellement disponibles à l'avance ont des corrélations linéaires faibles ; le modèle doit donc exploiter leurs interactions et les historiques récents."""
    )
)

cells.append(nbf.v4.new_code_cell("""clean_spark.unpersist()
spark.stop()
print("Session Spark arrêtée.")"""))

cells.append(nbf.v4.new_markdown_cell("""# Étape 3 — Machine learning avec Python

Le pipeline Python parcourt le CSV complet pour construire le planning et les historiques, puis entraîne un unique `CatBoostClassifier` sur un échantillon déterministe de 10 %. La cible `is_delayed` vaut 1 lorsque `arr_delay > 0`.

Le contrat métier demande au moins **50 % de précision**, **20 % de rappel** et une couverture comprise entre **5 et 20 %**. Les bornes basses de confiance à 95 % et la stabilité mensuelle sont également contrôlées."""))

cells.append(
    nbf.v4.new_code_cell(
        """from flight_delays.ml_data import load_ml_data
from flight_delays.prediction import (
    CURRENT_ARTIFACT_VERSION,
    load_model_bundle,
    predict_flight,
)
from flight_delays.training import save_training_outputs, train_models

MODEL_PATH = MODEL_DIRECTORY / "flight_delay_models.joblib"
METRICS_PATH = MODEL_DIRECTORY / "training_metrics.json"
IMPORTANCE_PATH = MODEL_DIRECTORY / "feature_importance.csv"

artifact_is_compatible = False
if MODEL_PATH.exists() and METRICS_PATH.exists() and IMPORTANCE_PATH.exists():
    try:
        bundle = load_model_bundle(str(MODEL_PATH))
        artifact_is_compatible = True
    except ValueError as error:
        print(error)

if RETRAIN_MODEL or not artifact_is_compatible:
    ml_input = FULL_DATA_PATH if FULL_DATA_PATH.exists() else RAW_SAMPLE_PATH
    sample_fraction = ML_SAMPLE_FRACTION if FULL_DATA_PATH.exists() else 1.0
    print(f"Préparation ML depuis {ml_input.name} avec fraction {sample_fraction:.0%}…")
    ml_data = load_ml_data(str(ml_input), sample_fraction=sample_fraction, seed=42)
    bundle, training_metrics, feature_importance = train_models(ml_data, seed=42)
    save_training_outputs(
        bundle,
        training_metrics,
        feature_importance,
        str(MODEL_PATH),
        str(METRICS_PATH),
        str(IMPORTANCE_PATH),
    )
else:
    training_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    feature_importance = pd.read_csv(IMPORTANCE_PATH)
    print(f"Artefact v{CURRENT_ARTIFACT_VERSION} rechargé sans réentraînement.")

print("Définition de la cible :", bundle["target_definition"])
print("Nombre de modèles dans l'artefact : 1 classifieur")"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """split_labels = {
    "train": "Entraînement (janvier-juillet)",
    "tuning": "Réglage (août)",
    "validation": "Validation du seuil (septembre-octobre)",
    "test": "Test (novembre-décembre)",
}
split_table = pd.DataFrame(
    [
        {
            "Jeu": split_labels[name],
            "Vols": values["row_count"],
            "Vols retardés": values["delayed_count"],
            "Taux de retard (%)": values["delayed_percentage"],
        }
        for name, values in training_metrics["data"].items()
        if name in split_labels
    ]
)
display(split_table.style.format({"Taux de retard (%)": "{:.2f}"}))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """classification = training_metrics["delay_classification"]
business_gate = classification["business_gate"]
validation_metrics = classification["validation"]
test_metrics = classification["test"]

performance_table = pd.DataFrame(
    [
        {
            "Jeu": "Validation",
            "Précision (%)": validation_metrics["precision"] * 100,
            "Rappel (%)": validation_metrics["recall"] * 100,
            "Couverture (%)": validation_metrics["alert_coverage"] * 100,
            "Alertes": validation_metrics["alert_count"],
            "ROC-AUC": validation_metrics["roc_auc"],
        },
        {
            "Jeu": "Test",
            "Précision (%)": test_metrics["precision"] * 100,
            "Rappel (%)": test_metrics["recall"] * 100,
            "Couverture (%)": test_metrics["alert_coverage"] * 100,
            "Alertes": test_metrics["alert_count"],
            "ROC-AUC": test_metrics["roc_auc"],
        },
    ]
)
display(performance_table.style.format(precision=2))

if business_gate["passed"]:
    display(Markdown("✅ **Le modèle respecte le contrat métier complet.**"))
else:
    display(Markdown("⚠️ **Aucune alerte publiable : le modèle reste expérimental.**"))
    for violation in business_gate["violations"]:
        display(Markdown(f"- {violation}"))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(18, 5))

confusion = np.array(
    [
        [test_metrics["true_negative"], test_metrics["false_positive"]],
        [test_metrics["false_negative"], test_metrics["true_positive"]],
    ]
)
sns.heatmap(
    confusion,
    annot=True,
    fmt=",",
    cmap="Blues",
    xticklabels=["À l'heure", "Retard"],
    yticklabels=["À l'heure", "Retard"],
    ax=axes[0],
)
axes[0].set(title="Matrice de confusion — test", xlabel="Prédit", ylabel="Réel")

monthly_rows = []
for month, values in classification["test_by_month"].items():
    monthly_rows.extend(
        [
            {"Mois": int(month), "Métrique": "Précision", "Valeur": values["precision"] * 100},
            {"Mois": int(month), "Métrique": "Rappel", "Valeur": values["recall"] * 100},
            {"Mois": int(month), "Métrique": "Couverture", "Valeur": values["alert_coverage"] * 100},
        ]
    )
sns.barplot(data=pd.DataFrame(monthly_rows), x="Mois", y="Valeur", hue="Métrique", ax=axes[1])
axes[1].set(title="Stabilité mensuelle", ylabel="Pourcentage")

f1_test = classification["f1_threshold_comparison"]["test"]
threshold_comparison = pd.DataFrame(
    [
        {"Seuil": "Métier", "Précision": test_metrics["precision"] * 100, "Rappel": test_metrics["recall"] * 100, "Couverture": test_metrics["alert_coverage"] * 100},
        {"Seuil": "F1", "Précision": f1_test["precision"] * 100, "Rappel": f1_test["recall"] * 100, "Couverture": f1_test["alert_coverage"] * 100},
    ]
).melt(id_vars="Seuil", var_name="Métrique", value_name="Valeur")
sns.barplot(data=threshold_comparison, x="Métrique", y="Valeur", hue="Seuil", ax=axes[2])
axes[2].set(title="Seuil métier contre seuil F1", ylabel="Pourcentage")

plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """top_importance = feature_importance.head(20).sort_values("importance")
plt.figure(figsize=(10, 8))
sns.barplot(data=top_importance, x="importance", y="feature")
plt.title("Vingt features les plus importantes")
plt.xlabel("Importance CatBoost")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

display(feature_importance.head(20))"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """example_flight = json.loads(
    (PROJECT_ROOT / "examples/flight.json").read_text(encoding="utf-8")
)
prediction = predict_flight(bundle, example_flight)

prediction_table = pd.DataFrame(
    {
        "Indicateur": [
            "Probabilité diagnostique de retard",
            "Seuil du modèle",
            "Classe interne non publiable",
            "Prédiction publiable",
            "Contexte du planning",
        ],
        "Valeur": [
            f"{prediction['delay_probability']:.1%}",
            f"{prediction['classification_threshold']:.1%}",
            "Retard" if prediction["diagnostic_is_delayed_prediction"] else "À l'heure",
            "Oui" if prediction["prediction_publishable"] else "Non",
            prediction["schedule_context_source"],
        ],
    }
)
display(prediction_table)

if not prediction["prediction_publishable"]:
    display(Markdown("**Aucune alerte publiable. Raisons :**"))
    for blocker in prediction["publication_blockers"]:
        display(Markdown(f"- {blocker}"))"""
    )
)

cells.append(nbf.v4.new_markdown_cell("""# Étape 4 — Visualisation et prédiction avec Streamlit

L'interface finale se trouve dans `streamlit_app.py`. Elle contient quatre pages : vue d'ensemble, explorateur des vols, performances du modèle et formulaire de diagnostic.

Elle ne relance ni Spark ni l'entraînement à chaque interaction. Elle utilise l'échantillon versionné pour les graphiques et charge l'artefact v6 en cache pour la prédiction. Si l'artefact local manque ou est incompatible, le reste du dashboard continue de fonctionner."""))

cells.append(
    nbf.v4.new_code_cell(
        """streamlit_command = "uv run streamlit run streamlit_app.py"
display(Markdown("### Lancer l'interface"))
print(streamlit_command)

streamlit_pages = pd.DataFrame(
    {
        "Page": [
            "Vue d'ensemble",
            "Explorer les vols",
            "Performance du modèle",
            "Diagnostic d'un vol",
        ],
        "Contenu": [
            "Indicateurs Spark et résultat ML",
            "Filtres, graphiques et table des 10 000 vols",
            "Gate métier, confusion, stabilité et importance",
            "Formulaire pré-départ et diagnostic protégé",
        ],
    }
)
display(streamlit_pages)"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        '''test_precision = test_metrics["precision"]
test_recall = test_metrics["recall"]
test_coverage = test_metrics["alert_coverage"]
publication_status = (
    "Le contrat métier complet est respecté."
    if business_gate["passed"]
    else "Le contrat métier complet n'est pas respecté : le modèle reste expérimental."
)

display(
    Markdown(
        f"""## Conclusion

Le run chargé obtient **{test_precision:.1%} de précision**, **{test_recall:.1%} de rappel** et signale **{test_coverage:.1%} des vols** sur son test. {publication_status} Streamlit expose ce statut au lieu de transformer le diagnostic en promesse certaine."""
    )
)'''
    )
)

notebook["cells"] = cells
output = Path(__file__).resolve().parents[1] / "notebooks/flight_delay_pipeline.ipynb"
nbf.write(notebook, output)
print(f"Notebook généré : {output}")
