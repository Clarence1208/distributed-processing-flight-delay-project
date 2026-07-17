"""Entraînement des modèles de retard et de causes avec CatBoost."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import joblib
import numpy as np
import pandas as pd
import polars
import sklearn
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from flight_delays.historical import build_history_profiles
from flight_delays.ml_data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MAX_PREDICTED_DELAY_MINUTES,
    REASON_TARGETS,
    load_ml_data,
    split_temporally,
)


DELAY_CLASSIFICATION_FEATURES = [
    name for name in FEATURE_COLUMNS if "_average_delay_minutes_" not in name
]


def _model_features(
    data: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Garantit les types attendus par CatBoost sans encodage one-hot."""

    selected_columns = columns or FEATURE_COLUMNS
    features = data[selected_columns].copy()
    for name in set(CATEGORICAL_FEATURES).intersection(selected_columns):
        features[name] = features[name].fillna("__missing__").astype(str)
    return features


def _new_classifier(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=8,
        random_seed=seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=80,
    )


def _fit_binary_classifier(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    validation_features: pd.DataFrame,
    validation_target: np.ndarray,
    seed: int,
):
    unique = np.unique(train_target)
    if len(unique) == 1:
        return DummyClassifier(
            strategy="constant", constant=int(unique[0])
        ).fit(train_features, train_target)

    model = _new_classifier(seed)
    fit_arguments: dict[str, Any] = {
        "X": train_features,
        "y": train_target,
        "cat_features": [
            name for name in CATEGORICAL_FEATURES if name in train_features.columns
        ],
    }
    if len(np.unique(validation_target)) == 2:
        fit_arguments.update(
            {
                "eval_set": (validation_features, validation_target),
                "use_best_model": True,
            }
        )
    model.fit(**fit_arguments)
    return model


def _positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return model.predict_proba(features)[:, classes.index(1)]


def _best_f1_threshold(target: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(target)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.clip(thresholds[int(np.argmax(scores))], 0.001, 0.999))


def _classification_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "sample_count": int(len(target)),
        "positive_count": int(target.sum()),
        "positive_percentage": float(target.mean() * 100),
        "predicted_positive_percentage": float(prediction.mean() * 100),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, prediction))
            if len(np.unique(target)) == 2
            else None
        ),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    metrics["log_loss"] = float(log_loss(target, clipped, labels=[0, 1]))
    metrics["brier_score"] = float(brier_score_loss(target, clipped))
    if len(np.unique(target)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(target, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(target, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "sample_count": int(len(target)),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
    }


def _data_summary(data: pd.DataFrame) -> dict[str, Any]:
    delayed = data["is_delayed_15"].to_numpy(dtype=np.int8)
    return {
        "row_count": int(len(data)),
        "delayed_count": int(delayed.sum()),
        "delayed_percentage": float(delayed.mean() * 100),
        "month_min": int(data["month"].min()),
        "month_max": int(data["month"].max()),
    }


def _classification_baselines(target: np.ndarray) -> dict[str, Any]:
    always_negative = np.zeros_like(target)
    always_positive = np.ones_like(target)
    return {
        "always_negative": {
            "accuracy": float(accuracy_score(target, always_negative)),
            "f1": float(f1_score(target, always_negative, zero_division=0)),
        },
        "always_positive": {
            "accuracy": float(accuracy_score(target, always_positive)),
            "f1": float(f1_score(target, always_positive, zero_division=0)),
        },
    }


def train_models(data: pd.DataFrame, seed: int = 42) -> tuple[dict, dict, pd.DataFrame]:
    """Entraîne les trois familles de modèles et calcule leurs métriques."""

    split = split_temporally(data)
    train_features = _model_features(split.train)
    validation_features = _model_features(split.validation)
    test_features = _model_features(split.test)
    train_classification_features = _model_features(
        split.train, DELAY_CLASSIFICATION_FEATURES
    )
    validation_classification_features = _model_features(
        split.validation, DELAY_CLASSIFICATION_FEATURES
    )
    test_classification_features = _model_features(
        split.test, DELAY_CLASSIFICATION_FEATURES
    )

    train_delay_target = split.train["is_delayed_15"].to_numpy(dtype=np.int8)
    validation_delay_target = split.validation["is_delayed_15"].to_numpy(
        dtype=np.int8
    )
    test_delay_target = split.test["is_delayed_15"].to_numpy(dtype=np.int8)

    delay_classifier = _fit_binary_classifier(
        train_classification_features,
        train_delay_target,
        validation_classification_features,
        validation_delay_target,
        seed,
    )
    validation_delay_probability = _positive_probability(
        delay_classifier, validation_classification_features
    )
    delay_threshold = _best_f1_threshold(
        validation_delay_target, validation_delay_probability
    )
    test_delay_probability = _positive_probability(
        delay_classifier, test_classification_features
    )

    delayed_train_mask = train_delay_target == 1
    delayed_validation_mask = validation_delay_target == 1
    delayed_test_mask = test_delay_target == 1
    train_delay_minutes = split.train.loc[
        delayed_train_mask, "delay_minutes"
    ].to_numpy(dtype=float)
    validation_delay_minutes = split.validation.loc[
        delayed_validation_mask, "delay_minutes"
    ].to_numpy(dtype=float)
    test_delay_minutes = split.test.loc[
        delayed_test_mask, "delay_minutes"
    ].to_numpy(dtype=float)

    delay_regressor = CatBoostRegressor(
        iterations=600,
        depth=8,
        learning_rate=0.08,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=8,
        random_seed=seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=80,
    )
    delay_regressor.fit(
        train_features.loc[delayed_train_mask],
        train_delay_minutes,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(
            validation_features.loc[delayed_validation_mask],
            validation_delay_minutes,
        ),
        use_best_model=True,
    )
    predicted_delay_minutes = np.clip(
        delay_regressor.predict(test_features.loc[delayed_test_mask]),
        15.0,
        float(MAX_PREDICTED_DELAY_MINUTES),
    )
    train_delay_median = float(np.median(train_delay_minutes))
    baseline_delay_prediction = np.full_like(test_delay_minutes, train_delay_median)

    reason_classifiers = {}
    reason_thresholds = {}
    reason_metrics = {}
    for offset, (reason, target_column) in enumerate(REASON_TARGETS.items(), start=1):
        train_target = split.train.loc[
            delayed_train_mask, target_column
        ].to_numpy(dtype=np.int8)
        validation_target = split.validation.loc[
            delayed_validation_mask, target_column
        ].to_numpy(dtype=np.int8)
        test_target = split.test.loc[
            delayed_test_mask, target_column
        ].to_numpy(dtype=np.int8)
        model = _fit_binary_classifier(
            train_features.loc[delayed_train_mask],
            train_target,
            validation_features.loc[delayed_validation_mask],
            validation_target,
            seed + offset,
        )
        validation_probability = _positive_probability(
            model, validation_features.loc[delayed_validation_mask]
        )
        threshold = _best_f1_threshold(validation_target, validation_probability)
        test_probability = _positive_probability(
            model, test_features.loc[delayed_test_mask]
        )
        reason_classifiers[reason] = model
        reason_thresholds[reason] = threshold
        reason_metrics[reason] = {
            "validation": _classification_metrics(
                validation_target, validation_probability, threshold
            ),
            "test": _classification_metrics(test_target, test_probability, threshold),
        }

    if isinstance(delay_classifier, CatBoostClassifier):
        importances = delay_classifier.get_feature_importance()
    else:
        importances = np.zeros(len(DELAY_CLASSIFICATION_FEATURES), dtype=float)
    feature_importance = pd.DataFrame(
        {"feature": DELAY_CLASSIFICATION_FEATURES, "importance": importances}
    ).sort_values("importance", ascending=False, ignore_index=True)

    metrics = {
        "data": {
            "all": _data_summary(data),
            "train": _data_summary(split.train),
            "validation": _data_summary(split.validation),
            "test": _data_summary(split.test),
        },
        "delay_classification": {
            "validation": _classification_metrics(
                validation_delay_target,
                validation_delay_probability,
                delay_threshold,
            ),
            "test": _classification_metrics(
                test_delay_target, test_delay_probability, delay_threshold
            ),
            "test_baselines": _classification_baselines(test_delay_target),
        },
        "delay_regression": {
            "test": _regression_metrics(
                test_delay_minutes, predicted_delay_minutes
            ),
            "median_baseline_test": _regression_metrics(
                test_delay_minutes, baseline_delay_prediction
            ),
            "training_target_median": train_delay_median,
        },
        "reasons": reason_metrics,
    }

    bundle = {
        "artifact_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "delay_classifier": delay_classifier,
        "delay_threshold": delay_threshold,
        "delay_regressor": delay_regressor,
        "reason_classifiers": reason_classifiers,
        "reason_thresholds": reason_thresholds,
        "feature_columns": FEATURE_COLUMNS,
        "delay_feature_columns": DELAY_CLASSIFICATION_FEATURES,
        "regression_feature_columns": FEATURE_COLUMNS,
        "reason_feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "history_profiles": build_history_profiles(data),
        "history_cutoff_date": str(data["flight_date"].max()),
        "reason_targets": REASON_TARGETS,
        "versions": {
            "python": platform.python_version(),
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "polars": polars.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "training_summary": metrics["data"],
    }
    return bundle, metrics, feature_importance


def save_training_outputs(
    bundle: dict,
    metrics: dict,
    feature_importance: pd.DataFrame,
    model_path: str,
    metrics_path: str,
    feature_importance_path: str,
) -> None:
    """Sauvegarde l'artefact local et les résultats d'évaluation."""

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_file, compress=3)

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    importance_file = Path(feature_importance_path)
    importance_file.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(importance_file, index=False)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entraîne les modèles Python de prédiction des retards."
    )
    parser.add_argument("--input", default="data/flight_data_2024.csv")
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-output", default="models/flight_delay_models.joblib"
    )
    parser.add_argument("--metrics-output", default="models/training_metrics.json")
    parser.add_argument(
        "--importance-output", default="models/feature_importance.csv"
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    data = load_ml_data(
        arguments.input,
        sample_fraction=arguments.sample_fraction,
        seed=arguments.seed,
    )
    print(f"Données ML préparées : {len(data):,} vols achevés.")
    bundle, metrics, feature_importance = train_models(data, seed=arguments.seed)
    save_training_outputs(
        bundle,
        metrics,
        feature_importance,
        arguments.model_output,
        arguments.metrics_output,
        arguments.importance_output,
    )
    classification = metrics["delay_classification"]["test"]
    regression = metrics["delay_regression"]["test"]
    print(
        "Classification test — "
        f"F1 : {classification['f1']:.3f}, "
        f"rappel : {classification['recall']:.3f}, "
        f"ROC-AUC : {classification['roc_auc']:.3f}."
    )
    print(
        "Régression conditionnelle test — "
        f"MAE : {regression['mae']:.1f} min, "
        f"RMSE : {regression['rmse']:.1f} min."
    )
    print(f"Modèles sauvegardés dans {arguments.model_output}.")


if __name__ == "__main__":
    main()
