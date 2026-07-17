"""Entraînement des modèles Python de retard et de causes."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delays.ml_data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MAX_PREDICTED_DELAY_MINUTES,
    NUMERIC_FEATURES,
    REASON_TARGETS,
    load_ml_data,
    split_temporally,
)


def build_preprocessor(minimum_category_frequency: int = 20) -> ColumnTransformer:
    """Crée le préprocesseur ajusté uniquement sur la période d'entraînement."""

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=minimum_category_frequency,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.3,
    )


def _new_classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-4,
        class_weight="balanced",
        random_state=seed,
    )


def _fit_binary_classifier(features, target: np.ndarray, seed: int):
    unique = np.unique(target)
    if len(unique) == 1:
        model = DummyClassifier(strategy="constant", constant=int(unique[0]))
    else:
        model = _new_classifier(seed)
    return model.fit(features, target)


def _calibrate_classifier(model, features, target: np.ndarray):
    """Calibre un modèle déjà ajusté sur l'ensemble de validation."""

    if len(np.unique(target)) < 2 or isinstance(model, DummyClassifier):
        return model
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(model),
        method="sigmoid",
    )
    return calibrated_model.fit(features, target)


def _positive_probability(model, features) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(features.shape[0], dtype=float)
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


def delay_minutes_from_log_predictions(predictions: np.ndarray) -> np.ndarray:
    """Convertit les sorties logarithmiques en estimations bornées et réalistes."""

    minimum = np.log1p(15.0)
    maximum = np.log1p(float(MAX_PREDICTED_DELAY_MINUTES))
    return np.expm1(np.clip(predictions, minimum, maximum))


def _data_summary(data: pd.DataFrame) -> dict[str, Any]:
    delayed = data["is_delayed_15"].to_numpy(dtype=np.int8)
    return {
        "row_count": int(len(data)),
        "delayed_count": int(delayed.sum()),
        "delayed_percentage": float(delayed.mean() * 100),
        "month_min": int(data["month"].min()),
        "month_max": int(data["month"].max()),
    }


def train_models(data: pd.DataFrame, seed: int = 42) -> tuple[dict, dict, pd.DataFrame]:
    """Entraîne les trois familles de modèles et calcule leurs métriques."""

    split = split_temporally(data)
    preprocessor = build_preprocessor()

    train_features = preprocessor.fit_transform(split.train[FEATURE_COLUMNS])
    validation_features = preprocessor.transform(split.validation[FEATURE_COLUMNS])
    test_features = preprocessor.transform(split.test[FEATURE_COLUMNS])

    train_delay_target = split.train["is_delayed_15"].to_numpy(dtype=np.int8)
    validation_delay_target = split.validation["is_delayed_15"].to_numpy(
        dtype=np.int8
    )
    test_delay_target = split.test["is_delayed_15"].to_numpy(dtype=np.int8)

    raw_delay_classifier = _fit_binary_classifier(
        train_features, train_delay_target, seed
    )
    delay_classifier = _calibrate_classifier(
        raw_delay_classifier,
        validation_features,
        validation_delay_target,
    )
    validation_delay_probability = _positive_probability(
        delay_classifier, validation_features
    )
    delay_threshold = _best_f1_threshold(
        validation_delay_target, validation_delay_probability
    )
    test_delay_probability = _positive_probability(delay_classifier, test_features)

    delayed_train_mask = train_delay_target == 1
    delayed_validation_mask = validation_delay_target == 1
    delayed_test_mask = test_delay_target == 1

    delay_regressor = SGDRegressor(
        loss="huber",
        epsilon=0.5,
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
    )
    train_delay_minutes = split.train.loc[
        delayed_train_mask, "delay_minutes"
    ].to_numpy(dtype=float)
    delay_regressor.fit(
        train_features[delayed_train_mask], np.log1p(train_delay_minutes)
    )

    test_delay_minutes = split.test.loc[
        delayed_test_mask, "delay_minutes"
    ].to_numpy(dtype=float)
    predicted_delay_minutes = delay_minutes_from_log_predictions(
        delay_regressor.predict(test_features[delayed_test_mask])
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
        raw_model = _fit_binary_classifier(
            train_features[delayed_train_mask], train_target, seed + offset
        )
        model = _calibrate_classifier(
            raw_model,
            validation_features[delayed_validation_mask],
            validation_target,
        )
        validation_probability = _positive_probability(
            model, validation_features[delayed_validation_mask]
        )
        threshold = _best_f1_threshold(validation_target, validation_probability)
        test_probability = _positive_probability(model, test_features[delayed_test_mask])
        reason_classifiers[reason] = model
        reason_thresholds[reason] = threshold
        reason_metrics[reason] = {
            "validation": _classification_metrics(
                validation_target, validation_probability, threshold
            ),
            "test": _classification_metrics(test_target, test_probability, threshold),
        }

    feature_names = preprocessor.get_feature_names_out()
    coefficients = (
        np.asarray(raw_delay_classifier.coef_).reshape(-1)
        if hasattr(raw_delay_classifier, "coef_")
        else np.zeros(len(feature_names), dtype=float)
    )
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False, ignore_index=True)

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
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessor": preprocessor,
        "delay_classifier": delay_classifier,
        "delay_threshold": delay_threshold,
        "delay_regressor": delay_regressor,
        "reason_classifiers": reason_classifiers,
        "reason_thresholds": reason_thresholds,
        "probability_calibration": "sigmoid",
        "feature_columns": FEATURE_COLUMNS,
        "reason_targets": REASON_TARGETS,
        "versions": {
            "python": platform.python_version(),
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
