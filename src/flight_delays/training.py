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
from flight_delays.schedule import build_schedule_profiles


DELAY_CLASSIFICATION_FEATURES = [
    name for name in FEATURE_COLUMNS if "_average_delay_minutes_" not in name
]

BUSINESS_MIN_PRECISION = 0.50
BUSINESS_MIN_RECALL = 0.20
BUSINESS_MIN_ALERT_COVERAGE = 0.05
BUSINESS_MAX_ALERT_COVERAGE = 0.10
BUSINESS_MIN_ALERT_COUNT = 500
CONFIDENCE_Z_95 = 1.959963984540054


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


def _new_classifier(
    seed: int,
    iterations: int = 600,
    evaluation_metric: str = "AUC",
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric=evaluation_metric,
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
    evaluation_metric: str = "AUC",
):
    unique = np.unique(train_target)
    if len(unique) == 1:
        return DummyClassifier(
            strategy="constant", constant=int(unique[0])
        ).fit(train_features, train_target)

    model = _new_classifier(seed, evaluation_metric=evaluation_metric)
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
    best_iterations = max(1, int(model.tree_count_))
    combined_features = pd.concat(
        [train_features, validation_features],
        ignore_index=True,
    )
    combined_target = np.concatenate([train_target, validation_target])
    final_model = _new_classifier(
        seed,
        iterations=best_iterations,
        evaluation_metric=evaluation_metric,
    )
    final_model.fit(
        combined_features,
        combined_target,
        cat_features=[
            name for name in CATEGORICAL_FEATURES if name in combined_features.columns
        ],
    )
    return final_model


def _new_regressor(seed: int, iterations: int = 600) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
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


def _business_requirements() -> dict[str, float | int]:
    """Retourne une copie des contraintes d'acceptation métier."""

    return {
        "minimum_precision": BUSINESS_MIN_PRECISION,
        "minimum_recall": BUSINESS_MIN_RECALL,
        "minimum_alert_coverage": BUSINESS_MIN_ALERT_COVERAGE,
        "maximum_alert_coverage": BUSINESS_MAX_ALERT_COVERAGE,
        "minimum_alert_count": BUSINESS_MIN_ALERT_COUNT,
        "confidence_level": 0.95,
    }


def _wilson_lower_bound(
    successes: np.ndarray | float | int,
    totals: np.ndarray | float | int,
) -> np.ndarray:
    """Calcule la borne basse de Wilson à 95 % pour une proportion."""

    successes_array = np.asarray(successes, dtype=float)
    totals_array = np.asarray(totals, dtype=float)
    proportion = np.divide(
        successes_array,
        totals_array,
        out=np.zeros_like(successes_array, dtype=float),
        where=totals_array > 0,
    )
    denominator = 1 + (CONFIDENCE_Z_95**2 / totals_array)
    center = proportion + (CONFIDENCE_Z_95**2 / (2 * totals_array))
    margin = CONFIDENCE_Z_95 * np.sqrt(
        proportion * (1 - proportion) / totals_array
        + CONFIDENCE_Z_95**2 / (4 * totals_array**2)
    )
    lower_bound = np.divide(
        center - margin,
        denominator,
        out=np.zeros_like(proportion, dtype=float),
        where=totals_array > 0,
    )
    return np.clip(lower_bound, 0.0, 1.0)


def _wilson_interval(successes: int, total: int) -> dict[str, float]:
    """Retourne l'intervalle de Wilson à 95 % pour les rapports JSON."""

    if total <= 0:
        return {"lower": 0.0, "upper": 1.0}
    proportion = successes / total
    denominator = 1 + (CONFIDENCE_Z_95**2 / total)
    center = proportion + CONFIDENCE_Z_95**2 / (2 * total)
    margin = CONFIDENCE_Z_95 * np.sqrt(
        proportion * (1 - proportion) / total
        + CONFIDENCE_Z_95**2 / (4 * total**2)
    )
    return {
        "lower": float(max(0.0, (center - margin) / denominator)),
        "upper": float(min(1.0, (center + margin) / denominator)),
    }


def _select_business_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Sélectionne le seuil sur validation, sans jamais consulter le test."""

    target = np.asarray(target, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    f1_threshold = _best_f1_threshold(target, probabilities)
    if len(target) == 0 or len(np.unique(target)) < 2:
        return {
            "threshold": 0.5,
            "strategy": "fallback_single_class",
            "feasible_on_validation": False,
            "fallback_reason": (
                "La validation ne contient pas les deux classes nécessaires."
            ),
            "f1_threshold": f1_threshold,
        }

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if thresholds.size == 0:
        return {
            "threshold": 0.5,
            "strategy": "fallback_no_threshold",
            "feasible_on_validation": False,
            "fallback_reason": "Aucun seuil candidat n'a pu être calculé.",
            "f1_threshold": f1_threshold,
        }

    candidate_precision = precision[:-1]
    candidate_recall = recall[:-1]
    sorted_probabilities = np.sort(probabilities)
    alert_counts = len(target) - np.searchsorted(
        sorted_probabilities, thresholds, side="left"
    )
    alert_coverage = alert_counts / len(target)
    true_positive_counts = np.rint(candidate_recall * int(target.sum()))
    precision_lower_bounds = _wilson_lower_bound(
        true_positive_counts, alert_counts
    )
    recall_lower_bounds = _wilson_lower_bound(
        true_positive_counts, int(target.sum())
    )
    coverage_center = (
        BUSINESS_MIN_ALERT_COVERAGE + BUSINESS_MAX_ALERT_COVERAGE
    ) / 2

    feasible = np.flatnonzero(
        (precision_lower_bounds >= BUSINESS_MIN_PRECISION)
        & (recall_lower_bounds >= BUSINESS_MIN_RECALL)
        & (alert_coverage >= BUSINESS_MIN_ALERT_COVERAGE)
        & (alert_coverage <= BUSINESS_MAX_ALERT_COVERAGE)
        & (alert_counts >= BUSINESS_MIN_ALERT_COUNT)
    )
    if feasible.size:
        selected = max(
            feasible.tolist(),
            key=lambda index: (
                candidate_recall[index],
                candidate_precision[index],
                -abs(alert_coverage[index] - coverage_center),
                thresholds[index],
            ),
        )
        strategy = "business_constraints"
        fallback_reason = None
        feasible_on_validation = True
    else:
        in_coverage_range = np.flatnonzero(
            (alert_coverage >= BUSINESS_MIN_ALERT_COVERAGE)
            & (alert_coverage <= BUSINESS_MAX_ALERT_COVERAGE)
        )
        if in_coverage_range.size:
            selected = min(
                in_coverage_range.tolist(),
                key=lambda index: (
                    abs(alert_coverage[index] - coverage_center),
                    -candidate_precision[index],
                    -candidate_recall[index],
                    -thresholds[index],
                ),
            )
            strategy = "fallback_target_alert_coverage"
            fallback_reason = (
                "Aucun seuil ne satisfait simultanément la précision, le rappel "
                "et la couverture attendus. Le seuil de repli cible le centre de la "
                "plage métier, soit 7,5 % d'alertes sur la validation."
            )
        else:
            selected = min(
                range(len(thresholds)),
                key=lambda index: (
                    abs(alert_coverage[index] - coverage_center),
                    -candidate_precision[index],
                    -candidate_recall[index],
                    -thresholds[index],
                ),
            )
            strategy = "fallback_closest_target_coverage"
            fallback_reason = (
                "La taille de la validation ne permet aucun seuil dans la plage "
                "de couverture attendue. Le seuil de repli s'en approche au mieux."
            )
        feasible_on_validation = False

    return {
        "threshold": float(np.clip(thresholds[selected], 0.001, 0.999)),
        "strategy": strategy,
        "feasible_on_validation": feasible_on_validation,
        "fallback_reason": fallback_reason,
        "f1_threshold": f1_threshold,
        "candidate_validation_precision": float(candidate_precision[selected]),
        "candidate_validation_recall": float(candidate_recall[selected]),
        "candidate_validation_alert_coverage": float(alert_coverage[selected]),
        "candidate_validation_alert_count": int(alert_counts[selected]),
        "candidate_validation_precision_lower_bound_95": float(
            precision_lower_bounds[selected]
        ),
        "candidate_validation_recall_lower_bound_95": float(
            recall_lower_bounds[selected]
        ),
    }


def _classification_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    alert_count = int(prediction.sum())
    alert_coverage = float(prediction.mean())
    metrics: dict[str, Any] = {
        "sample_count": int(len(target)),
        "positive_count": int(target.sum()),
        "positive_percentage": float(target.mean() * 100),
        "predicted_positive_count": alert_count,
        "predicted_positive_percentage": float(prediction.mean() * 100),
        "alert_count": alert_count,
        "alert_coverage": alert_coverage,
        "alert_coverage_percentage": alert_coverage * 100,
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
        "precision_confidence_interval_95": _wilson_interval(
            int(tp), int(tp + fp)
        ),
        "recall_confidence_interval_95": _wilson_interval(
            int(tp), int(tp + fn)
        ),
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


def _business_gate(classification_metrics: dict[str, Any]) -> dict[str, Any]:
    """Évalue des métriques sans modifier le seuil déjà sélectionné."""

    checks = {
        "minimum_precision_lower_bound_95": (
            classification_metrics["precision_confidence_interval_95"]["lower"]
            >= BUSINESS_MIN_PRECISION
        ),
        "minimum_recall_lower_bound_95": (
            classification_metrics["recall_confidence_interval_95"]["lower"]
            >= BUSINESS_MIN_RECALL
        ),
        "minimum_alert_coverage": (
            classification_metrics["alert_coverage"]
            >= BUSINESS_MIN_ALERT_COVERAGE
        ),
        "maximum_alert_coverage": (
            classification_metrics["alert_coverage"]
            <= BUSINESS_MAX_ALERT_COVERAGE
        ),
        "minimum_alert_count": (
            classification_metrics["alert_count"] >= BUSINESS_MIN_ALERT_COUNT
        ),
    }
    violations = []
    if not checks["minimum_precision_lower_bound_95"]:
        violations.append(
            "La borne basse à 95 % de la précision est inférieure à 50 %."
        )
    if not checks["minimum_recall_lower_bound_95"]:
        violations.append(
            "La borne basse à 95 % du rappel est inférieure à 20 %."
        )
    if not checks["minimum_alert_coverage"]:
        violations.append("Moins de 5 % des vols déclenchent une alerte.")
    if not checks["maximum_alert_coverage"]:
        violations.append("Plus de 10 % des vols déclenchent une alerte.")
    if not checks["minimum_alert_count"]:
        violations.append(
            "Moins de 500 alertes sont disponibles pour valider la fiabilité."
        )
    return {
        "passed": bool(all(checks.values())),
        "requirements": _business_requirements(),
        "checks": checks,
        "observed": {
            "precision": classification_metrics["precision"],
            "precision_confidence_interval_95": classification_metrics[
                "precision_confidence_interval_95"
            ],
            "recall": classification_metrics["recall"],
            "recall_confidence_interval_95": classification_metrics[
                "recall_confidence_interval_95"
            ],
            "alert_coverage": classification_metrics["alert_coverage"],
            "alert_count": classification_metrics["alert_count"],
            "sample_count": classification_metrics["sample_count"],
        },
        "violations": violations,
    }


def _monthly_classification_metrics(
    months: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Calcule les performances mensuelles au même seuil global."""

    monthly_metrics = {}
    monthly_gates = {}
    for month in sorted(np.unique(months)):
        mask = months == month
        metrics = _classification_metrics(
            target[mask], probabilities[mask], threshold
        )
        month_name = str(int(month))
        monthly_metrics[month_name] = metrics
        monthly_gates[month_name] = _business_gate(metrics)
    return monthly_metrics, monthly_gates


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
    tuning_features = _model_features(split.tuning)
    validation_features = _model_features(split.validation)
    test_features = _model_features(split.test)
    train_classification_features = _model_features(
        split.train, DELAY_CLASSIFICATION_FEATURES
    )
    tuning_classification_features = _model_features(
        split.tuning, DELAY_CLASSIFICATION_FEATURES
    )
    validation_classification_features = _model_features(
        split.validation, DELAY_CLASSIFICATION_FEATURES
    )
    test_classification_features = _model_features(
        split.test, DELAY_CLASSIFICATION_FEATURES
    )

    train_delay_target = split.train["is_delayed_15"].to_numpy(dtype=np.int8)
    tuning_delay_target = split.tuning["is_delayed_15"].to_numpy(dtype=np.int8)
    validation_delay_target = split.validation["is_delayed_15"].to_numpy(
        dtype=np.int8
    )
    test_delay_target = split.test["is_delayed_15"].to_numpy(dtype=np.int8)

    delay_classifier = _fit_binary_classifier(
        train_classification_features,
        train_delay_target,
        tuning_classification_features,
        tuning_delay_target,
        seed,
        evaluation_metric="PRAUC:type=Classic",
    )
    validation_delay_probability = _positive_probability(
        delay_classifier, validation_classification_features
    )
    threshold_selection = _select_business_threshold(
        validation_delay_target, validation_delay_probability
    )
    delay_threshold = threshold_selection["threshold"]
    test_delay_probability = _positive_probability(
        delay_classifier, test_classification_features
    )

    delayed_train_mask = train_delay_target == 1
    delayed_tuning_mask = tuning_delay_target == 1
    delayed_validation_mask = validation_delay_target == 1
    delayed_test_mask = test_delay_target == 1
    train_delay_minutes = split.train.loc[
        delayed_train_mask, "delay_minutes"
    ].to_numpy(dtype=float)
    tuning_delay_minutes = split.tuning.loc[
        delayed_tuning_mask, "delay_minutes"
    ].to_numpy(dtype=float)
    test_delay_minutes = split.test.loc[
        delayed_test_mask, "delay_minutes"
    ].to_numpy(dtype=float)

    delay_regressor = _new_regressor(seed)
    delay_regressor.fit(
        train_features.loc[delayed_train_mask],
        train_delay_minutes,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(
            tuning_features.loc[delayed_tuning_mask],
            tuning_delay_minutes,
        ),
        use_best_model=True,
    )
    best_regression_iterations = max(1, int(delay_regressor.tree_count_))
    combined_delayed_features = pd.concat(
        [
            train_features.loc[delayed_train_mask],
            tuning_features.loc[delayed_tuning_mask],
        ],
        ignore_index=True,
    )
    combined_delay_minutes = np.concatenate(
        [train_delay_minutes, tuning_delay_minutes]
    )
    delay_regressor = _new_regressor(seed, iterations=best_regression_iterations)
    delay_regressor.fit(
        combined_delayed_features,
        combined_delay_minutes,
        cat_features=CATEGORICAL_FEATURES,
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
        tuning_target = split.tuning.loc[
            delayed_tuning_mask, target_column
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
            tuning_features.loc[delayed_tuning_mask],
            tuning_target,
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

    validation_classification_metrics = _classification_metrics(
        validation_delay_target,
        validation_delay_probability,
        delay_threshold,
    )
    test_classification_metrics = _classification_metrics(
        test_delay_target, test_delay_probability, delay_threshold
    )
    validation_business_gate = _business_gate(validation_classification_metrics)
    test_business_gate = _business_gate(test_classification_metrics)
    test_monthly_metrics, test_monthly_gates = _monthly_classification_metrics(
        split.test["month"].to_numpy(dtype=np.int8),
        test_delay_target,
        test_delay_probability,
        delay_threshold,
    )
    monthly_stability_passed = bool(
        test_monthly_gates
        and all(gate["passed"] for gate in test_monthly_gates.values())
    )
    business_gate_violations = []
    if not threshold_selection["feasible_on_validation"]:
        business_gate_violations.append(
            "Aucun seuil ne respecte toutes les contraintes sur la validation."
        )
    if not test_business_gate["passed"]:
        business_gate_violations.append(
            "Les performances hors échantillon ne respectent pas toutes les "
            "contraintes métier."
        )
    if not monthly_stability_passed:
        business_gate_violations.append(
            "Les contraintes métier ne sont pas respectées pour chaque mois de test."
        )
    business_gate = {
        "passed": bool(
            threshold_selection["feasible_on_validation"]
            and validation_business_gate["passed"]
            and test_business_gate["passed"]
            and monthly_stability_passed
        ),
        "threshold_selected_on": "validation",
        "test_used_for_threshold_selection": False,
        "requirements": _business_requirements(),
        "validation": validation_business_gate,
        "test": test_business_gate,
        "monthly_stability_passed": monthly_stability_passed,
        "test_by_month": test_monthly_gates,
        "violations": business_gate_violations,
    }
    f1_threshold = threshold_selection["f1_threshold"]

    metrics = {
        "data": {
            "all": _data_summary(data),
            "train": _data_summary(split.train),
            "tuning": _data_summary(split.tuning),
            "validation": _data_summary(split.validation),
            "test": _data_summary(split.test),
        },
        "delay_classification": {
            "threshold_selection": threshold_selection,
            "business_gate": business_gate,
            "validation": validation_classification_metrics,
            "test": test_classification_metrics,
            "test_by_month": test_monthly_metrics,
            "f1_threshold_comparison": {
                "threshold": f1_threshold,
                "validation": _classification_metrics(
                    validation_delay_target,
                    validation_delay_probability,
                    f1_threshold,
                ),
                "test": _classification_metrics(
                    test_delay_target,
                    test_delay_probability,
                    f1_threshold,
                ),
            },
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
        "artifact_version": 5,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "delay_classifier": delay_classifier,
        "delay_threshold": delay_threshold,
        "delay_threshold_strategy": threshold_selection["strategy"],
        "business_gate": business_gate,
        "business_readiness": {
            "ready": business_gate["passed"],
            "status": "ready" if business_gate["passed"] else "not_ready",
            "violations": business_gate["violations"],
        },
        "delay_regressor": delay_regressor,
        "reason_classifiers": reason_classifiers,
        "reason_thresholds": reason_thresholds,
        "feature_columns": FEATURE_COLUMNS,
        "delay_feature_columns": DELAY_CLASSIFICATION_FEATURES,
        "regression_feature_columns": FEATURE_COLUMNS,
        "reason_feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "history_profiles": build_history_profiles(data),
        "schedule_profiles": build_schedule_profiles(data),
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
    business_gate = metrics["delay_classification"]["business_gate"]
    regression = metrics["delay_regression"]["test"]
    print(
        "Classification test — "
        f"précision : {classification['precision']:.3f}, "
        f"F1 : {classification['f1']:.3f}, "
        f"rappel : {classification['recall']:.3f}, "
        f"couverture : {classification['alert_coverage']:.1%}, "
        f"ROC-AUC : {classification['roc_auc']:.3f}."
    )
    print(
        "Validation métier — "
        + ("réussie." if business_gate["passed"] else "échouée : modèle non publiable.")
    )
    print(
        "Régression conditionnelle test — "
        f"MAE : {regression['mae']:.1f} min, "
        f"RMSE : {regression['rmse']:.1f} min."
    )
    print(f"Modèles sauvegardés dans {arguments.model_output}.")


if __name__ == "__main__":
    main()
