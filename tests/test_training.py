from __future__ import annotations

import numpy as np
import pytest

from flight_delays.ml_data import load_ml_data
from flight_delays.prediction import predict_flight
from flight_delays.training import (
    _business_gate,
    _classification_metrics,
    _select_business_threshold,
    train_models,
)


@pytest.fixture(scope="module")
def trained_models():
    """Entraîne un petit modèle reproductible sur le CSV versionné."""

    data = load_ml_data("data/flight_data_2024_sample.csv", sample_fraction=1.0)
    return train_models(data, seed=42)


def test_training_produces_all_models_and_metrics(trained_models):
    bundle, metrics, feature_importance = trained_models

    assert bundle["artifact_version"] == 6
    assert bundle["history_cutoff_date"] == "2024-12-31"
    assert bundle["target_definition"]["operator"] == ">"
    assert bundle["target_definition"]["threshold_minutes"] == 0
    assert "delay_regressor" not in bundle
    assert "reason_classifiers" not in bundle
    assert 0 < bundle["delay_threshold"] < 1
    assert metrics["delay_classification"]["test"]["sample_count"] > 0
    threshold_selection = metrics["delay_classification"]["threshold_selection"]
    business_gate = metrics["delay_classification"]["business_gate"]
    assert bundle["delay_threshold"] == threshold_selection["threshold"]
    assert bundle["delay_threshold_strategy"] == threshold_selection["strategy"]
    assert bundle["business_gate"] == business_gate
    assert bundle["business_readiness"]["ready"] == business_gate["passed"]
    assert bundle["schedule_profiles"]["defaults"]
    assert set(metrics["delay_classification"]["test_by_month"]) == {"11", "12"}
    assert business_gate["threshold_selected_on"] == "validation"
    assert business_gate["test_used_for_threshold_selection"] is False
    assert isinstance(business_gate["passed"], bool)
    assert "delay_regression" not in metrics
    assert "reasons" not in metrics
    assert metrics["artifact_version"] == 6
    assert metrics["target_definition"] == bundle["target_definition"]
    assert not feature_importance.empty


def test_business_threshold_satisfies_all_constraints_when_possible():
    target = np.array([1] * 2_000 + [0] * 8_000, dtype=np.int8)
    probabilities = np.array(
        [0.9] * 800
        + [0.2] * 1_200
        + [0.85] * 200
        + [0.1] * 7_800,
        dtype=float,
    )

    selection = _select_business_threshold(target, probabilities)
    metrics = _classification_metrics(target, probabilities, selection["threshold"])
    gate = _business_gate(metrics)

    assert selection["strategy"] == "business_constraints"
    assert selection["feasible_on_validation"] is True
    assert gate["passed"] is True
    assert metrics["precision"] >= 0.50
    assert metrics["recall"] >= 0.20
    assert 0.05 <= metrics["alert_coverage"] <= 0.20
    assert metrics["alert_count"] >= 500
    assert metrics["precision_confidence_interval_95"]["lower"] >= 0.50
    assert metrics["recall_confidence_interval_95"]["lower"] >= 0.20


def test_business_threshold_uses_honest_conservative_fallback():
    target = np.array([1] * 2_000 + [0] * 8_000, dtype=np.int8)
    probabilities = np.array(
        [0.1] * 2_000 + list(np.linspace(0.2, 0.99, 8_000)),
        dtype=float,
    )

    selection = _select_business_threshold(target, probabilities)
    metrics = _classification_metrics(target, probabilities, selection["threshold"])
    gate = _business_gate(metrics)

    assert selection["strategy"] == "fallback_target_alert_coverage"
    assert selection["feasible_on_validation"] is False
    assert selection["fallback_reason"]
    assert gate["passed"] is False
    assert 0.05 <= metrics["alert_coverage"] <= 0.20


def test_business_gate_rejects_an_unreliable_small_support():
    target = np.array([1] * 20 + [0] * 80, dtype=np.int8)
    probabilities = np.array(
        [0.9] * 4 + [0.1] * 16 + [0.8] * 4 + [0.1] * 76,
        dtype=float,
    )

    metrics = _classification_metrics(target, probabilities, threshold=0.8)
    gate = _business_gate(metrics)

    assert metrics["precision"] == 0.50
    assert metrics["recall"] == 0.20
    assert metrics["alert_coverage"] == 0.08
    assert gate["checks"]["minimum_alert_count"] is False
    assert gate["checks"]["minimum_precision_lower_bound_95"] is False
    assert gate["passed"] is False


def test_prediction_has_bounded_and_interpretable_outputs(trained_models):
    bundle, _, _ = trained_models
    flight = {
        "flight_date": "2025-01-05",
        "month": 1,
        "day_of_month": 5,
        "day_of_week": 7,
        "op_unique_carrier": "AA",
        "op_carrier_fl_num": 100,
        "origin": "JFK",
        "origin_state_nm": "New York",
        "dest": "LAX",
        "dest_state_nm": "California",
        "crs_dep_time": 830,
        "crs_arr_time": 1145,
        "crs_elapsed_time": 375,
        "distance": 2475,
    }

    result = predict_flight(bundle, flight)

    assert 0 <= result["delay_probability"] <= 1
    assert result["artifact_version"] == 6
    assert result["model_business_ready"] == bundle["business_readiness"]["ready"]
    assert result["schedule_context_source"] == "typical_schedule_profile"
    assert result["prediction_status"] in {
        "publishable",
        "experimental_model_not_ready",
        "experimental_missing_daily_schedule",
    }
    assert result["prediction_publishable"] is False
    if result["prediction_publishable"]:
        assert isinstance(result["published_delay_alert"], bool)
    else:
        assert result["published_delay_alert"] is None
    assert isinstance(result["diagnostic_is_delayed_prediction"], bool)
    assert "is_delayed_prediction" not in result
    assert "diagnostic_estimated_delay_minutes_if_delayed" not in result
    assert "diagnostic_reason_probabilities_if_delayed" not in result

    ready_bundle = {
        **bundle,
        "business_readiness": {
            "ready": True,
            "status": "ready",
            "violations": [],
        },
    }
    flight_with_exact_schedule = {
        **flight,
        **bundle["schedule_profiles"]["defaults"],
    }
    publishable_result = predict_flight(
        ready_bundle,
        flight_with_exact_schedule,
    )

    assert publishable_result["schedule_context_source"] == "provided_daily_schedule"
    assert publishable_result["prediction_publishable"] is True
    assert isinstance(publishable_result["published_delay_alert"], bool)
    assert publishable_result["publication_blockers"] == []
