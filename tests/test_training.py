from __future__ import annotations

import pytest

from flight_delays.ml_data import load_ml_data
from flight_delays.prediction import predict_flight
from flight_delays.training import train_models


@pytest.fixture(scope="module")
def trained_models():
    """Entraîne un petit modèle reproductible sur le CSV versionné."""

    data = load_ml_data("data/flight_data_2024_sample.csv", sample_fraction=1.0)
    return train_models(data, seed=42)


def test_training_produces_all_models_and_metrics(trained_models):
    bundle, metrics, feature_importance = trained_models

    assert bundle["artifact_version"] == 1
    assert set(bundle["reason_classifiers"]) == {
        "carrier",
        "weather",
        "nas",
        "security",
        "late_aircraft",
    }
    assert 0 < bundle["delay_threshold"] < 1
    assert metrics["delay_classification"]["test"]["sample_count"] > 0
    assert metrics["delay_regression"]["test"]["mae"] >= 0
    assert not feature_importance.empty


def test_prediction_has_bounded_and_interpretable_outputs(trained_models):
    bundle, _, _ = trained_models
    flight = {
        "month": 7,
        "day_of_month": 14,
        "day_of_week": 7,
        "op_unique_carrier": "AA",
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
    assert 15 <= result["estimated_delay_minutes_if_delayed"] <= 1440
    assert set(result["reason_probabilities_if_delayed"]) == set(
        bundle["reason_classifiers"]
    )
    assert all(
        0 <= probability <= 1
        for probability in result["reason_probabilities_if_delayed"].values()
    )
