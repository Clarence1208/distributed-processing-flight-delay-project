from __future__ import annotations

from datetime import date, time

import pytest

from flight_delays.dashboard import (
    build_flight_payload,
    filter_flights,
    find_model_path,
    load_dashboard_metrics,
    load_feature_importance,
    load_flight_sample,
    overview_statistics,
    prediction_presentation,
)


@pytest.fixture(scope="module")
def dashboard_sample():
    return load_flight_sample("data/flight_data_2024_sample.csv")


def test_dashboard_sample_uses_any_positive_delay_without_causes(dashboard_sample):
    statistics = overview_statistics(dashboard_sample)

    assert statistics["flight_count"] == 10_000
    assert statistics["completed_count"] == 9_836
    assert statistics["delayed_count"] == 3_578
    assert "is_delayed" in dashboard_sample.columns
    assert {
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    }.isdisjoint(dashboard_sample.columns)


def test_dashboard_filters_are_optional_and_composable(dashboard_sample):
    unfiltered = filter_flights(dashboard_sample)
    filtered = filter_flights(
        dashboard_sample,
        months=[1],
        carriers=["AA"],
        origins=["CLT"],
    )

    assert len(unfiltered) == len(dashboard_sample)
    assert not filtered.empty
    assert set(filtered["month"]) == {1}
    assert set(filtered["op_unique_carrier"]) == {"AA"}
    assert set(filtered["origin"]) == {"CLT"}


def test_build_flight_payload_derives_date_and_hhmm_fields():
    payload = build_flight_payload(
        flight_date=date(2025, 1, 5),
        carrier="aa",
        flight_number=100,
        origin="jfk",
        origin_state="New York",
        destination="lax",
        destination_state="California",
        scheduled_departure=time(8, 5),
        scheduled_arrival=time(0, 0),
        scheduled_duration=375,
        distance=2475,
    )

    assert payload["flight_date"] == "2025-01-05"
    assert payload["month"] == 1
    assert payload["day_of_month"] == 5
    assert payload["day_of_week"] == 7
    assert payload["crs_dep_time"] == 805
    assert payload["crs_arr_time"] == 0
    assert payload["op_unique_carrier"] == "AA"


def test_build_flight_payload_rejects_identical_airports():
    with pytest.raises(ValueError, match="doivent être différents"):
        build_flight_payload(
            flight_date=date(2025, 1, 5),
            carrier="AA",
            flight_number=100,
            origin="JFK",
            origin_state="New York",
            destination="JFK",
            destination_state="New York",
            scheduled_departure=time(8, 5),
            scheduled_arrival=time(10, 0),
            scheduled_duration=115,
            distance=0,
        )


def test_metrics_and_model_fallbacks_work_on_a_fresh_clone(tmp_path):
    metrics, source = load_dashboard_metrics(tmp_path)

    assert metrics["delay_classification"]["test"]["precision"] > 0.50
    assert source == "instantané officiel v6"
    assert find_model_path(tmp_path) is None


def test_dashboard_ignores_incompatible_v5_metrics_and_importance(tmp_path):
    official = tmp_path / "models" / "official"
    official.mkdir(parents=True)
    (official / "training_metrics.json").write_text(
        '{"artifact_version": 5}', encoding="utf-8"
    )
    (official / "feature_importance.csv").write_text(
        "feature,importance\ncarrier_average_delay_minutes_1d,1.0\n",
        encoding="utf-8",
    )

    metrics, metrics_source = load_dashboard_metrics(tmp_path)
    importance, importance_source = load_feature_importance(tmp_path)

    assert metrics["artifact_version"] == 6
    assert metrics_source == "instantané officiel v6"
    assert "average_delay_minutes" not in " ".join(importance["feature"])
    assert importance_source == "instantané officiel v6"


def test_non_publishable_prediction_never_becomes_a_public_decision():
    presentation = prediction_presentation(
        {
            "prediction_publishable": False,
            "published_delay_alert": True,
            "delay_probability": 0.72,
            "classification_threshold": 0.45,
            "diagnostic_is_delayed_prediction": True,
            "publication_blockers": ["Modèle expérimental."],
            "historical_context_date": "2024-12-31",
            "schedule_context_source": "typical_schedule_profile",
        }
    )

    assert presentation["publishable"] is False
    assert presentation["published_alert"] is None
    assert presentation["diagnostic_class"] is True
    assert presentation["blockers"] == ["Modèle expérimental."]
