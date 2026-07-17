from __future__ import annotations

import pandas as pd
import pytest

from flight_delays.ml_data import (
    FEATURE_COLUMNS,
    FORBIDDEN_PRE_DEPARTURE_COLUMNS,
    REASON_TARGETS,
    load_ml_data,
    prepare_prediction_frame,
    split_temporally,
)


@pytest.fixture(scope="module")
def ml_sample_data():
    """Charge le petit CSV avec Python uniquement, sans ouvrir de session Spark."""

    return load_ml_data("data/flight_data_2024_sample.csv", sample_fraction=1.0)


def test_load_ml_data_prepares_expected_rows_and_targets(ml_sample_data):
    assert len(ml_sample_data) == 9836
    assert set(FEATURE_COLUMNS).issubset(ml_sample_data.columns)
    assert set(REASON_TARGETS.values()).issubset(ml_sample_data.columns)
    assert set(ml_sample_data["is_delayed_15"].unique()).issubset({0, 1})
    assert not ml_sample_data["delay_minutes"].isna().any()
    assert "reason_late_aircraft" not in ml_sample_data.columns


def test_features_do_not_contain_post_departure_information():
    assert set(FEATURE_COLUMNS).isdisjoint(FORBIDDEN_PRE_DEPARTURE_COLUMNS)


def test_temporal_split_respects_month_boundaries(ml_sample_data):
    split = split_temporally(ml_sample_data)

    assert split.train["month"].max() <= 8
    assert split.validation["month"].between(9, 10).all()
    assert split.test["month"].min() >= 11
    assert len(split.train) + len(split.validation) + len(split.test) == len(
        ml_sample_data
    )


def test_prepare_prediction_frame_creates_the_same_features():
    flight = {
        "flight_date": "2024-07-14",
        "month": 7,
        "day_of_month": 14,
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

    prepared = prepare_prediction_frame(flight)

    assert prepared.columns.tolist() == FEATURE_COLUMNS
    assert prepared.loc[0, "route"] == "JFK-LAX"
    assert prepared.loc[0, "is_weekend"] == 1


def test_prepare_prediction_frame_rejects_missing_fields():
    with pytest.raises(ValueError, match="Champs obligatoires absents"):
        prepare_prediction_frame({"month": 1})


def test_load_ml_data_rejects_invalid_sample_fraction():
    with pytest.raises(ValueError, match="fraction d'échantillonnage"):
        load_ml_data("data/flight_data_2024_sample.csv", sample_fraction=1.1)


def test_prepare_prediction_frame_rejects_invalid_scheduled_time():
    flight = {
        "flight_date": "2024-07-14",
        "month": 7,
        "day_of_month": 14,
        "day_of_week": 7,
        "op_unique_carrier": "AA",
        "op_carrier_fl_num": 100,
        "origin": "JFK",
        "origin_state_nm": "New York",
        "dest": "LAX",
        "dest_state_nm": "California",
        "crs_dep_time": 1260,
        "crs_arr_time": 1145,
        "crs_elapsed_time": 375,
        "distance": 2475,
    }

    with pytest.raises(ValueError, match="format HHMM"):
        prepare_prediction_frame(flight)


def test_historical_rates_use_only_previous_days(ml_sample_data):
    raw = pd.read_csv("data/flight_data_2024_sample.csv")
    raw["flight_date"] = pd.to_datetime(raw["fl_date"]).dt.date
    previous_day = raw.loc[
        raw["flight_date"] == pd.Timestamp("2024-04-17").date()
    ]
    completed = previous_day.loc[
        (previous_day["cancelled"] == 0)
        & (previous_day["diverted"] == 0)
        & previous_day["arr_delay"].notna()
    ]
    expected_rate = float((completed["arr_delay"] >= 15).mean())
    target_rows = ml_sample_data.loc[
        ml_sample_data["flight_date"] == pd.Timestamp("2024-04-18").date()
    ]

    assert target_rows["global_delay_rate_1d"].nunique() == 1
    assert target_rows.iloc[0]["global_delay_rate_1d"] == pytest.approx(expected_rate)

    first_day = ml_sample_data.loc[
        ml_sample_data["flight_date"] == pd.Timestamp("2024-01-01").date()
    ]
    assert first_day["global_delay_rate_1d"].isna().all()
