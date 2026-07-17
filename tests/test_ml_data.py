from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from flight_delays.ml_data import (
    FEATURE_COLUMNS,
    FORBIDDEN_PRE_DEPARTURE_COLUMNS,
    REASON_TARGETS,
    load_ml_data,
    prepare_prediction_frame,
    split_temporally,
)
from flight_delays.schedule import (
    SCHEDULE_FEATURES,
    _arrival_day_offset_expression,
    build_schedule_profiles,
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

    assert split.train["month"].max() <= 7
    assert (split.tuning["month"] == 8).all()
    assert split.validation["month"].between(9, 10).all()
    assert split.test["month"].min() >= 11
    assert (
        len(split.train)
        + len(split.tuning)
        + len(split.validation)
        + len(split.test)
        == len(ml_sample_data)
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
    previous_day = raw.loc[raw["flight_date"] == pd.Timestamp("2024-04-17").date()]
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


def test_schedule_load_uses_the_complete_planning(ml_sample_data):
    raw = pd.read_csv("data/flight_data_2024_sample.csv")
    raw["departure_hour_index"] = (
        (
            (raw["crs_dep_time"].astype(int) // 100) * 60
            + raw["crs_dep_time"].astype(int) % 100
        )
        % 1440
    ) // 60
    raw["arrival_hour_index"] = (
        (
            (raw["crs_arr_time"].astype(int) // 100) * 60
            + raw["crs_arr_time"].astype(int) % 100
        )
        % 1440
    ) // 60

    cible = ml_sample_data.loc[
        (ml_sample_data["flight_date"] == pd.Timestamp("2024-09-19").date())
        & (ml_sample_data["op_unique_carrier"] == "MQ")
        & (ml_sample_data["flight_number"] == "3576")
    ].iloc[0]
    jour = raw.loc[raw["fl_date"] == "2024-09-19"]
    heures_voisines_depart = {
        (int(cible["departure_hour"]) + decalage) % 24 for decalage in (-1, 0, 1)
    }
    heures_voisines_arrivee = {
        (int(cible["arrival_hour"]) + decalage) % 24 for decalage in (-1, 0, 1)
    }

    assert cible["origin_scheduled_departures_hour"] == len(
        jour.loc[
            (jour["origin"] == cible["origin"])
            & (jour["departure_hour_index"] == int(cible["departure_hour"]))
        ]
    )
    assert cible["origin_scheduled_departures_3h"] == len(
        jour.loc[
            (jour["origin"] == cible["origin"])
            & jour["departure_hour_index"].isin(heures_voisines_depart)
        ]
    )
    assert cible["dest_scheduled_arrivals_hour"] == len(
        jour.loc[
            (jour["dest"] == cible["dest"])
            & (jour["arrival_hour_index"] == int(cible["arrival_hour"]))
        ]
    )
    assert cible["dest_scheduled_arrivals_3h"] == len(
        jour.loc[
            (jour["dest"] == cible["dest"])
            & jour["arrival_hour_index"].isin(heures_voisines_arrivee)
        ]
    )
    assert cible["route_scheduled_flights_day"] == len(
        jour.loc[(jour["origin"] == cible["origin"]) & (jour["dest"] == cible["dest"])]
    )
    assert cible["carrier_origin_scheduled_flights_day"] == len(
        jour.loc[
            (jour["op_unique_carrier"] == cible["op_unique_carrier"])
            & (jour["origin"] == cible["origin"])
        ]
    )


def test_schedule_profiles_provide_prediction_time_parity(ml_sample_data):
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
    profils = build_schedule_profiles(ml_sample_data)

    prepared = prepare_prediction_frame(
        flight,
        schedule_profiles=profils,
    )

    assert prepared[SCHEDULE_FEATURES].notna().all(axis=None)
    assert (prepared[SCHEDULE_FEATURES] > 0).all(axis=None)


def test_schedule_three_hour_window_crosses_midnight(ml_sample_data):
    raw = pd.read_csv("data/flight_data_2024_sample.csv")
    heures = raw["crs_dep_time"].astype(int)
    minutes = ((heures // 100) * 60 + heures % 100) % 1440
    raw["departure_scheduled_hour"] = pd.to_datetime(raw["fl_date"]) + pd.to_timedelta(
        minutes // 60, unit="h"
    )
    cible = ml_sample_data.loc[
        (ml_sample_data["flight_date"] == pd.Timestamp("2024-05-10").date())
        & (ml_sample_data["op_unique_carrier"] == "DL")
        & (ml_sample_data["flight_number"] == "868")
    ].iloc[0]
    heure_cible = pd.Timestamp("2024-05-10 00:00:00")
    voisinage = raw.loc[
        (raw["origin"] == cible["origin"])
        & raw["departure_scheduled_hour"].between(
            heure_cible - pd.Timedelta(hours=1),
            heure_cible + pd.Timedelta(hours=1),
        )
    ]

    assert cible["origin_scheduled_departures_hour"] == 1
    assert cible["origin_scheduled_departures_3h"] == len(voisinage) == 2
    assert voisinage["departure_scheduled_hour"].min() == pd.Timestamp(
        "2024-05-09 23:00:00"
    )


def test_arrival_day_uses_planned_duration_instead_of_hour_order():
    cases = pl.DataFrame(
        {
            "departure": [600, 1320],
            "arrival": [590, 360],
            "elapsed": [60.0, 330.0],
        }
    )

    offsets = cases.select(
        _arrival_day_offset_expression(
            pl.col("departure"),
            pl.col("arrival"),
            pl.col("elapsed"),
        ).alias("offset")
    )["offset"].to_list()

    assert offsets == [0, 1]
