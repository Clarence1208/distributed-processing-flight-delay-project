"""Chargement Python et feature engineering pour le machine learning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd
import polars as pl

from flight_delays.historical import (
    HISTORICAL_FEATURES,
    add_historical_features,
    add_profile_values,
)


DELAY_THRESHOLD_MINUTES = 15
MAX_PREDICTED_DELAY_MINUTES = 1440

BASE_NUMERIC_FEATURES = [
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "scheduled_departure_minutes",
    "scheduled_arrival_minutes",
    "departure_time_sin",
    "departure_time_cos",
    "arrival_time_sin",
    "arrival_time_cos",
    "crs_elapsed_time",
    "distance",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + HISTORICAL_FEATURES

CATEGORICAL_FEATURES = [
    "op_unique_carrier",
    "flight_number",
    "origin",
    "origin_state_nm",
    "dest",
    "dest_state_nm",
    "route",
    "carrier_route",
    "departure_hour",
    "arrival_hour",
    "origin_hour",
    "dest_hour",
    "month_category",
    "day_of_week_category",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REASON_TARGETS = {
    "carrier": "reason_carrier",
    "weather": "reason_weather",
    "nas": "reason_nas",
    "security": "reason_security",
    "late_aircraft": "reason_late_aircraft",
}

TARGET_COLUMNS = ["is_delayed_15", "delay_minutes", *REASON_TARGETS.values()]

FORBIDDEN_PRE_DEPARTURE_COLUMNS = {
    "dep_time",
    "dep_delay",
    "taxi_out",
    "wheels_off",
    "wheels_on",
    "taxi_in",
    "arr_time",
    "arr_delay",
    "actual_elapsed_time",
    "air_time",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
}

RAW_ML_COLUMNS = [
    "fl_date",
    "month",
    "day_of_month",
    "day_of_week",
    "op_unique_carrier",
    "op_carrier_fl_num",
    "origin",
    "origin_state_nm",
    "dest",
    "dest_state_nm",
    "crs_dep_time",
    "crs_arr_time",
    "cancelled",
    "diverted",
    "crs_elapsed_time",
    "distance",
    "arr_delay",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]


@dataclass(frozen=True)
class TemporalSplit:
    """Contient les trois sous-ensembles chronologiques."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _hhmm_minutes_expression(name: str) -> pl.Expr:
    value = pl.col(name).cast(pl.Int32)
    return (((value // 100) * 60 + (value % 100)) % 1440).cast(pl.Int16)


def _sample_filter(sample_fraction: float, seed: int) -> pl.Expr:
    """Construit un échantillonnage déterministe compatible avec le streaming."""

    modulus = 1_000_003
    threshold = max(1, int(sample_fraction * modulus))
    row_index = pl.col("_row_index").cast(pl.UInt64)
    sample_key = ((row_index % modulus) * 48_271 + seed) % modulus
    return sample_key < threshold


def load_ml_data(
    path: str,
    sample_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Parcourt le CSV et prépare les features sans utiliser Spark."""

    if not 0 < sample_fraction <= 1:
        raise ValueError("La fraction d'échantillonnage doit être dans ]0, 1].")

    lazy_data = (
        pl.scan_csv(
            path,
            null_values=[""],
            infer_schema_length=10_000,
            low_memory=True,
        )
        .with_row_index("_row_index")
        .select("_row_index", *RAW_ML_COLUMNS)
        .filter(
            (pl.col("cancelled") == 0)
            & (pl.col("diverted") == 0)
            & pl.col("arr_delay").is_not_null()
        )
    )
    if sample_fraction < 1:
        lazy_data = lazy_data.filter(_sample_filter(sample_fraction, seed))

    departure_minutes = _hhmm_minutes_expression("crs_dep_time")
    arrival_minutes = _hhmm_minutes_expression("crs_arr_time")
    two_pi = 2.0 * math.pi
    prepared = (
        lazy_data.with_columns(
            pl.col("fl_date").str.to_date().alias("flight_date"),
            departure_minutes.alias("scheduled_departure_minutes"),
            arrival_minutes.alias("scheduled_arrival_minutes"),
            (departure_minutes // 60).cast(pl.String).alias("departure_hour"),
            (arrival_minutes // 60).cast(pl.String).alias("arrival_hour"),
            pl.col("op_carrier_fl_num")
            .cast(pl.Int64)
            .cast(pl.String)
            .alias("flight_number"),
            pl.concat_str(["origin", "dest"], separator="-").alias("route"),
            pl.concat_str(
                ["op_unique_carrier", "origin", "dest"], separator="-"
            ).alias("carrier_route"),
            pl.concat_str(
                ["origin", (departure_minutes // 60).cast(pl.String)],
                separator="-",
            ).alias("origin_hour"),
            pl.concat_str(
                ["dest", (arrival_minutes // 60).cast(pl.String)],
                separator="-",
            ).alias("dest_hour"),
            (pl.col("day_of_week") >= 6).cast(pl.Int8).alias("is_weekend"),
            (two_pi * (pl.col("month") - 1) / 12).sin().alias("month_sin"),
            (two_pi * (pl.col("month") - 1) / 12).cos().alias("month_cos"),
            (two_pi * (pl.col("day_of_week") - 1) / 7)
            .sin()
            .alias("day_of_week_sin"),
            (two_pi * (pl.col("day_of_week") - 1) / 7)
            .cos()
            .alias("day_of_week_cos"),
            (two_pi * departure_minutes / 1440)
            .sin()
            .alias("departure_time_sin"),
            (two_pi * departure_minutes / 1440)
            .cos()
            .alias("departure_time_cos"),
            (two_pi * arrival_minutes / 1440).sin().alias("arrival_time_sin"),
            (two_pi * arrival_minutes / 1440).cos().alias("arrival_time_cos"),
            pl.col("month").cast(pl.String).alias("month_category"),
            pl.col("day_of_week")
            .cast(pl.String)
            .alias("day_of_week_category"),
            (pl.col("arr_delay") >= DELAY_THRESHOLD_MINUTES)
            .cast(pl.Int8)
            .alias("is_delayed_15"),
            pl.col("arr_delay").cast(pl.Float32).alias("delay_minutes"),
            (pl.col("carrier_delay") > 0).cast(pl.Int8).alias("reason_carrier"),
            (pl.col("weather_delay") > 0).cast(pl.Int8).alias("reason_weather"),
            (pl.col("nas_delay") > 0).cast(pl.Int8).alias("reason_nas"),
            (pl.col("security_delay") > 0)
            .cast(pl.Int8)
            .alias("reason_security"),
            (pl.col("late_aircraft_delay") > 0)
            .cast(pl.Int8)
            .alias("reason_late_aircraft"),
        )
        .with_columns(
            pl.col("flight_date").dt.ordinal_day().alias("day_of_year"),
            pl.col("flight_date").dt.week().alias("week_of_year"),
        )
        .select(
            "flight_date",
            "month",
            *BASE_NUMERIC_FEATURES,
            *CATEGORICAL_FEATURES,
            *TARGET_COLUMNS,
        )
        .collect(engine="streaming")
    )
    prepared = add_historical_features(prepared, path).select(
        "flight_date",
        "month",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    )
    frame = pd.DataFrame(prepared.to_dict(as_series=False))
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].fillna("__missing__").astype(str)
    return frame


def split_temporally(data: pd.DataFrame) -> TemporalSplit:
    """Sépare janvier-août, septembre-octobre et novembre-décembre."""

    train = data.loc[data["month"] <= 8].copy()
    validation = data.loc[data["month"].between(9, 10)].copy()
    test = data.loc[data["month"] >= 11].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Le découpage chronologique a produit un ensemble vide.")
    return TemporalSplit(train=train, validation=validation, test=test)


def prepare_prediction_frame(
    flight: dict[str, object],
    history_profiles: dict | None = None,
) -> pd.DataFrame:
    """Applique à un futur vol les mêmes transformations que pendant l'entraînement."""

    required = {
        "flight_date",
        "month",
        "day_of_month",
        "day_of_week",
        "op_unique_carrier",
        "op_carrier_fl_num",
        "origin",
        "origin_state_nm",
        "dest",
        "dest_state_nm",
        "crs_dep_time",
        "crs_arr_time",
        "crs_elapsed_time",
        "distance",
    }
    missing = sorted(required - flight.keys())
    if missing:
        raise ValueError(f"Champs obligatoires absents : {', '.join(missing)}")

    try:
        flight_date = date.fromisoformat(str(flight["flight_date"]))
    except ValueError as error:
        raise ValueError("La date du vol doit respecter le format YYYY-MM-DD.") from error
    month = int(flight["month"])
    day_of_month = int(flight["day_of_month"])
    day_of_week = int(flight["day_of_week"])
    departure = int(flight["crs_dep_time"])
    arrival = int(flight["crs_arr_time"])
    if (flight_date.month, flight_date.day) != (month, day_of_month):
        raise ValueError("La date ne correspond pas au mois et au jour fournis.")
    if flight_date.isoweekday() != day_of_week:
        raise ValueError("La date ne correspond pas au jour de la semaine fourni.")
    for label, value in (("départ", departure), ("arrivée", arrival)):
        if value != 2400 and not (0 <= value <= 2359 and value % 100 <= 59):
            raise ValueError(f"L'heure prévue de {label} doit respecter le format HHMM.")
    if float(flight["crs_elapsed_time"]) <= 0:
        raise ValueError("La durée prévue doit être strictement positive.")
    if float(flight["distance"]) < 0:
        raise ValueError("La distance ne peut pas être négative.")
    flight_number = int(float(flight["op_carrier_fl_num"]))
    if flight_number <= 0:
        raise ValueError("Le numéro de vol doit être strictement positif.")

    departure_minutes = (((departure // 100) * 60) + departure % 100) % 1440
    arrival_minutes = (((arrival // 100) * 60) + arrival % 100) % 1440
    two_pi = 2.0 * math.pi
    carrier = str(flight["op_unique_carrier"])
    origin = str(flight["origin"])
    destination = str(flight["dest"])
    departure_hour = str(departure_minutes // 60)
    arrival_hour = str(arrival_minutes // 60)

    row: dict[str, object] = {
        "day_of_month": day_of_month,
        "day_of_year": flight_date.timetuple().tm_yday,
        "week_of_year": flight_date.isocalendar().week,
        "is_weekend": int(day_of_week >= 6),
        "month_sin": math.sin(two_pi * (month - 1) / 12),
        "month_cos": math.cos(two_pi * (month - 1) / 12),
        "day_of_week_sin": math.sin(two_pi * (day_of_week - 1) / 7),
        "day_of_week_cos": math.cos(two_pi * (day_of_week - 1) / 7),
        "scheduled_departure_minutes": departure_minutes,
        "scheduled_arrival_minutes": arrival_minutes,
        "departure_time_sin": math.sin(two_pi * departure_minutes / 1440),
        "departure_time_cos": math.cos(two_pi * departure_minutes / 1440),
        "arrival_time_sin": math.sin(two_pi * arrival_minutes / 1440),
        "arrival_time_cos": math.cos(two_pi * arrival_minutes / 1440),
        "crs_elapsed_time": float(flight["crs_elapsed_time"]),
        "distance": float(flight["distance"]),
        "op_unique_carrier": carrier,
        "flight_number": str(flight_number),
        "origin": origin,
        "origin_state_nm": str(flight["origin_state_nm"]),
        "dest": destination,
        "dest_state_nm": str(flight["dest_state_nm"]),
        "route": f"{origin}-{destination}",
        "carrier_route": f"{carrier}-{origin}-{destination}",
        "departure_hour": departure_hour,
        "arrival_hour": arrival_hour,
        "origin_hour": f"{origin}-{departure_hour}",
        "dest_hour": f"{destination}-{arrival_hour}",
        "month_category": str(month),
        "day_of_week_category": str(day_of_week),
    }
    add_profile_values(row, flight, history_profiles)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
