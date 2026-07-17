"""Chargement Python et feature engineering pour le machine learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import polars as pl


DELAY_THRESHOLD_MINUTES = 15
MAX_PREDICTED_DELAY_MINUTES = 1440

NUMERIC_FEATURES = [
    "day_of_month",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "departure_time_sin",
    "departure_time_cos",
    "arrival_time_sin",
    "arrival_time_cos",
    "crs_elapsed_time",
    "distance",
]

CATEGORICAL_FEATURES = [
    "op_unique_carrier",
    "origin",
    "origin_state_nm",
    "dest",
    "dest_state_nm",
    "route",
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
    "month",
    "day_of_month",
    "day_of_week",
    "op_unique_carrier",
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

    if not 0 < sample_fraction <= 1:
        raise ValueError("La fraction d'échantillonnage doit être dans ]0, 1].")
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
    """Parcourt le CSV complet en streaming et retourne un échantillon exploitable."""

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
            pl.concat_str(["origin", "dest"], separator="-").alias("route"),
            (pl.col("day_of_week") >= 6).cast(pl.Int8).alias("is_weekend"),
            (two_pi * (pl.col("month") - 1) / 12).sin().alias("month_sin"),
            (two_pi * (pl.col("month") - 1) / 12).cos().alias("month_cos"),
            (two_pi * (pl.col("day_of_week") - 1) / 7)
            .sin()
            .alias("day_of_week_sin"),
            (two_pi * (pl.col("day_of_week") - 1) / 7)
            .cos()
            .alias("day_of_week_cos"),
            (two_pi * departure_minutes / 1440).sin().alias("departure_time_sin"),
            (two_pi * departure_minutes / 1440).cos().alias("departure_time_cos"),
            (two_pi * arrival_minutes / 1440).sin().alias("arrival_time_sin"),
            (two_pi * arrival_minutes / 1440).cos().alias("arrival_time_cos"),
            (pl.col("arr_delay") >= DELAY_THRESHOLD_MINUTES)
            .cast(pl.Int8)
            .alias("is_delayed_15"),
            pl.col("arr_delay").cast(pl.Float32).alias("delay_minutes"),
            (pl.col("carrier_delay") > 0).cast(pl.Int8).alias("reason_carrier"),
            (pl.col("weather_delay") > 0).cast(pl.Int8).alias("reason_weather"),
            (pl.col("nas_delay") > 0).cast(pl.Int8).alias("reason_nas"),
            (pl.col("security_delay") > 0).cast(pl.Int8).alias("reason_security"),
            (pl.col("late_aircraft_delay") > 0)
            .cast(pl.Int8)
            .alias("reason_late_aircraft"),
        )
        .select("month", *FEATURE_COLUMNS, *TARGET_COLUMNS)
        .collect(engine="streaming")
    )
    return pd.DataFrame(prepared.to_dict(as_series=False))


def split_temporally(data: pd.DataFrame) -> TemporalSplit:
    """Sépare janvier-août, septembre-octobre et novembre-décembre."""

    train = data.loc[data["month"] <= 8].copy()
    validation = data.loc[data["month"].between(9, 10)].copy()
    test = data.loc[data["month"] >= 11].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Le découpage chronologique a produit un ensemble vide.")
    return TemporalSplit(train=train, validation=validation, test=test)


def prepare_prediction_frame(flight: dict[str, object]) -> pd.DataFrame:
    """Applique aux données d'un futur vol les mêmes transformations temporelles."""

    required = {
        "month",
        "day_of_month",
        "day_of_week",
        "op_unique_carrier",
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

    month = int(flight["month"])
    day_of_month = int(flight["day_of_month"])
    day_of_week = int(flight["day_of_week"])
    departure = int(flight["crs_dep_time"])
    arrival = int(flight["crs_arr_time"])
    if not 1 <= month <= 12:
        raise ValueError("Le mois doit être compris entre 1 et 12.")
    if not 1 <= day_of_month <= 31:
        raise ValueError("Le jour du mois doit être compris entre 1 et 31.")
    if not 1 <= day_of_week <= 7:
        raise ValueError("Le jour de la semaine doit être compris entre 1 et 7.")
    for label, value in (("départ", departure), ("arrivée", arrival)):
        if value != 2400 and not (0 <= value <= 2359 and value % 100 <= 59):
            raise ValueError(f"L'heure prévue de {label} doit respecter le format HHMM.")
    if float(flight["crs_elapsed_time"]) <= 0:
        raise ValueError("La durée prévue doit être strictement positive.")
    if float(flight["distance"]) < 0:
        raise ValueError("La distance ne peut pas être négative.")
    departure_minutes = (((departure // 100) * 60) + departure % 100) % 1440
    arrival_minutes = (((arrival // 100) * 60) + arrival % 100) % 1440
    two_pi = 2.0 * math.pi

    row = {
        "day_of_month": day_of_month,
        "is_weekend": int(day_of_week >= 6),
        "month_sin": math.sin(two_pi * (month - 1) / 12),
        "month_cos": math.cos(two_pi * (month - 1) / 12),
        "day_of_week_sin": math.sin(two_pi * (day_of_week - 1) / 7),
        "day_of_week_cos": math.cos(two_pi * (day_of_week - 1) / 7),
        "departure_time_sin": math.sin(two_pi * departure_minutes / 1440),
        "departure_time_cos": math.cos(two_pi * departure_minutes / 1440),
        "arrival_time_sin": math.sin(two_pi * arrival_minutes / 1440),
        "arrival_time_cos": math.cos(two_pi * arrival_minutes / 1440),
        "crs_elapsed_time": float(flight["crs_elapsed_time"]),
        "distance": float(flight["distance"]),
        "op_unique_carrier": str(flight["op_unique_carrier"]),
        "origin": str(flight["origin"]),
        "origin_state_nm": str(flight["origin_state_nm"]),
        "dest": str(flight["dest"]),
        "dest_state_nm": str(flight["dest_state_nm"]),
        "route": f"{flight['origin']}-{flight['dest']}",
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
