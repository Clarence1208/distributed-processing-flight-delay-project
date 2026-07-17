"""Features de charge construites uniquement à partir du planning des vols."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl


SCHEDULE_FEATURES = [
    "origin_scheduled_departures_hour",
    "origin_scheduled_departures_3h",
    "dest_scheduled_arrivals_hour",
    "dest_scheduled_arrivals_3h",
    "route_scheduled_flights_day",
    "carrier_origin_scheduled_flights_day",
]

SCHEDULE_PROFILE_SPECS = {
    "origin": {
        "keys": (
            "origin",
            "month_category",
            "day_of_week_category",
            "departure_hour",
        ),
        "features": (
            "origin_scheduled_departures_hour",
            "origin_scheduled_departures_3h",
        ),
    },
    "dest": {
        "keys": (
            "dest",
            "month_category",
            "day_of_week_category",
            "arrival_hour",
        ),
        "features": (
            "dest_scheduled_arrivals_hour",
            "dest_scheduled_arrivals_3h",
        ),
    },
    "route": {
        "keys": ("route", "month_category", "day_of_week_category"),
        "features": ("route_scheduled_flights_day",),
    },
    "carrier_origin": {
        "keys": (
            "op_unique_carrier",
            "origin",
            "month_category",
            "day_of_week_category",
        ),
        "features": ("carrier_origin_scheduled_flights_day",),
    },
}


def _hhmm_minutes_expression(name: str) -> pl.Expr:
    valeur = pl.col(name).cast(pl.Int32)
    return (((valeur // 100) * 60 + (valeur % 100)) % 1440).cast(pl.Int16)


def _arrival_day_offset_expression(
    departure_minutes: pl.Expr,
    arrival_minutes: pl.Expr,
    elapsed_minutes: pl.Expr,
) -> pl.Expr:
    """Choisit le jour local d'arrivée donnant un décalage horaire plausible."""

    implicit_timezone_offset = (
        arrival_minutes.cast(pl.Float64)
        - departure_minutes.cast(pl.Float64)
        - elapsed_minutes.cast(pl.Float64)
    )
    return (
        pl.when(implicit_timezone_offset < -720)
        .then(1)
        .when(implicit_timezone_offset > 720)
        .then(-1)
        .otherwise(0)
        .cast(pl.Int8)
    )


def _add_three_hour_window(
    horaire: pl.DataFrame,
    aeroport: str,
    colonne_horaire: str,
    feature_heure: str,
    feature_trois_heures: str,
) -> pl.DataFrame:
    """Additionne le créneau courant et ses deux créneaux voisins."""

    voisinage = (
        pl.concat(
            [
                horaire.select(
                    aeroport,
                    (pl.col(colonne_horaire) + pl.duration(hours=decalage)).alias(
                        "_target_hour"
                    ),
                    feature_heure,
                )
                for decalage in (-1, 0, 1)
            ]
        )
        .group_by(aeroport, "_target_hour")
        .agg(pl.col(feature_heure).sum().cast(pl.Float32).alias(feature_trois_heures))
    )
    return horaire.join(
        voisinage,
        left_on=[aeroport, colonne_horaire],
        right_on=[aeroport, "_target_hour"],
        how="left",
    )


def build_schedule_tables(path: str) -> dict[str, pl.DataFrame]:
    """Agrège tout le planning avant que l'échantillon ML ne soit sélectionné."""

    date_vol = pl.col("fl_date").str.to_date()
    minutes_depart = _hhmm_minutes_expression("crs_dep_time")
    minutes_arrivee = _hhmm_minutes_expression("crs_arr_time")
    decalage_jour_arrivee = _arrival_day_offset_expression(
        minutes_depart,
        minutes_arrivee,
        pl.col("crs_elapsed_time"),
    )
    planning = (
        pl.scan_csv(
            path,
            null_values=[""],
            infer_schema_length=10_000,
            low_memory=True,
        )
        .select(
            date_vol.alias("flight_date"),
            "op_unique_carrier",
            "origin",
            "dest",
            "crs_elapsed_time",
            (
                date_vol.cast(pl.Datetime) + pl.duration(hours=minutes_depart // 60)
            ).alias("departure_scheduled_hour"),
            (
                date_vol.cast(pl.Datetime)
                + pl.duration(days=decalage_jour_arrivee)
                + pl.duration(hours=minutes_arrivee // 60)
            ).alias("arrival_scheduled_hour"),
        )
        .filter(
            pl.col("flight_date").is_not_null()
            & pl.col("op_unique_carrier").is_not_null()
            & pl.col("origin").is_not_null()
            & pl.col("dest").is_not_null()
            & pl.col("departure_scheduled_hour").is_not_null()
            & pl.col("arrival_scheduled_hour").is_not_null()
            & pl.col("crs_elapsed_time").is_not_null()
            & (pl.col("crs_elapsed_time") > 0)
        )
    )

    origine_horaire = planning.group_by("origin", "departure_scheduled_hour").agg(
        pl.len().cast(pl.Float32).alias("origin_scheduled_departures_hour")
    )
    destination_horaire = planning.group_by("dest", "arrival_scheduled_hour").agg(
        pl.len().cast(pl.Float32).alias("dest_scheduled_arrivals_hour")
    )
    route_journaliere = planning.group_by("flight_date", "origin", "dest").agg(
        pl.len().cast(pl.Float32).alias("route_scheduled_flights_day")
    )
    transporteur_origine_journalier = planning.group_by(
        "flight_date", "op_unique_carrier", "origin"
    ).agg(pl.len().cast(pl.Float32).alias("carrier_origin_scheduled_flights_day"))

    origine_horaire_collectee = origine_horaire.collect(engine="streaming")
    destination_horaire_collectee = destination_horaire.collect(engine="streaming")
    route_journaliere_collectee = route_journaliere.collect(engine="streaming")
    transporteur_origine_journalier_collecte = (
        transporteur_origine_journalier.collect(engine="streaming")
    )

    return {
        "origin": _add_three_hour_window(
            origine_horaire_collectee,
            "origin",
            "departure_scheduled_hour",
            "origin_scheduled_departures_hour",
            "origin_scheduled_departures_3h",
        ),
        "dest": _add_three_hour_window(
            destination_horaire_collectee,
            "dest",
            "arrival_scheduled_hour",
            "dest_scheduled_arrivals_hour",
            "dest_scheduled_arrivals_3h",
        ),
        "route": route_journaliere_collectee,
        "carrier_origin": transporteur_origine_journalier_collecte,
    }


def add_scheduled_congestion_features(
    data: pl.DataFrame,
    path: str,
) -> pl.DataFrame:
    """Joint des volumes planifiés connus avant le départ du vol."""

    tables = build_schedule_tables(path)
    decalage_jour_arrivee = _arrival_day_offset_expression(
        pl.col("scheduled_departure_minutes"),
        pl.col("scheduled_arrival_minutes"),
        pl.col("crs_elapsed_time"),
    )
    enrichies = data.with_columns(
        (
            pl.col("flight_date").cast(pl.Datetime)
            + pl.duration(hours=pl.col("scheduled_departure_minutes") // 60)
        ).alias("departure_scheduled_hour"),
        (
            pl.col("flight_date").cast(pl.Datetime)
            + pl.duration(days=decalage_jour_arrivee)
            + pl.duration(hours=pl.col("scheduled_arrival_minutes") // 60)
        ).alias("arrival_scheduled_hour"),
    )
    enrichies = enrichies.join(
        tables["origin"],
        on=["origin", "departure_scheduled_hour"],
        how="left",
    )
    enrichies = enrichies.join(
        tables["dest"],
        on=["dest", "arrival_scheduled_hour"],
        how="left",
    )
    enrichies = enrichies.join(
        tables["route"],
        on=["flight_date", "origin", "dest"],
        how="left",
    )
    enrichies = enrichies.join(
        tables["carrier_origin"],
        on=["flight_date", "op_unique_carrier", "origin"],
        how="left",
    )
    return enrichies.drop(
        "departure_scheduled_hour",
        "arrival_scheduled_hour",
    )


def build_schedule_profiles(data: pd.DataFrame) -> dict[str, Any]:
    """Construit des valeurs typiques utilisables pour un futur planning."""

    profils: dict[str, Any] = {
        "defaults": {
            nom: _finite_or_default(data[nom].median(), 0.0)
            for nom in SCHEDULE_FEATURES
        }
    }
    for prefixe, specification in SCHEDULE_PROFILE_SPECS.items():
        cles = list(specification["keys"])
        features = list(specification["features"])
        observations_uniques = data.drop_duplicates(
            subset=["flight_date", *cles, *features]
        )
        agregats = (
            observations_uniques.groupby(cles, dropna=False)[features]
            .median()
            .reset_index()
        )
        profils[prefixe] = {
            _profile_key(tuple(str(ligne[cle]) for cle in cles)): {
                nom: _finite_or_default(ligne[nom], profils["defaults"][nom])
                for nom in features
            }
            for _, ligne in agregats.iterrows()
        }
    return profils


def add_schedule_profile_values(
    row: dict[str, object],
    flight: dict[str, object],
    profiles: dict[str, Any] | None,
) -> None:
    """Ajoute les charges du planning ou leur profil typique au vol futur."""

    if profiles:
        row.update(profiles["defaults"])
        for prefixe, specification in SCHEDULE_PROFILE_SPECS.items():
            cle = _profile_key(tuple(str(row[nom]) for nom in specification["keys"]))
            row.update(profiles.get(prefixe, {}).get(cle, {}))
    else:
        row.update({nom: np.nan for nom in SCHEDULE_FEATURES})

    for nom in SCHEDULE_FEATURES:
        if nom in flight and flight[nom] is not None:
            valeur = float(flight[nom])
            if not np.isfinite(valeur) or valeur < 0:
                raise ValueError(
                    f"La feature de planning {nom} doit être positive ou nulle."
                )
            row[nom] = valeur


def _profile_key(values: tuple[str, ...]) -> str:
    return "|".join(values)


def _finite_or_default(value: object, default: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else default
