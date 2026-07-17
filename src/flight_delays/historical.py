"""Construction de features historiques disponibles avant le jour du vol."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl


HISTORY_SPECS = {
    "global": {"keys": (), "windows": (1, 3, 7, 28)},
    "carrier": {"keys": ("op_unique_carrier",), "windows": (1, 7, 28)},
    "origin": {"keys": ("origin",), "windows": (1, 3, 14, 28)},
    "dest": {"keys": ("dest",), "windows": (1, 3, 14, 28)},
    "route": {"keys": ("origin", "dest"), "windows": (7, 28)},
}


def _feature_names(prefix: str, windows: tuple[int, ...]) -> list[str]:
    names = []
    for window in windows:
        names.extend(
            [
                f"{prefix}_delay_rate_{window}d",
                f"{prefix}_disruption_rate_{window}d",
                f"{prefix}_flight_count_{window}d",
            ]
        )
    return names


HISTORICAL_FEATURES = [
    name
    for prefix, spec in HISTORY_SPECS.items()
    for name in _feature_names(prefix, spec["windows"])
]


def build_history_table(
    path: str,
    prefix: str,
    keys: tuple[str, ...],
    windows: tuple[int, ...],
) -> pl.DataFrame:
    """Agrège les vols antérieurs avec des fenêtres fermées avant le jour cible."""

    data = (
        pl.scan_csv(
            path,
            null_values=[""],
            infer_schema_length=10_000,
            low_memory=True,
        )
        .select("fl_date", *keys, "cancelled", "diverted", "arr_delay")
        .with_columns(
            pl.col("fl_date").str.to_date().alias("flight_date"),
            (
                (pl.col("cancelled") == 0)
                & (pl.col("diverted") == 0)
                & pl.col("arr_delay").is_not_null()
            )
            .cast(pl.Int8)
            .alias("completed_flag"),
            (
                (pl.col("cancelled") == 0)
                & (pl.col("diverted") == 0)
                & (pl.col("arr_delay") > 0).fill_null(False)
            )
            .cast(pl.Int8)
            .alias("delayed_flag"),
            (
                (pl.col("cancelled") == 1)
                | (pl.col("diverted") == 1)
                | (pl.col("arr_delay") > 0).fill_null(False)
            )
            .cast(pl.Int8)
            .alias("disrupted_flag"),
        )
    )
    groups = ["flight_date", *keys]
    daily = (
        data.group_by(*groups)
        .agg(
            pl.len().alias("flights"),
            pl.col("completed_flag").sum().alias("completed"),
            pl.col("delayed_flag").sum().alias("delayed"),
            pl.col("disrupted_flag").sum().alias("disrupted"),
        )
        .collect(engine="streaming")
        .sort(*keys, "flight_date")
    )

    rolling_expressions = []
    for window in windows:
        for source in (
            "flights",
            "completed",
            "delayed",
            "disrupted",
        ):
            expression = pl.col(source).rolling_sum_by(
                "flight_date",
                f"{window}d",
                closed="left",
                min_samples=1,
            )
            if keys:
                expression = expression.over(list(keys))
            rolling_expressions.append(expression.alias(f"{source}_{window}d"))
    daily = daily.with_columns(*rolling_expressions)

    feature_expressions = []
    for window in windows:
        feature_expressions.extend(
            [
                (
                    pl.col(f"delayed_{window}d")
                    / pl.col(f"completed_{window}d")
                ).alias(f"{prefix}_delay_rate_{window}d"),
                (
                    pl.col(f"disrupted_{window}d")
                    / pl.col(f"flights_{window}d")
                ).alias(f"{prefix}_disruption_rate_{window}d"),
                pl.col(f"flights_{window}d")
                .cast(pl.Float32)
                .alias(f"{prefix}_flight_count_{window}d"),
            ]
        )
    feature_names = _feature_names(prefix, windows)
    return daily.with_columns(*feature_expressions).select(
        *groups,
        *feature_names,
    )


def add_historical_features(data: pl.DataFrame, path: str) -> pl.DataFrame:
    """Joint les historiques calculés sur le CSV complet avant échantillonnage."""

    enriched = data
    for prefix, spec in HISTORY_SPECS.items():
        keys = spec["keys"]
        history = build_history_table(path, prefix, keys, spec["windows"])
        enriched = enriched.join(
            history,
            on=["flight_date", *keys],
            how="left",
        )
    return enriched


def build_history_profiles(data: pd.DataFrame) -> dict[str, Any]:
    """Conserve les statistiques les plus récentes pour une future prédiction."""

    profiles: dict[str, Any] = {
        "defaults": {
            name: float(data[name].median())
            if data[name].notna().any()
            else 0.0
            for name in HISTORICAL_FEATURES
        }
    }
    ordered = data.sort_values("flight_date")
    for prefix, spec in HISTORY_SPECS.items():
        keys = spec["keys"]
        feature_names = _feature_names(prefix, spec["windows"])
        if not keys:
            latest = ordered.iloc[-1]
            profiles[prefix] = {
                name: _finite_or_default(
                    latest[name], profiles["defaults"][name]
                )
                for name in feature_names
            }
            continue

        latest_rows = ordered.groupby(list(keys), dropna=False).tail(1)
        group_profiles = {}
        for _, row in latest_rows.iterrows():
            profile_key = _profile_key(tuple(str(row[key]) for key in keys))
            group_profiles[profile_key] = {
                name: _finite_or_default(
                    row[name], profiles["defaults"][name]
                )
                for name in feature_names
            }
        profiles[prefix] = group_profiles
    return profiles


def add_profile_values(
    row: dict[str, object],
    flight: dict[str, object],
    profiles: dict[str, Any] | None,
) -> None:
    """Ajoute au futur vol les derniers historiques disponibles dans l'artefact."""

    if not profiles:
        row.update({name: np.nan for name in HISTORICAL_FEATURES})
        return

    row.update(profiles["defaults"])
    for prefix, spec in HISTORY_SPECS.items():
        keys = spec["keys"]
        if not keys:
            row.update(profiles.get(prefix, {}))
            continue
        profile_key = _profile_key(tuple(str(flight[key]) for key in keys))
        row.update(profiles.get(prefix, {}).get(profile_key, {}))


def _profile_key(values: tuple[str, ...]) -> str:
    return "|".join(values)


def _finite_or_default(value: object, default: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else default
