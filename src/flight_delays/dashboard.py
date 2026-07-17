"""Données et transformations partagées par l'application Streamlit."""

from __future__ import annotations

import copy
import json
from datetime import date, time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MONTH_LABELS = {
    1: "Janvier",
    2: "Février",
    3: "Mars",
    4: "Avril",
    5: "Mai",
    6: "Juin",
    7: "Juillet",
    8: "Août",
    9: "Septembre",
    10: "Octobre",
    11: "Novembre",
    12: "Décembre",
}

MODEL_CANDIDATES = (
    "models/official/flight_delay_models.joblib",
    "models/flight_delay_models.joblib",
    "models/notebook_demo/flight_delay_models.joblib",
)

METRICS_CANDIDATES = (
    "models/official/training_metrics.json",
    "models/training_metrics.json",
    "models/notebook_demo/training_metrics.json",
)

IMPORTANCE_CANDIDATES = (
    "models/official/feature_importance.csv",
    "models/feature_importance.csv",
    "models/notebook_demo/feature_importance.csv",
)


# Instantané compact du run officiel v6. Le dashboard reste consultable sur un
# clone neuf sans versionner l'artefact Joblib ni le CSV complet.
REFERENCE_METRICS: dict[str, Any] = {
    "artifact_version": 6,
    "target_definition": {
        "source": "arr_delay",
        "operator": ">",
        "threshold_minutes": 0,
        "label": "is_delayed",
    },
    "data": {
        "all": {
            "row_count": 696_596,
            "delayed_count": 252_517,
            "delayed_percentage": 36.250136377469865,
        },
        "train": {
            "row_count": 401_778,
            "delayed_count": 157_369,
            "delayed_percentage": 39.16814758398917,
        },
        "tuning": {
            "row_count": 60_378,
            "delayed_count": 23_189,
            "delayed_percentage": 38.40637318228494,
        },
        "validation": {
            "row_count": 118_740,
            "delayed_count": 34_063,
            "delayed_percentage": 28.687047330301503,
        },
        "test": {
            "row_count": 115_700,
            "delayed_count": 37_896,
            "delayed_percentage": 32.75367329299914,
        },
    },
    "delay_classification": {
        "threshold_selection": {
            "threshold": 0.4481171691041839,
            "strategy": "fallback_target_alert_coverage",
            "feasible_on_validation": False,
            "f1_threshold": 0.2708387409959624,
        },
        "business_gate": {
            "passed": False,
            "requirements": {
                "minimum_precision": 0.5,
                "minimum_recall": 0.2,
                "minimum_alert_coverage": 0.05,
                "maximum_alert_coverage": 0.2,
                "minimum_alert_count": 500,
                "confidence_level": 0.95,
            },
            "violations": [
                "Aucun seuil ne respecte toutes les contraintes sur la validation.",
                "Les contraintes métier ne sont pas respectées pour chaque mois de test.",
            ],
        },
        "validation": {
            "sample_count": 118_740,
            "positive_count": 34_063,
            "alert_count": 14_843,
            "alert_coverage": 0.1250042108809163,
            "precision": 0.4908037458734757,
            "recall": 0.21386842028006928,
            "f1": 0.29791845581319265,
            "roc_auc": 0.6571298231659954,
            "average_precision": 0.4230065712957427,
            "true_negative": 77_119,
            "false_positive": 7_558,
            "false_negative": 26_778,
            "true_positive": 7_285,
            "precision_confidence_interval_95": {
                "lower": 0.4827648004166684,
                "upper": 0.4988474501918401,
            },
            "recall_confidence_interval_95": {
                "lower": 0.2095464184079455,
                "upper": 0.21825495191977498,
            },
        },
        "test": {
            "sample_count": 115_700,
            "positive_count": 37_896,
            "alert_count": 19_629,
            "alert_coverage": 0.1696542783059637,
            "precision": 0.5089408528198074,
            "recall": 0.2636162127929069,
            "f1": 0.3473272490221643,
            "roc_auc": 0.6431034714896139,
            "average_precision": 0.45897972770695394,
            "true_negative": 68_165,
            "false_positive": 9_639,
            "false_negative": 27_906,
            "true_positive": 9_990,
            "precision_confidence_interval_95": {
                "lower": 0.5019462071714275,
                "upper": 0.5159319996452583,
            },
            "recall_confidence_interval_95": {
                "lower": 0.25920435240325723,
                "upper": 0.26807599204181737,
            },
        },
        "test_by_month": {
            "11": {
                "sample_count": 57_189,
                "alert_count": 5_460,
                "alert_coverage": 0.09547290562870482,
                "precision": 0.49139194139194137,
                "recall": 0.16246820879253968,
            },
            "12": {
                "sample_count": 58_511,
                "alert_count": 14_169,
                "alert_coverage": 0.2421595939225103,
                "precision": 0.5157032959277296,
                "recall": 0.3417360396595267,
            },
        },
        "f1_threshold_comparison": {
            "threshold": 0.2708387409959624,
            "test": {
                "alert_count": 77_153,
                "alert_coverage": 0.6668366464995679,
                "precision": 0.3865954661516727,
                "recall": 0.7870751530504538,
                "f1": 0.5185095046458466,
            },
        },
        "test_baselines": {
            "always_negative": {"accuracy": 0.6724632670700087, "f1": 0.0},
            "always_positive": {
                "accuracy": 0.32753673292999136,
                "f1": 0.49345035026953826,
            },
        },
    },
}


REFERENCE_FEATURE_IMPORTANCE = [
    ("arrival_hour", 5.717504551964716),
    ("departure_time_sin", 5.464870531101951),
    ("dest_hour", 5.303388859715049),
    ("dest_delay_rate_1d", 5.283099408669815),
    ("departure_hour", 4.965178602012107),
    ("scheduled_departure_minutes", 4.935473955887735),
    ("route_delay_rate_7d", 4.884483824406976),
    ("route_disruption_rate_7d", 4.85222042879249),
    ("day_of_week_category", 4.772268682216807),
    ("op_unique_carrier", 4.568694162880508),
    ("route_delay_rate_28d", 4.426535302472779),
    ("origin_disruption_rate_1d", 3.970249591331937),
    ("carrier_delay_rate_1d", 3.9535829584850593),
    ("origin_delay_rate_1d", 3.2216614094345433),
    ("origin_hour", 3.0619819713316723),
]


def _first_existing(project_root: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        path = project_root / candidate
        if path.is_file():
            return path
    return None


def find_model_path(project_root: Path) -> Path | None:
    """Retourne le premier artefact local selon l'ordre de priorité."""

    return _first_existing(project_root, MODEL_CANDIDATES)


def load_dashboard_metrics(project_root: Path) -> tuple[dict[str, Any], str]:
    """Charge les métriques locales ou l'instantané versionné de référence."""

    metrics_path = _first_existing(project_root, METRICS_CANDIDATES)
    if metrics_path is None:
        return copy.deepcopy(REFERENCE_METRICS), "instantané officiel v6"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("artifact_version") != 6:
            raise ValueError("Version de métriques incompatible.")
        if metrics.get("target_definition", {}).get("label") != "is_delayed":
            raise ValueError("Cible de métriques incompatible.")
        classification = metrics["delay_classification"]
        classification["test"]
        classification["business_gate"]
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ):
        return copy.deepcopy(REFERENCE_METRICS), "instantané officiel v6"
    relative_path = metrics_path.relative_to(project_root)
    return metrics, f"fichier local {relative_path}"


def load_feature_importance(project_root: Path) -> tuple[pd.DataFrame, str]:
    """Charge l'importance locale ou une sélection du run officiel."""

    importance_path = _first_existing(project_root, IMPORTANCE_CANDIDATES)
    if importance_path is None:
        frame = pd.DataFrame(
            REFERENCE_FEATURE_IMPORTANCE, columns=["feature", "importance"]
        )
        return frame, "instantané officiel v6"
    try:
        frame = pd.read_csv(importance_path, usecols=["feature", "importance"])
        frame["importance"] = pd.to_numeric(frame["importance"], errors="raise")
        feature_names = frame["feature"].astype(str)
        if feature_names.str.contains("average_delay_minutes", regex=False).any():
            raise ValueError("Importance issue d'un ancien artefact.")
    except (OSError, ValueError, KeyError):
        frame = pd.DataFrame(
            REFERENCE_FEATURE_IMPORTANCE, columns=["feature", "importance"]
        )
        return frame, "instantané officiel v6"
    relative_path = importance_path.relative_to(project_root)
    return frame, f"fichier local {relative_path}"


def load_flight_sample(path: Path | str) -> pd.DataFrame:
    """Charge l'échantillon versionné sans lire les colonnes de causes."""

    columns = [
        "month",
        "day_of_month",
        "day_of_week",
        "fl_date",
        "op_unique_carrier",
        "op_carrier_fl_num",
        "origin",
        "origin_city_name",
        "origin_state_nm",
        "dest",
        "dest_city_name",
        "dest_state_nm",
        "crs_dep_time",
        "crs_arr_time",
        "arr_delay",
        "cancelled",
        "diverted",
        "crs_elapsed_time",
        "distance",
    ]
    data = pd.read_csv(path, usecols=columns, parse_dates=["fl_date"])
    for name in (
        "month",
        "day_of_month",
        "day_of_week",
        "cancelled",
        "diverted",
    ):
        data[name] = pd.to_numeric(data[name], errors="coerce").fillna(0).astype(int)
    data["is_completed"] = (
        data["cancelled"].eq(0)
        & data["diverted"].eq(0)
        & data["arr_delay"].notna()
    )
    data["is_delayed"] = data["is_completed"] & data["arr_delay"].gt(0)
    data["flight_status"] = np.select(
        [
            data["cancelled"].eq(1),
            data["diverted"].eq(1),
            data["is_delayed"],
        ],
        ["Annulé", "Dérouté", "Arrivé en retard"],
        default="À l'heure ou en avance",
    )
    data["month_label"] = data["month"].map(MONTH_LABELS)
    return data


def filter_flights(
    data: pd.DataFrame,
    months: Iterable[int] | None = None,
    carriers: Iterable[str] | None = None,
    origins: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Applique les filtres interactifs de l'explorateur."""

    filtered = data
    selected_months = list(months or [])
    selected_carriers = list(carriers or [])
    selected_origins = list(origins or [])
    if selected_months:
        filtered = filtered.loc[filtered["month"].isin(selected_months)]
    if selected_carriers:
        filtered = filtered.loc[
            filtered["op_unique_carrier"].isin(selected_carriers)
        ]
    if selected_origins:
        filtered = filtered.loc[filtered["origin"].isin(selected_origins)]
    return filtered.copy()


def overview_statistics(data: pd.DataFrame) -> dict[str, float | int]:
    """Calcule les indicateurs globaux d'un ensemble filtré."""

    completed = data.loc[data["is_completed"]]
    delayed_count = int(completed["is_delayed"].sum())
    return {
        "flight_count": int(len(data)),
        "completed_count": int(len(completed)),
        "delayed_count": delayed_count,
        "delay_rate": delayed_count / len(completed) if len(completed) else 0.0,
        "cancelled_count": int(data["cancelled"].eq(1).sum()),
        "diverted_count": int(data["diverted"].eq(1).sum()),
    }


def monthly_delay_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Agrège volume et taux de retard par mois."""

    completed = data.loc[data["is_completed"]]
    if completed.empty:
        return pd.DataFrame(columns=["month", "month_label", "flights", "delay_rate"])
    metrics = (
        completed.groupby("month", as_index=False)
        .agg(flights=("is_delayed", "size"), delay_rate=("is_delayed", "mean"))
        .sort_values("month")
    )
    metrics["month_label"] = metrics["month"].map(MONTH_LABELS)
    return metrics


def grouped_delay_metrics(
    data: pd.DataFrame,
    group_column: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Agrège les groupes les plus représentés parmi les vols achevés."""

    completed = data.loc[data["is_completed"]]
    if completed.empty:
        return pd.DataFrame(columns=[group_column, "flights", "delay_rate"])
    return (
        completed.groupby(group_column, as_index=False)
        .agg(flights=("is_delayed", "size"), delay_rate=("is_delayed", "mean"))
        .sort_values(["flights", "delay_rate"], ascending=[False, False])
        .head(limit)
    )


def airport_state_mapping(data: pd.DataFrame) -> dict[str, str]:
    """Construit la correspondance aéroport vers nom complet de l'État."""

    origin = data[["origin", "origin_state_nm"]].rename(
        columns={"origin": "airport", "origin_state_nm": "state"}
    )
    destination = data[["dest", "dest_state_nm"]].rename(
        columns={"dest": "airport", "dest_state_nm": "state"}
    )
    airports = (
        pd.concat([origin, destination], ignore_index=True)
        .dropna()
        .drop_duplicates("airport")
        .sort_values("airport")
    )
    return dict(zip(airports["airport"], airports["state"], strict=False))


def time_to_hhmm(value: time) -> int:
    """Convertit une heure Python en entier HHMM."""

    return value.hour * 100 + value.minute


def build_flight_payload(
    *,
    flight_date: date,
    carrier: str,
    flight_number: int,
    origin: str,
    origin_state: str,
    destination: str,
    destination_state: str,
    scheduled_departure: time,
    scheduled_arrival: time,
    scheduled_duration: float,
    distance: float,
) -> dict[str, object]:
    """Transforme le formulaire en entrée validée pour ``predict_flight``."""

    carrier = carrier.strip().upper()
    origin = origin.strip().upper()
    destination = destination.strip().upper()
    if not carrier or not origin or not destination:
        raise ValueError("La compagnie et les deux aéroports sont obligatoires.")
    if origin == destination:
        raise ValueError("Les aéroports de départ et d'arrivée doivent être différents.")
    if int(flight_number) <= 0:
        raise ValueError("Le numéro de vol doit être strictement positif.")
    if float(scheduled_duration) <= 0:
        raise ValueError("La durée prévue doit être strictement positive.")
    if float(distance) < 0:
        raise ValueError("La distance ne peut pas être négative.")
    return {
        "flight_date": flight_date.isoformat(),
        "month": flight_date.month,
        "day_of_month": flight_date.day,
        "day_of_week": flight_date.isoweekday(),
        "op_unique_carrier": carrier,
        "op_carrier_fl_num": int(flight_number),
        "origin": origin,
        "origin_state_nm": origin_state,
        "dest": destination,
        "dest_state_nm": destination_state,
        "crs_dep_time": time_to_hhmm(scheduled_departure),
        "crs_arr_time": time_to_hhmm(scheduled_arrival),
        "crs_elapsed_time": float(scheduled_duration),
        "distance": float(distance),
    }


def prediction_presentation(result: dict[str, Any]) -> dict[str, Any]:
    """Sépare strictement la décision publiable du diagnostic expérimental."""

    publishable = bool(result.get("prediction_publishable", False))
    return {
        "publishable": publishable,
        "published_alert": (
            result.get("published_delay_alert") if publishable else None
        ),
        "diagnostic_probability": float(result.get("delay_probability", 0.0)),
        "diagnostic_threshold": float(result.get("classification_threshold", 0.5)),
        "diagnostic_class": bool(
            result.get("diagnostic_is_delayed_prediction", False)
        ),
        "blockers": list(result.get("publication_blockers", [])),
        "historical_context_date": result.get("historical_context_date"),
        "schedule_context_source": result.get("schedule_context_source", "missing"),
    }
