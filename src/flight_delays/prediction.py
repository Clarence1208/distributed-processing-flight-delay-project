"""Chargement des modèles et prédiction d'un futur vol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from flight_delays.ml_data import FEATURE_COLUMNS, prepare_prediction_frame
from flight_delays.schedule import SCHEDULE_FEATURES


CURRENT_ARTIFACT_VERSION = 6


def load_model_bundle(path: str) -> dict:
    """Charge uniquement un artefact Joblib produit localement et jugé fiable."""

    bundle = joblib.load(path)
    artifact_version = bundle.get("artifact_version")
    if artifact_version != CURRENT_ARTIFACT_VERSION:
        raise ValueError(
            "L'artefact est incompatible avec le pipeline actuel : "
            f"version {artifact_version!r} trouvée, "
            f"version {CURRENT_ARTIFACT_VERSION} attendue. Réentraînez le modèle."
        )
    return bundle


def _positive_probability(model, features) -> float:
    classes = list(model.classes_)
    if 1 not in classes:
        return 0.0
    return float(model.predict_proba(features)[0, classes.index(1)])


def predict_flight(bundle: dict, flight: dict[str, object]) -> dict:
    """Retourne la probabilité et la décision de retard d'un futur vol."""

    frame = prepare_prediction_frame(
        flight,
        history_profiles=bundle.get("history_profiles"),
        schedule_profiles=bundle.get("schedule_profiles"),
    )
    delay_features = frame[bundle.get("delay_feature_columns", FEATURE_COLUMNS)]
    delay_probability = _positive_probability(
        bundle["delay_classifier"], delay_features
    )
    is_delayed = delay_probability >= bundle["delay_threshold"]
    business_readiness = bundle.get(
        "business_readiness",
        {
            "ready": False,
            "status": "not_ready",
            "violations": [
                "L'artefact ne contient pas de validation métier exploitable."
            ],
        },
    )
    business_ready = bool(business_readiness.get("ready", False))
    provided_schedule_features = [
        name for name in SCHEDULE_FEATURES if flight.get(name) is not None
    ]
    exact_schedule_context = len(provided_schedule_features) == len(SCHEDULE_FEATURES)
    if exact_schedule_context:
        schedule_context_source = "provided_daily_schedule"
    elif provided_schedule_features and bundle.get("schedule_profiles"):
        schedule_context_source = "typical_schedule_profile_with_overrides"
    elif bundle.get("schedule_profiles"):
        schedule_context_source = "typical_schedule_profile"
    else:
        schedule_context_source = "missing"
    prediction_publishable = business_ready and exact_schedule_context
    publication_blockers = list(business_readiness.get("violations", []))
    if not exact_schedule_context:
        publication_blockers.append(
            "Les six volumes du planning journalier exact ne sont pas fournis."
        )
    if prediction_publishable:
        prediction_status = "publishable"
    elif not business_ready:
        prediction_status = "experimental_model_not_ready"
    else:
        prediction_status = "experimental_missing_daily_schedule"
    published_delay_alert = bool(is_delayed) if prediction_publishable else None
    return {
        "artifact_version": bundle.get("artifact_version"),
        "target_definition": bundle.get("target_definition"),
        "model_business_ready": business_ready,
        "model_business_status": business_readiness.get("status", "not_ready"),
        "business_gate_violations": business_readiness.get("violations", []),
        "prediction_publishable": prediction_publishable,
        "publication_blockers": publication_blockers,
        "prediction_status": prediction_status,
        "published_delay_alert": published_delay_alert,
        "historical_context_date": bundle.get("history_cutoff_date"),
        "schedule_context_source": schedule_context_source,
        "delay_probability": delay_probability,
        "classification_threshold": float(bundle["delay_threshold"]),
        "classification_threshold_strategy": bundle.get(
            "delay_threshold_strategy", "unknown"
        ),
        "diagnostic_is_delayed_prediction": bool(is_delayed),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prédit le retard d'un vol décrit dans un fichier JSON."
    )
    parser.add_argument(
        "--model", default="models/official/flight_delay_models.joblib"
    )
    parser.add_argument("--flight-json", required=True)
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    flight = json.loads(Path(arguments.flight_json).read_text(encoding="utf-8"))
    bundle = load_model_bundle(arguments.model)
    print(json.dumps(predict_flight(bundle, flight), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
