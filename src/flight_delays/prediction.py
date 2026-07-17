"""Chargement des modèles et prédiction d'un futur vol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from flight_delays.ml_data import (
    FEATURE_COLUMNS,
    MAX_PREDICTED_DELAY_MINUTES,
    prepare_prediction_frame,
)


def load_model_bundle(path: str) -> dict:
    """Charge uniquement un artefact Joblib produit localement et jugé fiable."""

    return joblib.load(path)


def _positive_probability(model, features) -> float:
    classes = list(model.classes_)
    if 1 not in classes:
        return 0.0
    return float(model.predict_proba(features)[0, classes.index(1)])


def predict_flight(bundle: dict, flight: dict[str, object]) -> dict:
    """Retourne probabilité, minutes conditionnelles et raisons possibles."""

    frame = prepare_prediction_frame(flight, bundle.get("history_profiles"))
    delay_features = frame[bundle.get("delay_feature_columns", FEATURE_COLUMNS)]
    regression_features = frame[
        bundle.get("regression_feature_columns", FEATURE_COLUMNS)
    ]
    reason_features = frame[bundle.get("reason_feature_columns", FEATURE_COLUMNS)]
    delay_probability = _positive_probability(
        bundle["delay_classifier"], delay_features
    )
    is_delayed = delay_probability >= bundle["delay_threshold"]
    conditional_minutes = float(
        np.clip(
            bundle["delay_regressor"].predict(regression_features)[0],
            15.0,
            float(MAX_PREDICTED_DELAY_MINUTES),
        )
    )
    reason_probabilities = {
        reason: _positive_probability(model, reason_features)
        for reason, model in bundle["reason_classifiers"].items()
    }
    ordered_reasons = dict(
        sorted(reason_probabilities.items(), key=lambda item: item[1], reverse=True)
    )
    predicted_reasons = [
        reason
        for reason, probability in ordered_reasons.items()
        if probability >= bundle["reason_thresholds"][reason]
    ]
    return {
        "historical_context_date": bundle.get("history_cutoff_date"),
        "delay_probability": delay_probability,
        "classification_threshold": float(bundle["delay_threshold"]),
        "is_delayed_15_prediction": bool(is_delayed),
        "estimated_delay_minutes_if_delayed": conditional_minutes,
        "predicted_delay_minutes": conditional_minutes if is_delayed else 0.0,
        "reason_probabilities_if_delayed": ordered_reasons,
        "predicted_reasons_if_delayed": predicted_reasons,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prédit le retard d'un vol décrit dans un fichier JSON."
    )
    parser.add_argument("--model", default="models/flight_delay_models.joblib")
    parser.add_argument("--flight-json", required=True)
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    flight = json.loads(Path(arguments.flight_json).read_text(encoding="utf-8"))
    bundle = load_model_bundle(arguments.model)
    print(json.dumps(predict_flight(bundle, flight), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
