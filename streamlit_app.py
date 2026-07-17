"""Interface Streamlit du projet de prédiction des retards."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from flight_delays.dashboard import (
    MONTH_LABELS,
    airport_state_mapping,
    build_flight_payload,
    filter_flights,
    find_model_path,
    grouped_delay_metrics,
    load_dashboard_metrics,
    load_feature_importance,
    load_flight_sample,
    monthly_delay_metrics,
    overview_statistics,
    prediction_presentation,
)
from flight_delays.prediction import load_model_bundle, predict_flight


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_ROOT / "data/flight_data_2024_sample.csv"
MONTH_ORDER = list(MONTH_LABELS.values())
NAVIGATION = (
    "Vue d'ensemble",
    "Explorer les vols",
    "Performance du modèle",
    "Diagnostic d'un vol",
)


st.set_page_config(
    page_title="Retards des vols US",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _load_sample(path: str, modified_at_ns: int) -> pd.DataFrame:
    del modified_at_ns
    return load_flight_sample(Path(path))


@st.cache_data(show_spinner=False)
def _load_metrics(modified_at_ns: int | None) -> tuple[dict, str]:
    del modified_at_ns
    return load_dashboard_metrics(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def _load_importance(modified_at_ns: int | None) -> tuple[pd.DataFrame, str]:
    del modified_at_ns
    return load_feature_importance(PROJECT_ROOT)


@st.cache_resource(show_spinner="Chargement du modèle CatBoost…")
def _load_bundle(path: str, modified_at_ns: int) -> dict:
    del modified_at_ns
    return load_model_bundle(path)


def _candidate_mtime(pattern: str) -> int | None:
    candidates = sorted(PROJECT_ROOT.glob(pattern))
    return max((path.stat().st_mtime_ns for path in candidates), default=None)


def _percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f} %"


def _number(value: int | float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _month_chart(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month_label:N",
                title="Mois",
                sort=MONTH_ORDER,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("delay_rate:Q", title="Taux de retard", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("month_label:N", title="Mois"),
                alt.Tooltip("flights:Q", title="Vols achevés", format=","),
                alt.Tooltip("delay_rate:Q", title="Taux", format=".1%"),
            ],
        )
        .properties(height=330)
    )


def _status_chart(data: pd.DataFrame) -> alt.Chart:
    status = (
        data.groupby("flight_status", as_index=False)
        .size()
        .rename(columns={"size": "flights"})
        .sort_values("flights")
    )
    return (
        alt.Chart(status)
        .mark_bar()
        .encode(
            x=alt.X("flights:Q", title="Nombre de vols"),
            y=alt.Y("flight_status:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("flight_status:N", title="Statut"),
                alt.Tooltip("flights:Q", title="Vols", format=","),
            ],
        )
        .properties(height=330)
    )


def _render_gate(metrics: dict) -> None:
    classification = metrics["delay_classification"]
    gate = classification["business_gate"]
    if gate["passed"]:
        st.success(
            "Le modèle respecte le contrat métier complet et peut fournir une "
            "alerte lorsque le planning journalier exact est disponible."
        )
        return
    test = classification["test"]
    minimum_precision = gate["requirements"]["minimum_precision"]
    st.warning(
        "Le modèle ne respecte pas encore le contrat métier complet. Sur le test "
        f"global, sa précision est de {_percent(test['precision'])} pour un objectif "
        f"d'au moins {_percent(minimum_precision)}, avec {_percent(test['recall'])} "
        f"de rappel et {_percent(test['alert_coverage'])} de vols signalés."
    )
    with st.expander("Pourquoi le modèle reste expérimental"):
        for violation in gate.get("violations", []):
            st.markdown(f"- {violation}")


def render_overview(sample: pd.DataFrame, metrics: dict, metrics_source: str) -> None:
    st.title("Prédiction des retards des vols domestiques US")
    st.caption(
        "Un vol est considéré en retard dès que son retard à l'arrivée est "
        "strictement supérieur à zéro minute."
    )
    _render_gate(metrics)

    statistics = overview_statistics(sample)
    columns = st.columns(4)
    columns[0].metric("Vols de l'échantillon Spark", _number(statistics["flight_count"]))
    columns[1].metric("Vols achevés", _number(statistics["completed_count"]))
    columns[2].metric("Arrivés en retard", _number(statistics["delayed_count"]))
    columns[3].metric("Taux de retard", _percent(float(statistics["delay_rate"])))

    st.subheader("Ce que montrent les données")
    left, right = st.columns(2)
    with left:
        st.markdown("**Évolution mensuelle du taux de retard**")
        st.altair_chart(_month_chart(monthly_delay_metrics(sample)), width="stretch")
    with right:
        st.markdown("**Répartition des statuts de vol**")
        st.altair_chart(_status_chart(sample), width="stretch")

    st.subheader("Résultat actuel du modèle")
    test = metrics["delay_classification"]["test"]
    model_columns = st.columns(4)
    model_columns[0].metric("Précision test", _percent(test["precision"]))
    model_columns[1].metric("Rappel test", _percent(test["recall"]))
    model_columns[2].metric("Vols signalés", _percent(test["alert_coverage"]))
    model_columns[3].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    st.caption(
        "La précision répond à la question : parmi les vols signalés, quelle "
        "proportion est réellement arrivée en retard ? Source : " + metrics_source + "."
    )


def render_explorer(sample: pd.DataFrame) -> None:
    st.title("Explorer les 10 000 vols Spark")
    st.caption(
        "Les filtres agissent sur l'échantillon versionné. Les colonnes de causes "
        "postérieures au vol ne sont ni chargées ni affichées."
    )

    first, second, third = st.columns(3)
    with first:
        months = st.multiselect(
            "Mois",
            options=list(MONTH_LABELS),
            format_func=lambda value: MONTH_LABELS[value],
            key="explorer_months",
        )
    with second:
        carriers = st.multiselect(
            "Compagnies",
            options=sorted(sample["op_unique_carrier"].dropna().unique()),
            key="explorer_carriers",
        )
    with third:
        origins = st.multiselect(
            "Aéroports de départ",
            options=sorted(sample["origin"].dropna().unique()),
            key="explorer_origins",
        )

    filtered = filter_flights(sample, months, carriers, origins)
    if filtered.empty:
        st.info("Aucun vol ne correspond aux filtres sélectionnés.")
        return

    statistics = overview_statistics(filtered)
    summary_columns = st.columns(4)
    summary_columns[0].metric("Vols filtrés", _number(statistics["flight_count"]))
    summary_columns[1].metric("Taux de retard", _percent(float(statistics["delay_rate"])))
    summary_columns[2].metric("Annulations", _number(statistics["cancelled_count"]))
    summary_columns[3].metric("Déroutements", _number(statistics["diverted_count"]))

    left, right = st.columns(2)
    with left:
        st.markdown("**Taux de retard par mois**")
        st.altair_chart(
            _month_chart(monthly_delay_metrics(filtered)),
            width="stretch",
        )
    with right:
        carrier_metrics = grouped_delay_metrics(
            filtered, "op_unique_carrier", limit=10
        )
        carrier_chart = (
            alt.Chart(carrier_metrics)
            .mark_bar()
            .encode(
                x=alt.X(
                    "delay_rate:Q",
                    title="Taux de retard",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("op_unique_carrier:N", title="Compagnie", sort="-x"),
                tooltip=[
                    alt.Tooltip("op_unique_carrier:N", title="Compagnie"),
                    alt.Tooltip("flights:Q", title="Vols", format=","),
                    alt.Tooltip("delay_rate:Q", title="Taux", format=".1%"),
                ],
            )
            .properties(height=330)
        )
        st.markdown("**Compagnies les plus représentées**")
        st.altair_chart(carrier_chart, width="stretch")

    completed = filtered.loc[filtered["is_completed"]].copy()
    completed["display_delay"] = completed["arr_delay"].clip(-30, 180)
    histogram = (
        alt.Chart(completed)
        .mark_bar()
        .encode(
            x=alt.X(
                "display_delay:Q",
                title="Retard à l'arrivée (minutes, valeurs limitées de −30 à 180)",
                bin=alt.Bin(maxbins=42),
            ),
            y=alt.Y("count():Q", title="Nombre de vols"),
            tooltip=[alt.Tooltip("count():Q", title="Vols")],
        )
        .properties(height=300)
    )
    st.markdown("**Distribution du retard à l'arrivée**")
    st.altair_chart(histogram, width="stretch")

    display_columns = [
        "fl_date",
        "op_unique_carrier",
        "op_carrier_fl_num",
        "origin",
        "dest",
        "crs_dep_time",
        "crs_arr_time",
        "arr_delay",
        "flight_status",
    ]
    st.markdown("**Vols correspondants**")
    st.dataframe(
        filtered[display_columns].rename(
            columns={
                "fl_date": "Date",
                "op_unique_carrier": "Compagnie",
                "op_carrier_fl_num": "Numéro",
                "origin": "Départ",
                "dest": "Arrivée",
                "crs_dep_time": "Départ prévu",
                "crs_arr_time": "Arrivée prévue",
                "arr_delay": "Retard (min)",
                "flight_status": "Statut",
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )


def _confusion_chart(test: dict) -> alt.Chart:
    confusion = pd.DataFrame(
        [
            {"Réel": "À l'heure", "Prédit": "À l'heure", "Vols": test["true_negative"]},
            {"Réel": "À l'heure", "Prédit": "Retard", "Vols": test["false_positive"]},
            {"Réel": "Retard", "Prédit": "À l'heure", "Vols": test["false_negative"]},
            {"Réel": "Retard", "Prédit": "Retard", "Vols": test["true_positive"]},
        ]
    )
    base = alt.Chart(confusion).encode(
        x=alt.X("Prédit:N", title="Classe prédite"),
        y=alt.Y("Réel:N", title="Classe réelle"),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("Vols:Q", title="Vols"),
        tooltip=["Réel:N", "Prédit:N", alt.Tooltip("Vols:Q", format=",")],
    )
    labels = base.mark_text().encode(text=alt.Text("Vols:Q", format=","))
    return (heatmap + labels).properties(height=280)


def render_performance(metrics: dict, metrics_source: str) -> None:
    st.title("Performance du classifieur")
    st.caption(
        "Le modèle répond uniquement à la question : le vol arrivera-t-il avec "
        "un retard strictement positif ?"
    )
    _render_gate(metrics)

    classification = metrics["delay_classification"]
    validation = classification["validation"]
    test = classification["test"]
    requirements = classification["business_gate"]["requirements"]

    columns = st.columns(4)
    columns[0].metric(
        "Précision test",
        _percent(test["precision"]),
        help="Objectif : borne basse à 95 % supérieure ou égale à 50 %.",
    )
    columns[1].metric("Rappel test", _percent(test["recall"]))
    columns[2].metric("Couverture test", _percent(test["alert_coverage"]))
    columns[3].metric("Alertes test", _number(test["alert_count"]))

    score_rows = []
    for split_name, split_metrics in (
        ("Validation", validation),
        ("Test", test),
    ):
        for metric_name, key in (
            ("Précision", "precision"),
            ("Rappel", "recall"),
            ("Couverture", "alert_coverage"),
        ):
            score_rows.append(
                {
                    "Jeu": split_name,
                    "Métrique": metric_name,
                    "Valeur": split_metrics[key],
                }
            )
    score_frame = pd.DataFrame(score_rows)
    score_chart = (
        alt.Chart(score_frame)
        .mark_bar()
        .encode(
            x=alt.X("Métrique:N", title=None),
            xOffset="Jeu:N",
            y=alt.Y("Valeur:Q", title="Proportion", axis=alt.Axis(format="%")),
            color=alt.Color("Jeu:N", title=None),
            tooltip=["Jeu:N", "Métrique:N", alt.Tooltip("Valeur:Q", format=".1%")],
        )
        .properties(height=330)
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Validation et test au seuil métier**")
        st.altair_chart(score_chart, width="stretch")
        st.caption(
            f"Objectifs : précision ≥ {requirements['minimum_precision']:.0%}, "
            f"rappel ≥ {requirements['minimum_recall']:.0%}, couverture entre "
            f"{requirements['minimum_alert_coverage']:.0%} et "
            f"{requirements['maximum_alert_coverage']:.0%}."
        )
    with right:
        st.markdown("**Matrice de confusion sur le test**")
        st.altair_chart(_confusion_chart(test), width="stretch")

    monthly_rows = []
    for month, month_metrics in classification["test_by_month"].items():
        month_label = MONTH_LABELS[int(month)]
        for metric_name, key in (
            ("Précision", "precision"),
            ("Rappel", "recall"),
            ("Couverture", "alert_coverage"),
        ):
            monthly_rows.append(
                {
                    "Mois": month_label,
                    "Métrique": metric_name,
                    "Valeur": month_metrics[key],
                }
            )
    monthly_chart = (
        alt.Chart(pd.DataFrame(monthly_rows))
        .mark_bar()
        .encode(
            x=alt.X("Mois:N", title=None, sort=MONTH_ORDER),
            xOffset="Métrique:N",
            y=alt.Y("Valeur:Q", title="Proportion", axis=alt.Axis(format="%")),
            color=alt.Color("Métrique:N", title=None),
            tooltip=["Mois:N", "Métrique:N", alt.Tooltip("Valeur:Q", format=".1%")],
        )
        .properties(height=320)
    )
    st.markdown("**Stabilité mensuelle du test**")
    st.altair_chart(monthly_chart, width="stretch")
    monthly_summary = " ; ".join(
        f"{MONTH_LABELS[int(month)]} : précision {_percent(values['precision'])}, "
        f"rappel {_percent(values['recall'])}, couverture "
        f"{_percent(values['alert_coverage'])}"
        for month, values in classification["test_by_month"].items()
    )
    st.caption(monthly_summary + ".")

    business = test
    f1_test = classification["f1_threshold_comparison"]["test"]
    comparison = pd.DataFrame(
        [
            {
                "Seuil": "Seuil métier",
                "Précision": business["precision"],
                "Rappel": business["recall"],
                "Couverture": business["alert_coverage"],
            },
            {
                "Seuil": "Seuil F1",
                "Précision": f1_test["precision"],
                "Rappel": f1_test["recall"],
                "Couverture": f1_test["alert_coverage"],
            },
        ]
    ).melt(id_vars="Seuil", var_name="Métrique", value_name="Valeur")
    comparison_chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("Métrique:N", title=None),
            xOffset="Seuil:N",
            y=alt.Y("Valeur:Q", title="Proportion", axis=alt.Axis(format="%")),
            color=alt.Color("Seuil:N", title=None),
            tooltip=["Seuil:N", "Métrique:N", alt.Tooltip("Valeur:Q", format=".1%")],
        )
        .properties(height=330)
    )
    st.markdown("**Pourquoi ne pas maximiser uniquement F1 ?**")
    st.altair_chart(comparison_chart, width="stretch")
    st.caption(
        f"Le seuil F1 signalerait {_percent(f1_test['alert_coverage'])} des vols "
        f"avec {_percent(f1_test['precision'])} de précision, contre "
        f"{_percent(business['alert_coverage'])} au seuil métier. Un seuil qui "
        "multiplie les alertes peut sembler meilleur en F1 tout en étant moins "
        "crédible pour les utilisateurs."
    )

    importance, importance_source = _load_importance(
        _candidate_mtime("models/**/feature_importance.csv")
    )
    top_importance = importance.nlargest(15, "importance").sort_values("importance")
    importance_chart = (
        alt.Chart(top_importance)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importance CatBoost"),
            y=alt.Y("feature:N", title=None, sort=None),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".3f"),
            ],
        )
        .properties(height=430)
    )
    st.markdown("**Features les plus importantes**")
    st.altair_chart(importance_chart, width="stretch")
    st.caption(f"Métriques : {metrics_source}. Importance : {importance_source}.")


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    return options.index(preferred) if preferred in options else fallback


def render_prediction(sample: pd.DataFrame, model_path: Path | None) -> None:
    st.title("Diagnostic d'un vol")
    st.caption(
        "Le formulaire utilise uniquement des informations prévues avant le départ. "
        "Il ne demande aucune cause de retard."
    )

    if model_path is None:
        st.warning(
            "Aucun artefact v6 n'est disponible sur cet ordinateur. Le dashboard "
            "reste consultable, mais une prédiction nécessite un entraînement local."
        )
        st.code(
            "uv run train-flight-models --input data/flight_data_2024.csv "
            "--sample-fraction 0.1",
            language="bash",
        )
    else:
        st.info(f"Artefact local détecté : `{model_path.relative_to(PROJECT_ROOT)}`")

    airport_states = airport_state_mapping(sample)
    airports = sorted(airport_states)
    carriers = sorted(sample["op_unique_carrier"].dropna().unique().tolist())

    with st.form("flight_prediction_form"):
        first, second, third = st.columns(3)
        with first:
            flight_date = st.date_input(
                "Date du vol",
                value=date(2025, 1, 5),
                format="DD/MM/YYYY",
                key="flight_date",
            )
            carrier = st.selectbox(
                "Code compagnie",
                carriers,
                index=_default_index(carriers, "AA"),
                key="carrier",
            )
            flight_number = st.number_input(
                "Numéro de vol",
                min_value=1,
                max_value=99_999,
                value=100,
                step=1,
                key="flight_number",
            )
        with second:
            origin = st.selectbox(
                "Aéroport de départ",
                airports,
                index=_default_index(airports, "JFK"),
                key="origin",
            )
            destination = st.selectbox(
                "Aéroport d'arrivée",
                airports,
                index=_default_index(airports, "LAX", fallback=1),
                key="destination",
            )
            distance = st.number_input(
                "Distance prévue (miles)",
                min_value=0.0,
                max_value=10_000.0,
                value=2_475.0,
                step=10.0,
                key="distance",
            )
        with third:
            scheduled_departure = st.time_input(
                "Heure locale de départ prévue",
                value=time(8, 30),
                step=300,
                key="scheduled_departure",
            )
            scheduled_arrival = st.time_input(
                "Heure locale d'arrivée prévue",
                value=time(11, 45),
                step=300,
                key="scheduled_arrival",
            )
            scheduled_duration = st.number_input(
                "Durée prévue publiée (minutes)",
                min_value=1.0,
                max_value=2_000.0,
                value=375.0,
                step=5.0,
                help="Elle ne peut pas être déduite des heures locales à cause des fuseaux.",
                key="scheduled_duration",
            )

        submitted = st.form_submit_button(
            "Produire le diagnostic",
            type="primary",
            width="stretch",
            disabled=model_path is None,
        )

    if submitted and model_path is not None:
        try:
            payload = build_flight_payload(
                flight_date=flight_date,
                carrier=carrier,
                flight_number=int(flight_number),
                origin=origin,
                origin_state=airport_states[origin],
                destination=destination,
                destination_state=airport_states[destination],
                scheduled_departure=scheduled_departure,
                scheduled_arrival=scheduled_arrival,
                scheduled_duration=scheduled_duration,
                distance=distance,
            )
            bundle = _load_bundle(str(model_path), model_path.stat().st_mtime_ns)
            result = predict_flight(bundle, payload)
            st.session_state["last_prediction"] = {
                "result": result,
                "payload": payload,
                "model_path": str(model_path),
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(f"Le diagnostic n'a pas pu être calculé : {error}")

    stored = st.session_state.get("last_prediction")
    if not stored or (model_path is not None and stored["model_path"] != str(model_path)):
        return

    presentation = prediction_presentation(stored["result"])
    st.subheader("Résultat")
    if presentation["publishable"]:
        if presentation["published_alert"]:
            st.error("Alerte : le modèle prévoit une arrivée en retard.")
        else:
            st.success("Le modèle ne déclenche pas d'alerte de retard.")
    else:
        st.warning(
            "Diagnostic expérimental uniquement : aucune décision « retard » ou "
            "« à l'heure » ne doit être publiée à partir de ce résultat."
        )
        for blocker in presentation["blockers"]:
            st.markdown(f"- {blocker}")

    result_columns = st.columns(3)
    result_columns[0].metric(
        "Probabilité diagnostique de retard",
        _percent(presentation["diagnostic_probability"]),
    )
    result_columns[1].metric(
        "Seuil du modèle",
        _percent(presentation["diagnostic_threshold"]),
    )
    result_columns[2].metric(
        "Classe interne non publiable",
        "Retard" if presentation["diagnostic_class"] else "À l'heure",
    )

    probability_frame = pd.DataFrame(
        [
            {"Valeur": "Probabilité", "Proportion": presentation["diagnostic_probability"]},
            {"Valeur": "Seuil", "Proportion": presentation["diagnostic_threshold"]},
        ]
    )
    probability_chart = (
        alt.Chart(probability_frame)
        .mark_bar()
        .encode(
            x=alt.X("Proportion:Q", title="Proportion", axis=alt.Axis(format="%")),
            y=alt.Y("Valeur:N", title=None),
            tooltip=["Valeur:N", alt.Tooltip("Proportion:Q", format=".1%")],
        )
        .properties(height=150)
    )
    st.altair_chart(probability_chart, width="stretch")

    context_labels = {
        "provided_daily_schedule": "planning journalier exact",
        "typical_schedule_profile": "profil de planning typique",
        "typical_schedule_profile_with_overrides": "profil typique complété",
        "missing": "planning indisponible",
    }
    st.caption(
        "Contexte historique arrêté au "
        f"{presentation['historical_context_date'] or 'jour inconnu'} ; planning : "
        f"{context_labels.get(presentation['schedule_context_source'], 'inconnu')}."
    )


sample = _load_sample(str(SAMPLE_PATH), SAMPLE_PATH.stat().st_mtime_ns)
metrics, metrics_source = _load_metrics(
    _candidate_mtime("models/**/training_metrics.json")
)
model_path = find_model_path(PROJECT_ROOT)
model_load_error = None
if model_path is not None:
    try:
        _load_bundle(str(model_path), model_path.stat().st_mtime_ns)
    except Exception as error:  # frontière UI : un ancien artefact ne doit pas bloquer l'app
        model_load_error = str(error)
        model_path = None

with st.sidebar:
    st.header("Navigation")
    selected_page = st.radio(
        "Choisir une page",
        NAVIGATION,
        label_visibility="collapsed",
        key="navigation",
    )
    st.divider()
    st.caption("Cible : `arr_delay > 0`")
    st.caption("Données Spark : 10 000 vols")
    st.caption("Données ML : 10 % du CSV complet")
    if model_load_error:
        st.warning("Artefact local incompatible avec la version v6")
    elif model_path is None:
        st.warning("Modèle local absent")
    else:
        st.success("Artefact local détecté")

if selected_page == "Vue d'ensemble":
    render_overview(sample, metrics, metrics_source)
elif selected_page == "Explorer les vols":
    render_explorer(sample)
elif selected_page == "Performance du modèle":
    render_performance(metrics, metrics_source)
else:
    render_prediction(sample, model_path)
