from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_streamlit_app_pages_render_without_exception():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert app.title[0].value == "Prédiction des retards des vols domestiques US"

    app.sidebar.radio[0].set_value("Explorer les vols")
    app.run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "Explorer les 10 000 vols Spark"

    app.sidebar.radio[0].set_value("Performance du modèle")
    app.run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "Performance du classifieur"

    app.sidebar.radio[0].set_value("Diagnostic d'un vol")
    app.run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "Diagnostic d'un vol"


def test_streamlit_prediction_form_runs_with_the_versioned_v6_model():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    app.sidebar.radio[0].set_value("Diagnostic d'un vol")
    app.run(timeout=60)

    submit_button = next(
        button for button in app.button if button.label == "Produire le diagnostic"
    )
    assert submit_button.disabled is False
    submit_button.click()
    app.run(timeout=60)

    assert not app.exception
    assert any(header.value == "Résultat" for header in app.subheader)
    assert any(
        metric.label == "Probabilité diagnostique de retard"
        for metric in app.metric
    )
    assert any("expérimental" in warning.value for warning in app.warning)
