from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/flight_delay_pipeline.ipynb")


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_contains_the_four_project_steps() -> None:
    notebook = _load_notebook()
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "# Étape 1 — Parsing avec PySpark" in markdown
    assert "# Étape 2 — Analyse avec PySpark" in markdown
    assert "# Étape 3 — Machine learning avec Python" in markdown
    assert "# Étape 4 — Visualisation et prédiction" in markdown


def test_notebook_reuses_the_tested_pipeline() -> None:
    notebook = _load_notebook()
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "parse_and_validate" in code
    assert "write_analysis" in code
    assert "train_models" in code
    assert "predict_flight" in code
    assert "SPARK_SAMPLE_SIZE = 10_000" in code
    assert "USE_FULL_ML_DATA = True" in code
    assert "ML_SAMPLE_FRACTION = 0.1 if USE_FULL_ML_DATA else 1.0" in code


def test_notebook_code_cells_have_valid_python_syntax() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cellule_{index}", "exec")


def test_notebook_exposes_the_business_acceptance_gate() -> None:
    notebook = _load_notebook()
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "50 % de précision" in markdown
    assert "20 % des retards" in markdown
    assert "5 à 10 %" in markdown
    assert "business_gate" in code
    assert "alert_coverage" in code
    assert "prediction_publishable" in code
    assert "Aucune alerte publiable" in code
