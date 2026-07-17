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


def test_notebook_code_cells_have_valid_python_syntax() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cellule_{index}", "exec")
