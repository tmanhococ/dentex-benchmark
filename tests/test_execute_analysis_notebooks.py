import json
from pathlib import Path
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.parametrize("nb_name", [
    "dentex_eda_analysis.ipynb",
    "dentex_inference_error_analysis.ipynb"
])
def test_headless_execution_of_analysis_notebooks(nb_name, tmp_path):
    nb_path = Path("notebooks") / nb_name
    assert nb_path.exists()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    # Execute in repository root
    ep.preprocess(nb, {"metadata": {"path": "."}})

    # Verify no cell contains an unhandled exception output
    for cell in nb.cells:
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    pytest.fail(f"Cell execution failed in {nb_name}: {out.get('ename')}: {out.get('evalue')}")
