"""
tests/test_colab_augment_notebook.py
Verifies that 04_colab_dentex_augment_run_all.ipynb is structurally valid,
syntactically clean, and executes without unhandled errors.
"""

import ast
import json
import shutil
from pathlib import Path
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


def test_notebook_structure_and_schema():
    nb_path = Path("notebooks/04_colab_dentex_augment_run_all.ipynb")
    assert nb_path.exists(), "Notebook 04_colab_dentex_augment_run_all.ipynb does not exist!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, "Notebook missing 'cells' key"
    assert "metadata" in nb, "Notebook missing 'metadata' key"
    assert nb.get("nbformat") == 4, f"Unexpected nbformat: {nb.get('nbformat')}"
    assert len(nb["cells"]) >= 10, f"Expected at least 10 cells, got {len(nb['cells'])}"

    for idx, cell in enumerate(nb["cells"]):
        assert "cell_type" in cell, f"Cell {idx} missing cell_type"
        assert "source" in cell, f"Cell {idx} missing source"
        assert len(cell["source"]) > 0, f"Cell {idx} is empty"


def test_notebook_code_cells_ast_syntax():
    nb_path = Path("notebooks/04_colab_dentex_augment_run_all.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Cell {idx} failed Python syntax check: {e}")


def test_headless_execution_colab_notebook(tmp_path):
    """
    Executes notebook headlessly with mock data to guarantee zero runtime crashes.
    """
    nb_path = Path("notebooks/04_colab_dentex_augment_run_all.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Set up temporary test drive directory
    mock_drive_dir = tmp_path / "mock_drive"
    mock_output_dir = tmp_path / "mock_drive_output"
    mock_drive_dir.mkdir(parents=True, exist_ok=True)
    mock_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy local sample data into mock drive
    if Path("data/raw").exists():
        shutil.copytree("data/raw", mock_drive_dir / "raw", dirs_exist_ok=True)
    if Path("data/processed").exists():
        shutil.copytree("data/processed", mock_drive_dir / "processed", dirs_exist_ok=True)

    # Modify Cell 0 in the executed notebook to point to tmp_path and use 1 epoch
    for cell in nb.cells:
        if cell.cell_type == "code" and "DRIVE_DATA_DIR" in "".join(cell.source):
            lines = cell.source
            new_lines = []
            for l in lines:
                if l.startswith("DRIVE_DATA_DIR ="):
                    new_lines.append(f"DRIVE_DATA_DIR = r'{mock_drive_dir}'\n")
                elif l.startswith("DRIVE_OUTPUT_DIR ="):
                    new_lines.append(f"DRIVE_OUTPUT_DIR = r'{mock_output_dir}'\n")
                elif l.startswith("EPOCHS ="):
                    new_lines.append("EPOCHS = 1\n")
                elif l.startswith("NUM_AUGMENT_PER_IMAGE ="):
                    new_lines.append("NUM_AUGMENT_PER_IMAGE = 1\n")
                else:
                    new_lines.append(l)
            cell.source = "".join(new_lines)
            break

    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "."}})

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    pytest.fail(f"Cell {idx} execution failed: {out.get('ename')}: {out.get('evalue')}")
