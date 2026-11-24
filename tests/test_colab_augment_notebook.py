"""
tests/test_colab_augment_notebook.py
Verifies that 04_colab_dentex_augment_run_all.ipynb is structurally valid, syntactically clean,
and runs end to end (Run all) on a DENTEX-format mock dataset in a throwaway copy of the repo.
"""

import ast
import json
import shutil
import sys
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NB_PATH = Path("notebooks/04_colab_dentex_augment_run_all.ipynb")


def test_notebook_structure_and_schema():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    assert len(nb["cells"]) >= 10
    for idx, cell in enumerate(nb["cells"]):
        assert cell.get("source"), f"Cell {idx} is empty"


def test_notebook_code_cells_ast_syntax():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            try:
                ast.parse("".join(cell["source"]))
            except SyntaxError as e:
                pytest.fail(f"Cell {idx} failed Python syntax check: {e}")


def test_notebook_never_guesses_fdi_or_fakes_patches():
    src = NB_PATH.read_text(encoding="utf-8")
    assert "infer_fdi" not in src
    assert "dummy" not in src.lower()


def test_headless_execution_colab_notebook(tmp_path):
    """Run all on mock DENTEX data: 1 model x (baseline, augmented), 1 epoch."""
    sys.path.insert(0, str(Path("tests/augment").resolve()))
    from test_real_swap_pipeline import _make_dentex_dataset

    repo = tmp_path / "repo"
    for d in ["dentex_bench", "scripts", "configs"]:
        shutil.copytree(d, repo / d, ignore=shutil.ignore_patterns("__pycache__"))
    drive = tmp_path / "drive" / "DENTEX" / "training_data" / "quadrant-enumeration-disease"
    _make_dentex_dataset(drive, n_images=10)
    out_dir = tmp_path / "drive_out"

    nb = nbformat.read(NB_PATH, as_version=4)
    overrides = {
        "DRIVE_DATA_DIR =": f"DRIVE_DATA_DIR = r'{tmp_path / 'drive' / 'DENTEX'}'",
        "DRIVE_OUTPUT_DIR =": f"DRIVE_OUTPUT_DIR = r'{out_dir}'",
        "MODELS =": "MODELS = ['faster_rcnn']",
        "EPOCHS =": "EPOCHS = 1",
        "NUM_AUGMENT_PER_IMAGE =": "NUM_AUGMENT_PER_IMAGE = 1",
        "QC_PAIRS =": "QC_PAIRS = 2",
    }
    cfg_cell = next(c for c in nb.cells if c.cell_type == "code" and "DRIVE_DATA_DIR =" in c.source)
    lines = []
    for line in cfg_cell.source.splitlines():
        key = next((k for k in overrides if line.startswith(k)), None)
        lines.append(overrides[key] if key else line)
    cfg_cell.source = "\n".join(lines)

    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(repo)}})

    run_out = out_dir / "v2_realswap"
    status = json.loads((run_out / "train_status.json").read_text())
    assert status == {"baseline/faster_rcnn": "done", "augmented/faster_rcnn": "done"}
    with open(run_out / "results_table.csv", encoding="utf-8") as f:
        assert len(f.read().strip().splitlines()) == 3  # header + 2 runs
    assert (run_out / "qc_before_after.png").is_file()
    assert (run_out / "checkpoints" / "augmented" / "faster_rcnn" / "last.pth").is_file()
