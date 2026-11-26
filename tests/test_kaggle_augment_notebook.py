"""
tests/test_kaggle_augment_notebook.py
Verifies 05_kaggle_dentex_augment_run_all.ipynb: valid structure and syntax, and Run all end to end on
DENTEX-format mock data laid out like /kaggle/input, then a second "version" that gets the first
version's output attached as input and resumes training from its checkpoints.
"""

import ast
import json
import shutil
import sys
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NB_PATH = Path("notebooks/05_kaggle_dentex_augment_run_all.ipynb")


def test_notebook_structure_and_syntax():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    assert len(nb["cells"]) >= 10
    for idx, cell in enumerate(nb["cells"]):
        assert cell.get("source"), f"Cell {idx} is empty"
        if cell["cell_type"] == "code":
            try:
                ast.parse("".join(cell["source"]))
            except SyntaxError as e:
                pytest.fail(f"Cell {idx} failed Python syntax check: {e}")
    src = NB_PATH.read_text(encoding="utf-8")
    assert "infer_fdi" not in src and "dummy" not in src.lower()
    assert "/kaggle/input" in src and "--time-limit-min" in src


def _run(nb_template, repo, overrides):
    nb = nbformat.from_dict(json.loads(json.dumps(nb_template)))
    cfg_cell = next(c for c in nb.cells if c.cell_type == "code" and "DATA_ROOT =" in c.source)
    lines = []
    for line in cfg_cell.source.splitlines():
        key = next((k for k in overrides if line.startswith(k)), None)
        lines.append(overrides[key] if key else line)
    cfg_cell.source = "\n".join(lines)
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(repo)}})
    return nb


def test_headless_run_then_resume_in_next_version(tmp_path):
    sys.path.insert(0, str(Path("tests/augment").resolve()))
    from test_real_swap_pipeline import _make_dentex_dataset

    repo = tmp_path / "repo"
    for d in ["dentex_bench", "scripts", "configs"]:
        shutil.copytree(d, repo / d, ignore=shutil.ignore_patterns("__pycache__"))
    kaggle_input = tmp_path / "input"
    _make_dentex_dataset(kaggle_input / "dentex-challenge-2023" / "training_data" / "quadrant-enumeration-disease", n_images=10)
    nb_template = nbformat.read(NB_PATH, as_version=4)
    common = {
        "DATA_ROOT =": f"DATA_ROOT = r'{kaggle_input}'",
        "MODELS =": "MODELS = ['faster_rcnn']",
        "NUM_AUGMENT_PER_IMAGE =": "NUM_AUGMENT_PER_IMAGE = 1",
        "QC_PAIRS =": "QC_PAIRS = 2",
    }

    # Version 1: 1 epoch
    work1 = tmp_path / "working1"
    _run(nb_template, repo, {**common, "OUTPUT_ROOT =": f"OUTPUT_ROOT = r'{work1}'", "EPOCHS =": "EPOCHS = 1"})
    out1 = work1 / "v2_realswap"
    assert json.loads((out1 / "train_status.json").read_text()) == {
        "baseline/faster_rcnn": "done", "augmented/faster_rcnn": "done"}
    assert len((out1 / "results_table.csv").read_text().strip().splitlines()) == 3
    assert (out1 / "qc_before_after.png").is_file()
    assert (work1 / "results_small.zip").is_file()

    # Version 2: previous output attached under /kaggle/input, EPOCHS raised to 2 -> resumes at epoch 2
    shutil.copytree(work1, kaggle_input / "my-notebook-output")
    work2 = tmp_path / "working2"
    _run(nb_template, repo, {**common, "OUTPUT_ROOT =": f"OUTPUT_ROOT = r'{work2}'", "EPOCHS =": "EPOCHS = 2"})
    out2 = work2 / "v2_realswap"
    for ds in ("baseline", "augmented"):
        hist = json.loads((out2 / "checkpoints" / ds / "faster_rcnn" / "history.json").read_text())
        assert [h["epoch"] for h in hist] == [1, 2]
        assert all("seconds" in h for h in hist)
    log = (out2 / "logs" / "train_baseline_faster_rcnn.log").read_text(encoding="utf-8")
    assert "Resumed from" in log
