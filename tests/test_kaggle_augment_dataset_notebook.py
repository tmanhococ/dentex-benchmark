"""
tests/test_kaggle_augment_dataset_notebook.py
Verifies 06_kaggle_dentex_augment_dataset_export.ipynb: it reuses notebook 05's data/augment cells
verbatim, trains nothing, and on mock DENTEX data writes a complete, self-consistent dataset.
"""

import ast
import json
import shutil
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NB_PATH = Path("notebooks/06_kaggle_dentex_augment_dataset_export.ipynb")
NB05_PATH = Path("notebooks/05_kaggle_dentex_augment_run_all.ipynb")


def _code(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_reuses_notebook05_augment_cells_and_never_trains():
    code06, code05 = _code(NB_PATH), _code(NB05_PATH)
    for src in code06:
        ast.parse(src)
    for marker in ["convert_dentex_annotations_to_coco", "build_tooth_bank(", "run_offline(aug_cfg", "qc_before_after.png"]:
        cell05 = next(c for c in code05 if marker in c)
        assert cell05 in code06, f"cell with {marker} differs from notebook 05"
    joined = "\n".join(code06)
    assert "scripts/train.py" not in joined and "scripts/evaluate.py" not in joined


def test_headless_export_writes_complete_dataset(tmp_path):
    sys.path.insert(0, str(Path("tests/augment").resolve()))
    from test_real_swap_pipeline import _make_dentex_dataset

    repo = tmp_path / "repo"
    for d in ["dentex_bench", "scripts", "configs"]:
        shutil.copytree(d, repo / d, ignore=shutil.ignore_patterns("__pycache__"))
    kaggle_input = tmp_path / "input"
    _make_dentex_dataset(kaggle_input / "dentex" / "quadrant-enumeration-disease", n_images=10)
    work = tmp_path / "working"

    nb = nbformat.read(NB_PATH, as_version=4)
    overrides = {"DATA_ROOT =": f"DATA_ROOT = r'{kaggle_input}'", "OUTPUT_ROOT =": f"OUTPUT_ROOT = r'{work}'",
                 "QC_PAIRS =": "QC_PAIRS = 2"}
    cfg_cell = next(c for c in nb.cells if c.cell_type == "code" and "DATA_ROOT =" in c.source)
    cfg_cell.source = "\n".join(next((overrides[k] for k in overrides if l.startswith(k)), l) for l in cfg_cell.source.splitlines())
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(repo)}})

    out = work / "dentex_v2_augmented"
    ann = {n: json.loads((out / "annotations" / f"instances_{n}.json").read_text()) for n in ["train", "train_aug", "val"]}
    on_disk = {p.name for p in (out / "images").rglob("*.png")}
    for c in ann.values():
        assert {im["file_name"] for im in c["images"]} <= on_disk
    syn = [im for im in ann["train_aug"]["images"] if im["file_name"].startswith("syn_")]
    assert syn and len(list((out / "images" / "synthetic").glob("syn_*.png"))) == len(syn)
    assert len(ann["train_aug"]["images"]) == len(ann["train"]["images"]) + len(syn)
    assert not ({im["file_name"] for im in ann["train"]["images"]} & {im["file_name"] for im in ann["val"]["images"]})
    assert (out / "tooth_bank").is_dir() and any((out / "tooth_bank").rglob("*.png"))
    assert (out / "augment_state" / "recipes.jsonl").is_file()
    for f in ["README.md", "dataset_info.json", "qc_before_after.png", "bank_report.json"]:
        assert (out / f).is_file(), f
    assert not (out / "checkpoints").exists() and not (out / "augmented_data").exists()
    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["synthetic_images"] == len(syn)
