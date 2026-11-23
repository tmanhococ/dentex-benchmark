# tests/augment/test_qc.py
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from dentex_bench.augment.qc import compute_seam_score, save_debug_composite
from scripts.qc_report import generate_report


def test_seam_score_perfect_blend():
    # Uniform image has gradient 0 everywhere -> seam score ~ 0.0 (or ~1.0 with eps)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    score = compute_seam_score(img, mask)
    assert abs(score - 0.0) < 0.1 or abs(score - 1.0) < 0.5


def test_seam_score_high_contrast_hard_boundary():
    # Hard square with strong gradient at boundary
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    score = compute_seam_score(img, mask)
    # Harsh boundary should yield seam score > 2.0
    assert score > 2.0


def test_seam_score_empty_and_edge_masks():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Empty mask
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    score_empty = compute_seam_score(img, empty_mask)
    assert isinstance(score_empty, float)
    assert score_empty == 0.0

    # Full mask
    full_mask = np.full((100, 100), 255, dtype=np.uint8)
    score_full = compute_seam_score(img, full_mask)
    assert isinstance(score_full, float)


def test_save_debug_composite(tmp_path):
    orig = np.full((100, 120, 3), 50, dtype=np.uint8)
    syn = np.full((100, 120, 3), 150, dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    new_boxes = [[20.0, 20.0, 40.0, 40.0]]

    out_path = tmp_path / "debug_composite.png"
    save_debug_composite(orig, syn, mask, new_boxes, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Test with 2D grayscale images
    orig_gray = np.full((100, 120), 50, dtype=np.uint8)
    syn_gray = np.full((100, 120), 150, dtype=np.uint8)
    out_path_gray = tmp_path / "debug_gray.png"
    save_debug_composite(orig_gray, syn_gray, mask, [{"bbox": [20, 20, 40, 40]}], out_path_gray)
    assert out_path_gray.exists()


def test_qc_report_generation(tmp_path):
    run_dir = tmp_path / "run_test_01"
    run_dir.mkdir(parents=True)
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True)

    manifest_data = {
        "run_name": "run_test_01",
        "status": "complete",
        "progress": {"counts": {"recipes": 2, "ops": 3, "rejected_ops": 5}},
    }
    with open(state_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    recipes = [
        {
            "recipe_id": "rec_1_0",
            "target_image_id": 1,
            "ops": [
                {"op": "swap", "blend_mode": "hard", "seam_score": 2.8},
                {"op": "swap", "blend_mode": "core_preserving", "seam_score": 1.1},
            ],
            "rejections": [
                {"code": "R_OVERLAP", "tries": 2},
                {"code": "R_SEAM", "tries": 1},
                {"code": "R_OVERLAP", "tries": 3},
            ],
        },
        {
            "recipe_id": "rec_2_0",
            "target_image_id": 2,
            "ops": [
                {"op": "swap", "blend_mode": "core_preserving", "seam_score": 1.3},
            ],
            "rejections": [
                {"code": "R_SCALE", "tries": 1},
                {"code": "R_SEAM", "tries": 2},
            ],
        },
    ]
    with open(state_dir / "recipes.jsonl", "w", encoding="utf-8") as f:
        for r in recipes:
            f.write(json.dumps(r) + "\n")

    generate_report(run_dir)

    summary_file = run_dir / "qc" / "summary.json"
    assert summary_file.exists()

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "success"
    assert summary["rejected_by_reason"]["R_OVERLAP"] == 2
    assert summary["rejected_by_reason"]["R_SEAM"] == 2
    assert summary["rejected_by_reason"]["R_SCALE"] == 1
    assert abs(summary["seam_scores_mean"] - ((2.8 + 1.1 + 1.3) / 3)) < 1e-3


def test_qc_report_cli(tmp_path):
    run_dir = tmp_path / "run_cli_test"
    run_dir.mkdir(parents=True)
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True)

    manifest_data = {"status": "complete"}
    with open(state_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
    with open(state_dir / "recipes.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"recipe_id": "r1", "ops": [], "rejections": []}) + "\n")

    cmd = [sys.executable, "scripts/qc_report.py", "--run", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert (run_dir / "qc" / "summary.json").exists()
