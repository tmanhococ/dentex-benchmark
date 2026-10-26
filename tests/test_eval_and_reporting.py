# tests/test_eval_and_reporting.py
import json
import sys
from pathlib import Path
import pytest
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.eval.coco_eval import evaluate_coco_predictions, evaluate_coco
from dentex_bench.eval.reporter import (
    format_benchmark_markdown,
    format_benchmark_latex,
    save_reports,
    generate_markdown_report,
    generate_latex_table,
)


@pytest.fixture
def mock_results():
    return {
        "Faster R-CNN": {
            "mAP50": 0.652, "mAP50_95": 0.415,
            "AP_Caries": 0.680, "AP_DeepCaries": 0.590,
            "AP_PeriapicalLesion": 0.640, "AP_ImpactedTooth": 0.700,
            "latency_ms": 32.5, "peak_vram_mb": 1850
        },
        "DINO": {
            "mAP50": 0.710, "mAP50_95": 0.468,
            "AP_Caries": 0.725, "AP_DeepCaries": 0.650,
            "AP_PeriapicalLesion": 0.690, "AP_ImpactedTooth": 0.775,
            "latency_ms": 48.0, "peak_vram_mb": 2400
        },
        "HierarchicalDet": {
            "mAP50": 0.724, "mAP50_95": 0.481,
            "AP_Caries": 0.740, "AP_DeepCaries": 0.672,
            "AP_PeriapicalLesion": 0.710, "AP_ImpactedTooth": 0.775,
            "latency_ms": 65.0, "peak_vram_mb": 2900
        }
    }


def test_reporter_formatting(mock_results):
    md_str = format_benchmark_markdown(mock_results)
    header_line = md_str.splitlines()[0]
    for col in ["Model", "mAP50", "mAP50:95", "Caries", "Deep Caries", "Periapical", "Impacted", "Latency (ms)", "Peak VRAM (MB)"]:
        assert col in header_line
    assert header_line.startswith("|") and header_line.endswith("|")
    assert "HierarchicalDet" in md_str
    assert "Faster R-CNN" in md_str
    assert "DINO" in md_str

    tex_str = format_benchmark_latex(mock_results)
    assert "\\begin{table}" in tex_str
    assert "\\end{table}" in tex_str
    assert "DENTEX 4-Class" in tex_str


def test_save_reports_and_helpers(mock_results, tmp_path):
    output_dir = tmp_path / "reports"
    save_reports(mock_results, output_dir)

    md_file = output_dir / "benchmark_summary.md"
    tex_file = output_dir / "benchmark_table.tex"

    assert md_file.exists()
    assert tex_file.exists()
    assert "HierarchicalDet" in md_file.read_text(encoding="utf-8")
    assert "\\begin{table}" in tex_file.read_text(encoding="utf-8")

    # Test single file generator helpers
    custom_md = tmp_path / "custom.md"
    generate_markdown_report(mock_results, custom_md)
    assert custom_md.exists()
    assert "HierarchicalDet" in custom_md.read_text(encoding="utf-8")

    custom_tex = tmp_path / "custom.tex"
    generate_latex_table(mock_results, custom_tex)
    assert custom_tex.exists()
    assert "\\begin{table}" in custom_tex.read_text(encoding="utf-8")


def test_evaluate_coco_empty_predictions(tmp_path):
    # Dummy GT file
    gt_file = tmp_path / "empty_gt.json"
    coco_data = {
        "images": [{"id": 1, "file_name": "img1.png", "width": 100, "height": 100}],
        "categories": [
            {"id": 1, "name": "Caries"},
            {"id": 2, "name": "Deep Caries"},
            {"id": 3, "name": "Periapical Lesion"},
            {"id": 4, "name": "Impacted Tooth"}
        ],
        "annotations": []
    }
    gt_file.write_text(json.dumps(coco_data), encoding="utf-8")

    # Empty prediction list
    metrics = evaluate_coco_predictions(gt_file, [])
    assert metrics["mAP50"] == 0.0
    assert metrics["mAP50_95"] == 0.0
    assert metrics["AP_Caries"] == 0.0

    # Predictions with 0 boxes
    empty_pred = [{
        "image_id": 1,
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
        "labels": torch.zeros((0,), dtype=torch.int64)
    }]
    metrics2 = evaluate_coco(gt_file, empty_pred)
    assert metrics2["mAP50"] == 0.0
    assert metrics2["mAP50_95"] == 0.0


def test_evaluate_coco_predictions_with_synthetic_coco(tmp_path):
    gt_file = tmp_path / "test_gt.json"
    coco_data = {
        "images": [
            {"id": 1, "file_name": "img1.png", "width": 500, "height": 500},
            {"id": 2, "file_name": "img2.png", "width": 500, "height": 500}
        ],
        "categories": [
            {"id": 1, "name": "Caries"},
            {"id": 2, "name": "Deep Caries"},
            {"id": 3, "name": "Periapical Lesion"},
            {"id": 4, "name": "Impacted Tooth"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50.0, 50.0, 40.0, 40.0], "area": 1600.0, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 4, "bbox": [100.0, 100.0, 50.0, 50.0], "area": 2500.0, "iscrowd": 0}
        ]
    }
    gt_file.write_text(json.dumps(coco_data), encoding="utf-8")

    # Perfect predictions [x1, y1, x2, y2]
    predictions = [
        {
            "image_id": 1,
            "boxes": torch.tensor([[50.0, 50.0, 90.0, 90.0]]),
            "scores": torch.tensor([0.98]),
            "labels": torch.tensor([1])
        },
        {
            "image_id": 2,
            "boxes": torch.tensor([[100.0, 100.0, 150.0, 150.0]]),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([4])
        }
    ]

    metrics = evaluate_coco_predictions(gt_file, predictions)
    assert "mAP50" in metrics
    assert "mAP50_95" in metrics
    assert "AP_Caries" in metrics
    assert "AP_DeepCaries" in metrics
    assert "AP_PeriapicalLesion" in metrics
    assert "AP_ImpactedTooth" in metrics
    assert metrics["mAP50"] > 0.0
    assert metrics["AP_Caries"] > 0.0
    assert metrics["AP_ImpactedTooth"] > 0.0


def test_evaluate_script_cli(tmp_path):
    import subprocess
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--output_dir", str(tmp_path / "cli_out")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert (tmp_path / "cli_out" / "benchmark_summary.md").exists()
    assert (tmp_path / "cli_out" / "benchmark_table.tex").exists()

    # Test with --compare_all flag
    cmd_compare = [
        sys.executable,
        "scripts/evaluate.py",
        "--compare_all",
        "--output_dir", str(tmp_path / "cli_compare_out")
    ]
    res_compare = subprocess.run(cmd_compare, capture_output=True, text=True)
    assert res_compare.returncode == 0
    assert (tmp_path / "cli_compare_out" / "benchmark_summary.md").exists()

    # Test with --model flag
    cmd_model = [
        sys.executable,
        "scripts/evaluate.py",
        "--model", "faster_rcnn",
        "--output_dir", str(tmp_path / "cli_model_out")
    ]
    res_model = subprocess.run(cmd_model, capture_output=True, text=True)
    assert res_model.returncode == 0
    assert (tmp_path / "cli_model_out" / "benchmark_summary.md").exists()


def test_evaluate_script_dynamic_checkpoint(tmp_path):
    import subprocess
    from dentex_bench.config import BenchmarkConfig
    from dentex_bench.models.dino import DINORunner

    # Create dummy checkpoint
    cfg = BenchmarkConfig.from_auto_env()
    runner = DINORunner(cfg)
    runner.build_model()
    ckpt_path = tmp_path / "dino_checkpoint.pth"
    runner.save_checkpoint(ckpt_path, extra_info={"epoch": 1, "loss": 0.5})
    assert ckpt_path.exists()

    # Create dummy val annotation
    val_ann_path = tmp_path / "val_ann.json"
    coco_data = {
        "images": [{"id": 1, "file_name": "xray1.png", "width": 500, "height": 500}],
        "categories": [
            {"id": 1, "name": "Caries"},
            {"id": 2, "name": "Deep Caries"},
            {"id": 3, "name": "Periapical Lesion"},
            {"id": 4, "name": "Impacted Tooth"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 50.0, 50.0], "area": 2500.0, "iscrowd": 0}
        ]
    }
    val_ann_path.write_text(json.dumps(coco_data), encoding="utf-8")

    out_dir = tmp_path / "dynamic_out"
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--model", "dino",
        "--checkpoint", str(ckpt_path),
        "--val_ann", str(val_ann_path),
        "--output_dir", str(out_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}\nOutput: {res.stdout}"
    assert (out_dir / "benchmark_summary.md").exists()
    assert (out_dir / "benchmark_table.tex").exists()
    content = (out_dir / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "DINO" in content



