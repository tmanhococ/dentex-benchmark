"""
Regression tests for the integrity defects found in the first Colab run:
- evaluate.py reported hardcoded metrics instead of evaluating checkpoints
- missing images were silently replaced by black canvases
- same-named files from other DENTEX subsets could be paired with the wrong annotations
- evaluation compared boxes in resized coordinates against original-resolution ground truth
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.data.dataset import DentexDataset
from dentex_bench.eval.coco_eval import evaluate_runner

CATEGORIES = [
    {"id": 1, "name": "Caries"},
    {"id": 2, "name": "Deep Caries"},
    {"id": 3, "name": "Periapical Lesion"},
    {"id": 4, "name": "Impacted Tooth"},
]


def _write_coco(path: Path, images, annotations):
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}), encoding="utf-8")


EVALUATE_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py")


@pytest.mark.parametrize("flags", [["--compare_all"], ["--model", "hierarchical_net"], []])
def test_evaluate_without_checkpoints_fails_and_reports_nothing(tmp_path, flags):
    # Run from an empty workspace: there is no checkpoint, so there must be no metric
    res = subprocess.run([sys.executable, EVALUATE_SCRIPT, *flags], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode != 0
    assert not (tmp_path / "outputs" / "benchmark_summary.md").exists()


def test_dataset_raises_on_missing_image(tmp_path):
    ann = tmp_path / "ann.json"
    _write_coco(ann, [{"id": 1, "file_name": "absent.png", "width": 64, "height": 32}], [])
    ds = DentexDataset(img_dir=tmp_path, ann_file=ann)
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_dataset_picks_same_named_file_matching_annotation_size(tmp_path):
    # DENTEX subsets reuse names such as train_1.png for different radiographs
    (tmp_path / "quadrant" / "xrays").mkdir(parents=True)
    (tmp_path / "quadrant_enumeration_disease" / "xrays").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "quadrant" / "xrays" / "train_1.png"), np.full((50, 70, 3), 10, np.uint8))
    cv2.imwrite(str(tmp_path / "quadrant_enumeration_disease" / "xrays" / "train_1.png"), np.full((32, 64, 3), 200, np.uint8))
    ann = tmp_path / "ann.json"
    _write_coco(ann, [{"id": 1, "file_name": "train_1.png", "width": 64, "height": 32}], [])

    img, _ = DentexDataset(img_dir=tmp_path, ann_file=ann, resize=False)[0]
    assert tuple(img.shape[-2:]) == (32, 64)
    assert img.mean().item() == pytest.approx(200 / 255, abs=1e-3)


def test_dataset_raises_when_no_candidate_matches_annotation_size(tmp_path):
    cv2.imwrite(str(tmp_path / "train_1.png"), np.zeros((50, 70, 3), np.uint8))
    ann = tmp_path / "ann.json"
    _write_coco(ann, [{"id": 1, "file_name": "train_1.png", "width": 64, "height": 32}], [])
    with pytest.raises(ValueError):
        DentexDataset(img_dir=tmp_path, ann_file=ann)[0]


def test_loading_checkpoint_from_other_architecture_is_rejected(tmp_path):
    from dentex_bench.config import BenchmarkConfig
    from dentex_bench.models import HierarchicalNetRunner

    cfg = BenchmarkConfig.from_auto_env()
    runner = HierarchicalNetRunner(cfg)
    runner.build_model(pretrained=False)
    stale = tmp_path / "stale.pth"
    # Checkpoints written by the pre-fix global-pool models carry no architecture id
    torch.save({"model_state_dict": runner.model.state_dict()}, stale)
    with pytest.raises(ValueError, match="retrain"):
        runner.load_checkpoint(stale)


class _OracleRunner:
    """Returns the ground-truth boxes, in the coordinates of whatever image it receives."""

    def __init__(self, coco):
        self.items = []
        for im in coco["images"]:
            anns = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
            self.items.append((im, anns))
        self.cursor = 0

    def predict(self, images):
        out = []
        for img in images:
            im, anns = self.items[self.cursor]
            self.cursor += 1
            sx = img.shape[-1] / im["width"]
            sy = img.shape[-2] / im["height"]
            boxes = [[x * sx, y * sy, (x + w) * sx, (y + h) * sy] for x, y, w, h in (a["bbox"] for a in anns)]
            out.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor([a["category_id"] for a in anns]),
                "scores": torch.ones(len(anns)),
            })
        return out


def test_evaluate_runner_scores_perfect_predictions_at_original_resolution(tmp_path):
    cv2.imwrite(str(tmp_path / "scan.png"), np.zeros((1300, 2800, 3), np.uint8))
    images = [{"id": 1, "file_name": "scan.png", "width": 2800, "height": 1300}]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [400.0, 300.0, 200.0, 250.0], "area": 50000.0, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 4, "bbox": [2000.0, 700.0, 300.0, 280.0], "area": 84000.0, "iscrowd": 0},
    ]
    ann = tmp_path / "val.json"
    _write_coco(ann, images, anns)

    metrics = evaluate_runner(_OracleRunner(json.loads(ann.read_text())), ann, tmp_path)
    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["mAP50_95"] == pytest.approx(1.0)


def test_parser_rejects_duplicate_image_ids(tmp_path):
    from dentex_bench.data.parser import convert_dentex_annotations_to_coco

    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10},
                   {"id": 1, "file_name": "b.png", "width": 10, "height": 10}],
        "annotations": [], "categories": CATEGORIES,
    }), encoding="utf-8")
    # A duplicated id would put two radiographs under one key and could straddle train/val
    with pytest.raises(ValueError, match="duplicate"):
        convert_dentex_annotations_to_coco(raw, tmp_path / "t.json", tmp_path / "v.json")


def test_class_without_ground_truth_has_undefined_ap(tmp_path):
    from dentex_bench.eval.coco_eval import evaluate_coco_predictions

    ann = tmp_path / "val.json"
    _write_coco(ann, [{"id": 1, "file_name": "s.png", "width": 100, "height": 100}],
                [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 30.0, 30.0], "area": 900.0, "iscrowd": 0}])
    preds = [{"image_id": 1, "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]), "labels": torch.tensor([1]), "scores": torch.tensor([0.9])}]
    metrics = evaluate_coco_predictions(ann, preds)
    assert metrics["AP_Caries"] == pytest.approx(1.0)
    # No Deep Caries lesion exists, so its AP is undefined rather than a measured 0
    assert np.isnan(metrics["AP_DeepCaries"])
