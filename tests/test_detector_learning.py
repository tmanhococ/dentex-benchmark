"""
Capability tests: a detector must localize objects whose position changes from image to image.

The original HierarchicalNet / DiffusionNet heads global-average-pooled the feature maps, so their
box outputs could not depend on where a lesion is. They can never pass these tests; any real
detector overfits four synthetic scans within a few hundred iterations.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.dataset import DentexDataset
from dentex_bench.eval.coco_eval import evaluate_runner
from dentex_bench.models import HierarchicalNetRunner, DiffusionNetRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="overfit tests need a GPU")

# (image size, list of (category, x, y, w, h)); object position and class vary per image
SCENES = [
    ((256, 384), [(1, 40, 60, 60, 60), (4, 250, 120, 100, 70)]),
    ((256, 384), [(4, 30, 150, 100, 70), (1, 280, 40, 60, 60)]),
    ((256, 384), [(1, 170, 100, 60, 60)]),
    ((256, 384), [(4, 150, 20, 100, 70), (1, 60, 170, 60, 60)]),
]
INTENSITY = {1: 230, 4: 120}


def _make_dataset(root: Path) -> Path:
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i, ((h, w), objs) in enumerate(SCENES, start=1):
        img = rng.integers(20, 60, size=(h, w), dtype=np.uint8)
        for cat, x, y, bw, bh in objs:
            if cat == 1:
                img[y:y + bh, x:x + bw] = INTENSITY[cat]
            else:
                cv2.ellipse(img, (x + bw // 2, y + bh // 2), (bw // 2, bh // 2), 0, 0, 360, INTENSITY[cat], -1)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
        cv2.imwrite(str(root / f"scan_{i}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        images.append({"id": i, "file_name": f"scan_{i}.png", "width": w, "height": h})
    ann = root / "instances.json"
    ann.write_text(json.dumps({"images": images, "annotations": anns, "categories": [
        {"id": 1, "name": "Caries"}, {"id": 2, "name": "Deep Caries"},
        {"id": 3, "name": "Periapical Lesion"}, {"id": 4, "name": "Impacted Tooth"}]}))
    return ann


def _overfit(runner_cls, tmp_path, iterations, lr):
    ann = _make_dataset(tmp_path)
    cfg = BenchmarkConfig.from_auto_env()
    cfg.img_size = 256
    cfg.max_size = 384
    cfg.gradient_accumulation_steps = 1
    cfg.learning_rate = lr
    torch.manual_seed(0)
    runner = runner_cls(cfg)
    runner.build_model(pretrained=True)

    ds = DentexDataset(img_dir=tmp_path, ann_file=ann, resize=False)
    samples = [ds[i] for i in range(len(ds))]
    batches = [((img,), (tgt,)) for img, tgt in samples]
    for _ in range(iterations // len(batches)):
        runner.train_epoch(batches, epoch=1)
    return evaluate_runner(runner, ann, tmp_path)


def test_hierarchical_net_localizes_moving_lesions(tmp_path):
    metrics = _overfit(HierarchicalNetRunner, tmp_path, iterations=240, lr=1e-4)
    assert metrics["mAP50"] >= 0.8, metrics


def test_diffusion_net_localizes_moving_lesions(tmp_path):
    metrics = _overfit(DiffusionNetRunner, tmp_path, iterations=600, lr=1e-4)
    assert metrics["mAP50"] >= 0.8, metrics
