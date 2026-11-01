# tests/test_train_cli.py
import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.models import (
    FasterRCNNRunner,
    DINORunner,
    HierarchicalDetRunner,
    HierarchicalNetRunner,
    DiffusionNetRunner
)
from scripts.train import collate_fn, parse_args, main


def test_runner_factory():
    """Verify integration of all model runner constructors from BenchmarkConfig."""
    cfg = BenchmarkConfig.from_auto_env()
    runner_map = {
        "hierarchical_net": HierarchicalNetRunner,
        "diffusion_net": DiffusionNetRunner,
        "faster_rcnn": FasterRCNNRunner,
        "dino": DINORunner,
        "hierarchical_det": HierarchicalDetRunner
    }
    for name, runner_cls in runner_map.items():
        runner = runner_cls(cfg)
        assert runner is not None
        assert runner.config.num_classes == 4


def test_collate_fn():
    """Verify custom collate_fn unzips batch items into tuple of images and tuple of targets."""
    img1 = torch.zeros((3, 64, 64))
    img2 = torch.ones((3, 64, 64))
    target1 = {"boxes": torch.zeros((1, 4)), "labels": torch.tensor([1])}
    target2 = {"boxes": torch.ones((1, 4)), "labels": torch.tensor([2])}

    batch = [(img1, target1), (img2, target2)]
    images, targets = collate_fn(batch)

    assert len(images) == 2
    assert len(targets) == 2
    assert torch.equal(images[0], img1)
    assert torch.equal(images[1], img2)
    assert torch.equal(targets[0]["labels"], target1["labels"])
    assert torch.equal(targets[1]["labels"], target2["labels"])


def test_parse_args():
    """Verify CLI argument parsing defaults and overrides."""
    args = parse_args(["--model", "faster_rcnn", "--epochs", "5", "--batch_size", "2", "--lr", "0.0002"])
    assert args.model == "faster_rcnn"
    assert args.epochs == 5
    assert args.batch_size == 2
    assert args.lr == 0.0002
    assert args.data_dir == "data"
    assert args.output_dir == "outputs"


def test_train_cli_help():
    """Smoke test running scripts/train.py --help via subprocess."""
    cmd = [sys.executable, "scripts/train.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--model" in res.stdout
    assert "faster_rcnn" in res.stdout
    assert "dino" in res.stdout
    assert "hierarchical_det" in res.stdout


def test_train_cli_missing_annotations(tmp_path):
    """Verify graceful handling when annotations file does not exist."""
    fake_data = tmp_path / "empty_data"
    ret = main([
        "--model", "faster_rcnn",
        "--data_dir", str(fake_data),
        "--output_dir", str(tmp_path / "out")
    ])
    assert ret != 0


def test_train_cli_synthetic_epoch(tmp_path):
    """Verify end-to-end training execution of 1 epoch on synthetic dataset with checkpoint generation."""
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"
    processed_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    ann_data = {
        "images": [{"id": 1, "file_name": "sample.png", "width": 64, "height": 64}],
        "categories": [
            {"id": 1, "name": "Caries"},
            {"id": 2, "name": "Deep Caries"},
            {"id": 3, "name": "Periapical Lesion"},
            {"id": 4, "name": "Impacted"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0}
        ]
    }
    with open(processed_dir / "instances_train.json", "w") as f:
        json.dump(ann_data, f)

    output_dir = tmp_path / "outputs"
    ret = main([
        "--model", "faster_rcnn",
        "--epochs", "1",
        "--batch_size", "1",
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir)
    ])

    assert ret == 0
    model_out = output_dir / "faster_rcnn"
    assert (model_out / "last.pth").exists()
    assert (model_out / "best.pth").exists()

    ckpt = torch.load(model_out / "last.pth", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "extra_info" in ckpt
    assert ckpt["extra_info"]["epoch"] == 1
    assert "loss" in ckpt["extra_info"]
