# tests/test_dino.py
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.dino import DINORunner


def test_dino_runner_build():
    cfg = BenchmarkConfig.from_auto_env()
    runner = DINORunner(cfg)
    model = runner.build_model()
    assert model is not None
    assert runner.model is model
    assert runner.optimizer is not None

    # Test inference pass with dummy tensor
    dummy_imgs = [torch.rand(3, 400, 400)]
    preds = runner.predict(dummy_imgs)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "labels" in preds[0]
    assert "scores" in preds[0]


def test_dino_runner_predict_varying_sizes():
    cfg = BenchmarkConfig.from_auto_env()
    runner = DINORunner(cfg)
    runner.build_model()

    # Pass list of tensors with varying sizes
    dummy_imgs = [
        torch.rand(3, 300, 400),
        torch.rand(3, 400, 300)
    ]
    preds = runner.predict(dummy_imgs)
    assert len(preds) == 2
    for p in preds:
        assert "boxes" in p
        assert "labels" in p
        assert "scores" in p
        assert p["boxes"].ndim == 2
        assert p["boxes"].shape[-1] == 4


def test_dino_runner_train_epoch():
    cfg = BenchmarkConfig.from_auto_env()
    cfg.gradient_accumulation_steps = 2
    runner = DINORunner(cfg)
    runner.build_model()

    # Create dummy dataloader with batches of varying image shapes and targets
    batch1 = (
        [torch.rand(3, 320, 320), torch.rand(3, 320, 320)],
        [
            {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]), "labels": torch.tensor([2])}
        ]
    )
    batch2 = (
        [torch.rand(3, 300, 400), torch.rand(3, 350, 320)],
        [
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)},
            {"boxes": torch.tensor([[15.0, 15.0, 45.0, 45.0]]), "labels": torch.tensor([4])}
        ]
    )
    dataloader = [batch1, batch2]

    metrics = runner.train_epoch(dataloader, epoch=1)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] > 0


def test_dino_runner_checkpoint(tmp_path):
    cfg = BenchmarkConfig.from_auto_env()
    runner = DINORunner(cfg)
    runner.build_model()

    ckpt_path = tmp_path / "checkpoints" / "dino_test.pth"
    runner.save_checkpoint(ckpt_path, extra_info={"epoch": 3, "val_loss": 0.85})
    assert ckpt_path.exists()

    runner2 = DINORunner(cfg)
    runner2.build_model()
    extra_info = runner2.load_checkpoint(ckpt_path)
    assert extra_info["epoch"] == 3
    assert extra_info["val_loss"] == 0.85
