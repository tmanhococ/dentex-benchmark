# tests/test_faster_rcnn.py
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.faster_rcnn import FasterRCNNRunner


def test_faster_rcnn_runner_init():
    cfg = BenchmarkConfig.from_auto_env()
    runner = FasterRCNNRunner(cfg)
    model = runner.build_model()
    assert model is not None
    assert runner.model is model
    assert runner.optimizer is not None

    # Test forward pass with dummy batch
    dummy_imgs = [torch.rand(3, 300, 300).to(runner.device)]
    dummy_targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=runner.device),
        "labels": torch.tensor([1], device=runner.device)
    }]

    model.train()
    loss_dict = model(dummy_imgs, dummy_targets)
    assert "loss_classifier" in loss_dict
    assert "loss_box_reg" in loss_dict


def test_faster_rcnn_runner_predict():
    cfg = BenchmarkConfig.from_auto_env()
    runner = FasterRCNNRunner(cfg)
    runner.build_model()

    dummy_imgs = [torch.rand(3, 300, 300)]
    predictions = runner.predict(dummy_imgs)
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert "boxes" in predictions[0]
    assert "labels" in predictions[0]
    assert "scores" in predictions[0]


def test_faster_rcnn_runner_checkpoint(tmp_path):
    cfg = BenchmarkConfig.from_auto_env()
    runner = FasterRCNNRunner(cfg)
    runner.build_model()

    ckpt_path = tmp_path / "checkpoints" / "faster_rcnn_test.pth"
    runner.save_checkpoint(ckpt_path, extra_info={"epoch": 2, "map_50": 0.45})
    assert ckpt_path.exists()

    runner2 = FasterRCNNRunner(cfg)
    runner2.build_model()
    extra_info = runner2.load_checkpoint(ckpt_path)
    assert extra_info["epoch"] == 2
    assert extra_info["map_50"] == 0.45


def test_faster_rcnn_runner_train_epoch():
    cfg = BenchmarkConfig.from_auto_env()
    # Fast test config
    cfg.gradient_accumulation_steps = 1
    runner = FasterRCNNRunner(cfg)
    runner.build_model()

    dummy_batch = (
        [torch.rand(3, 300, 300)],
        [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1])
        }]
    )
    dataloader = [dummy_batch]

    result = runner.train_epoch(dataloader, epoch=1)
    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0
