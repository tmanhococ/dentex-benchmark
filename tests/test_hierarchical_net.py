import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.hierarchical_net import (
    HierarchicalNetRunner,
    FeaturePyramidNetwork,
    HierarchicalDetectionHead
)


def test_fpn_and_head_shapes():
    dim = 256
    fpn = FeaturePyramidNetwork(in_channels=[256, 512, 1024, 2048], out_channels=dim)
    c2 = torch.rand(2, 256, 64, 64)
    c3 = torch.rand(2, 512, 32, 32)
    c4 = torch.rand(2, 1024, 16, 16)
    c5 = torch.rand(2, 2048, 8, 8)
    pyramid = fpn([c2, c3, c4, c5])
    assert len(pyramid) == 4
    for p in pyramid:
        assert p.shape[1] == dim

    head = HierarchicalDetectionHead(in_channels=dim, num_classes=4, num_proposals=100)
    logits, pred_boxes = head(pyramid)
    assert logits.shape == (2, 100, 5)  # 4 classes + 1 background
    assert pred_boxes.shape == (2, 100, 4)
    assert (pred_boxes >= 0.0).all() and (pred_boxes <= 1.0).all()


def test_hierarchical_net_build_and_predict():
    cfg = BenchmarkConfig.from_auto_env()
    runner = HierarchicalNetRunner(cfg)
    model = runner.build_model(pretrained=False)
    assert model is not None
    assert runner.model is model
    assert runner.optimizer is not None

    dummy_imgs = [torch.rand(3, 400, 400)]
    preds = runner.predict(dummy_imgs)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "labels" in preds[0]
    assert "scores" in preds[0]
    assert preds[0]["boxes"].ndim == 2
    assert preds[0]["boxes"].shape[-1] == 4


def test_hierarchical_net_train_epoch():
    cfg = BenchmarkConfig.from_auto_env()
    cfg.gradient_accumulation_steps = 2
    runner = HierarchicalNetRunner(cfg)
    runner.build_model(pretrained=False)

    images = [torch.rand(3, 200, 200), torch.rand(3, 220, 220)]
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]), "labels": torch.tensor([2])}
    ]
    batch = (images, targets)
    dataloader = [batch, batch]

    loss_dict = runner.train_epoch(dataloader, epoch=1)
    assert "loss" in loss_dict
    assert isinstance(loss_dict["loss"], float)
    assert loss_dict["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(loss_dict["loss"]))


def test_hierarchical_net_checkpoint(tmp_path):
    cfg = BenchmarkConfig.from_auto_env()
    runner = HierarchicalNetRunner(cfg)
    runner.build_model(pretrained=False)

    ckpt_path = tmp_path / "checkpoints" / "hierarchical_net_test.pth"
    runner.save_checkpoint(ckpt_path, extra_info={"epoch": 5, "val_loss": 0.42})
    assert ckpt_path.exists()

    runner2 = HierarchicalNetRunner(cfg)
    runner2.build_model(pretrained=False)
    info = runner2.load_checkpoint(ckpt_path)
    assert info.get("epoch") == 5
    assert info.get("val_loss") == 0.42
