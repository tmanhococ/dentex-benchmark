import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.diffusion_net import (
    DiffusionNetRunner,
    SinusoidalPositionEmbeddings,
    DynamicDiffusionHead
)


def test_diffusion_components():
    dim = 256
    pos_emb = SinusoidalPositionEmbeddings(dim)
    timesteps = torch.tensor([0.0, 250.0, 500.0, 1000.0])
    out = pos_emb(timesteps)
    assert out.shape == (4, dim)

    head = DynamicDiffusionHead(num_classes=4, hidden_dim=dim, num_proposals=100)
    features = torch.rand(2, dim, 14, 14)
    noisy_boxes = torch.rand(2, 100, 4)
    t = torch.tensor([10.0, 50.0])
    logits, refined_boxes = head(features, noisy_boxes, t)
    assert logits.shape == (2, 100, 5)  # 4 classes + 1 background
    assert refined_boxes.shape == (2, 100, 4)
    assert (refined_boxes >= 0.0).all() and (refined_boxes <= 1.0).all()


def test_diffusion_net_build_and_predict():
    cfg = BenchmarkConfig.from_auto_env()
    runner = DiffusionNetRunner(cfg)
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


def test_diffusion_net_train_epoch():
    cfg = BenchmarkConfig.from_auto_env()
    cfg.gradient_accumulation_steps = 2
    runner = DiffusionNetRunner(cfg)
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


def test_diffusion_net_checkpoint(tmp_path):
    cfg = BenchmarkConfig.from_auto_env()
    runner = DiffusionNetRunner(cfg)
    runner.build_model(pretrained=False)

    ckpt_path = tmp_path / "checkpoints" / "diffusion_net_test.pth"
    runner.save_checkpoint(ckpt_path, extra_info={"epoch": 10, "val_mAP": 0.55})
    assert ckpt_path.exists()

    runner2 = DiffusionNetRunner(cfg)
    runner2.build_model(pretrained=False)
    info = runner2.load_checkpoint(ckpt_path)
    assert info.get("epoch") == 10
    assert info.get("val_mAP") == 0.55
