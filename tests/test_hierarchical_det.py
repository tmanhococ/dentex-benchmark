import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.hierarchical_det import (
    HierarchicalDetRunner,
    SinusoidalPositionEmbeddings,
    DynamicDiffusionHead
)


def test_sinusoidal_position_embeddings_and_head():
    # Test SinusoidalPositionEmbeddings
    dim = 256
    pos_emb = SinusoidalPositionEmbeddings(dim)
    timesteps = torch.tensor([0.0, 250.0, 500.0, 1000.0])
    out = pos_emb(timesteps)
    assert out.shape == (4, dim)

    # Test DynamicDiffusionHead
    head = DynamicDiffusionHead(num_classes=4, hidden_dim=dim, num_proposals=100)
    features = torch.rand(2, dim, 12, 12)
    noisy_boxes = torch.rand(2, 100, 4)
    t = torch.tensor([10.0, 50.0])
    logits, refined_boxes = head(features, noisy_boxes, t)
    assert logits.shape == (2, 100, 5)  # 4 classes + 1 background
    assert refined_boxes.shape == (2, 100, 4)
    assert (refined_boxes >= 0.0).all() and (refined_boxes <= 1.0).all()


def test_hierarchical_det_build():
    cfg = BenchmarkConfig.from_auto_env()
    runner = HierarchicalDetRunner(cfg)
    model = runner.build_model()
    assert model is not None
    assert runner.model is model
    assert runner.optimizer is not None

    # Test DDIM inference pass with dummy tensor
    dummy_imgs = [torch.rand(3, 400, 400)]
    preds = runner.predict(dummy_imgs)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "labels" in preds[0]
    assert "scores" in preds[0]
    assert preds[0]["boxes"].ndim == 2
    assert preds[0]["boxes"].shape[-1] == 4


def test_hierarchical_det_predict_varying_sizes():
    cfg = BenchmarkConfig.from_auto_env()
    runner = HierarchicalDetRunner(cfg)
    runner.build_model()

    # Empty list edge case
    assert runner.predict([]) == []

    # Pass list of tensors with varying sizes
    w1, h1 = 400, 300
    w2, h2 = 300, 450
    dummy_imgs = [
        torch.rand(3, h1, w1),
        torch.rand(3, h2, w2)
    ]
    preds = runner.predict(dummy_imgs)
    assert len(preds) == 2

    for i, (p, (w, h)) in enumerate(zip(preds, [(w1, h1), (w2, h2)])):
        assert "boxes" in p
        assert "labels" in p
        assert "scores" in p
        assert p["boxes"].ndim == 2
        assert p["boxes"].shape[-1] == 4
        # Verify clamping within image dimensions
        if p["boxes"].numel() > 0:
            assert (p["boxes"][:, 0] >= 0.0).all()
            assert (p["boxes"][:, 0] <= float(w)).all()
            assert (p["boxes"][:, 1] >= 0.0).all()
            assert (p["boxes"][:, 1] <= float(h)).all()
            assert (p["boxes"][:, 2] >= 0.0).all()
            assert (p["boxes"][:, 2] <= float(w)).all()
            assert (p["boxes"][:, 3] >= 0.0).all()
            assert (p["boxes"][:, 3] <= float(h)).all()
            # Verify 1-based class labels
            assert (p["labels"] >= 1).all()
            assert (p["labels"] <= 4).all()


def test_hierarchical_det_train_epoch():
    cfg = BenchmarkConfig.from_auto_env()
    cfg.gradient_accumulation_steps = 2
    runner = HierarchicalDetRunner(cfg)
    runner.build_model()

    # Dummy dataloader with batches of varying image shapes and targets
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


def test_hierarchical_det_checkpoint(tmp_path):
    cfg = BenchmarkConfig.from_auto_env()
    runner = HierarchicalDetRunner(cfg)
    runner.build_model()

    ckpt_path = tmp_path / "checkpoints" / "hierarchical_det_test.pth"
    runner.save_checkpoint(ckpt_path, extra_info={"epoch": 2, "val_loss": 0.42})
    assert ckpt_path.exists()

    runner2 = HierarchicalDetRunner(cfg)
    runner2.build_model()
    extra_info = runner2.load_checkpoint(ckpt_path)
    assert extra_info["epoch"] == 2
    assert extra_info["val_loss"] == 0.42
