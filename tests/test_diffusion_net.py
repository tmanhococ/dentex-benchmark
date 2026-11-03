import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.diffusion_net import DiffusionNetRunner


def _runner(**overrides):
    cfg = BenchmarkConfig.from_auto_env()
    cfg.img_size, cfg.max_size = 256, 512
    for k, v in overrides.items():
        setattr(cfg, k, v)
    runner = DiffusionNetRunner(cfg)
    runner.build_model(pretrained=False)
    return runner


def test_diffusion_net_predicts_boxes_in_input_pixel_frame():
    runner = _runner()
    preds = runner.predict([torch.rand(3, 300, 700)])
    assert len(preds) == 1
    boxes, labels, scores = preds[0]["boxes"], preds[0]["labels"], preds[0]["scores"]
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    assert len(boxes) == len(labels) == len(scores) <= 100
    if len(boxes):
        assert boxes[:, 0::2].min() >= 0 and boxes[:, 0::2].max() <= 700
        assert boxes[:, 1::2].min() >= 0 and boxes[:, 1::2].max() <= 300
        assert labels.min() >= 1 and labels.max() <= 4


def test_diffusion_net_train_epoch():
    runner = _runner(gradient_accumulation_steps=2)
    images = [torch.rand(3, 200, 200), torch.rand(3, 220, 220)]
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
        # Scans without any of the 4 pathologies exist in DENTEX
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}
    ]
    loss_dict = runner.train_epoch([(images, targets), (images, targets)], epoch=1)
    assert isinstance(loss_dict["loss"], float)
    assert loss_dict["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(loss_dict["loss"]))


def test_diffusion_net_checkpoint(tmp_path):
    runner = _runner()
    ckpt_path = tmp_path / "checkpoints" / "diffusion_net_test.pth"
    runner.save_checkpoint(ckpt_path, extra_info={"epoch": 5, "val_mAP50": 0.42})
    assert ckpt_path.exists()

    info = _runner().load_checkpoint(ckpt_path)
    assert info.get("epoch") == 5
    assert info.get("val_mAP50") == 0.42


def test_diffusion_net_checkpoint_from_pretrained_run_loads_into_evaluation_runner(tmp_path):
    # Training builds with COCO weights, evaluation rebuilds without them before loading best.pth
    cfg = BenchmarkConfig.from_auto_env()
    trained = DiffusionNetRunner(cfg)
    trained.build_model(pretrained=True)
    ckpt = tmp_path / "best.pth"
    trained.save_checkpoint(ckpt)

    evaluator = DiffusionNetRunner(cfg)
    evaluator.build_model(pretrained=False)
    evaluator.load_checkpoint(ckpt)
    trainable = lambda r: [n for n, p in r.model.named_parameters() if p.requires_grad]
    assert trainable(evaluator) == trainable(trained)


def test_diffusion_net_inference_is_reproducible():
    # Sampling starts from Gaussian boxes; benchmark numbers must not change between runs
    runner = _runner()
    img = torch.rand(3, 300, 700)
    first, second = runner.predict([img])[0], runner.predict([img])[0]
    assert torch.equal(first["boxes"], second["boxes"])
    assert torch.equal(first["scores"], second["scores"])
