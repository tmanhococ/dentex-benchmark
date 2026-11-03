# scripts/train.py
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add project root so it runs standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.dataset import DentexDataset
from dentex_bench.eval.coco_eval import evaluate_runner
from dentex_bench.models import (
    FasterRCNNRunner,
    DINORunner,
    HierarchicalDetRunner,
    HierarchicalNetRunner,
    DiffusionNetRunner
)


def collate_fn(batch):
    """Collate function for object detection datasets returning tuple of images and tuple of targets."""
    return tuple(zip(*batch))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a detection model on DENTEX 4 classes")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hierarchical_net", "diffusion_net", "faster_rcnn", "dino", "hierarchical_det"],
        help="Detector model runner to train"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing raw/ and processed/")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory where checkpoints are saved")
    return parser.parse_args(args)


def plot_training_history(history: list, output_file: Path):
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, marker="o", color="#2b5c8f", label="Training Loss")
    ax.set_title("Training Loss & Validation mAP50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    if any("val_mAP50" in h for h in history):
        ax2 = ax.twinx()
        ax2.plot(epochs, [h.get("val_mAP50", float("nan")) for h in history], marker="s", color="#e27c38", label="Val mAP50")
        ax2.set_ylabel("Val mAP50")
        ax2.legend(loc="upper right")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)

    cfg = BenchmarkConfig.from_auto_env()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr

    print(f"=== Starting Training: {args.model.upper()} ===")
    print(f"Device: {cfg.device} ({cfg.vram_gb} GB VRAM) | Batch size: {cfg.batch_size} (Grad Accum: {cfg.gradient_accumulation_steps})")

    # Initialize runner
    runners = {
        "hierarchical_net": HierarchicalNetRunner,
        "diffusion_net": DiffusionNetRunner,
        "faster_rcnn": FasterRCNNRunner,
        "dino": DINORunner,
        "hierarchical_det": HierarchicalDetRunner
    }
    runner = runners[args.model](cfg)
    runner.build_model()

    data_dir = Path(args.data_dir)
    train_ann = data_dir / "processed" / "instances_train.json"
    val_ann = data_dir / "processed" / "instances_val.json"
    if not train_ann.exists():
        print(f"[dentex_bench] Error: Training annotations not found at {train_ann}.")
        print("Please run `python scripts/download_and_prepare.py` first.")
        return 1

    # Images are looked up recursively; a missing or mismatched image raises instead of being faked
    img_dir = data_dir / "raw" if (data_dir / "raw").is_dir() else data_dir

    dataset = DentexDataset(img_dir=img_dir, ann_file=train_ann, target_short=cfg.img_size, max_long=cfg.max_size)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers
    )

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if val_ann.exists():
        print(f"best.pth is selected by validation mAP50 on {val_ann}")
    else:
        print(f"[dentex_bench] Warning: {val_ann} not found; best.pth falls back to the lowest training loss.")

    best_score = float("-inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        loss_dict = runner.train_epoch(dataloader, epoch)
        record = {"epoch": epoch, "loss": loss_dict["loss"]}
        if val_ann.exists():
            val_metrics = evaluate_runner(runner, val_ann, img_dir)
            record.update({"val_mAP50": val_metrics["mAP50"], "val_mAP50_95": val_metrics["mAP50_95"]})
            score = val_metrics["mAP50"]
        else:
            score = -loss_dict["loss"]
        history.append(record)
        print(f"Epoch [{epoch}/{cfg.epochs}] - " + ", ".join(f"{k}: {v:.4f}" for k, v in record.items() if k != "epoch"))

        # Always update last.pth at current epoch
        runner.save_checkpoint(output_dir / "last.pth", extra_info=record)
        if score > best_score:
            best_score = score
            runner.save_checkpoint(output_dir / "best.pth", extra_info=record)

    # Save history json and curve plot
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, output_dir / "training_curves.png")

    print(f"=== Training Finished for {args.model}. Checkpoints saved to {output_dir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
