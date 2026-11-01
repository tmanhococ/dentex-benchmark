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
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
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

    train_ann = Path(args.data_dir) / "processed" / "instances_train.json"
    if not train_ann.exists():
        fallback_ann = Path("data/processed/instances_train.json")
        if fallback_ann.exists():
            train_ann = fallback_ann

    if not train_ann.exists():
        print(f"[dentex_bench] Error: Training annotations not found at {train_ann}.")
        print("Please run `python scripts/download_and_prepare.py` first.")
        return 1

    img_dir = Path(args.data_dir) / "raw"
    if not img_dir.exists() or not any(img_dir.glob("*.*")):
        if Path(args.data_dir).exists() and any(Path(args.data_dir).glob("**/*.png")):
            img_dir = Path(args.data_dir)
        elif Path("data/raw").exists():
            img_dir = Path("data/raw")
        elif Path("data").exists():
            img_dir = Path("data")

    dataset = DentexDataset(img_dir=img_dir, ann_file=train_ann)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers
    )

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        loss_dict = runner.train_epoch(dataloader, epoch)
        current_loss = loss_dict["loss"]
        history.append({"epoch": epoch, "loss": current_loss})
        print(f"Epoch [{epoch}/{cfg.epochs}] - Loss: {current_loss:.4f}")

        # Always update last.pth at current epoch
        runner.save_checkpoint(output_dir / "last.pth", extra_info={"epoch": epoch, "loss": current_loss})
        # Save best.pth when loss improves
        if current_loss < best_loss:
            best_loss = current_loss
            runner.save_checkpoint(output_dir / "best.pth", extra_info={"epoch": epoch, "loss": current_loss})

    # Save history json and curve plot
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, output_dir / "training_curves.png")

    print(f"=== Training Finished for {args.model}. Checkpoints saved to {output_dir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
