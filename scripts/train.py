# scripts/train.py
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root so it runs standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.dataset import DentexDataset
from dentex_bench.models import FasterRCNNRunner, DINORunner, HierarchicalDetRunner


def collate_fn(batch):
    """Collate function for object detection datasets returning tuple of images and tuple of targets."""
    return tuple(zip(*batch))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a detection model on DENTEX 4 classes")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["faster_rcnn", "dino", "hierarchical_det"],
        help="Detector model runner to train"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing raw/ and processed/")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory where checkpoints are saved")
    return parser.parse_args(args)


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
        "faster_rcnn": FasterRCNNRunner,
        "dino": DINORunner,
        "hierarchical_det": HierarchicalDetRunner
    }
    runner = runners[args.model](cfg)
    runner.build_model()

    train_ann = Path(args.data_dir) / "processed" / "instances_train.json"
    img_dir = Path(args.data_dir) / "raw"

    if not train_ann.exists():
        print(f"[dentex_bench] Error: Training annotations not found at {train_ann}.")
        print("Please run `python scripts/download_and_prepare.py` first.")
        return 1

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
    for epoch in range(1, cfg.epochs + 1):
        loss_dict = runner.train_epoch(dataloader, epoch)
        current_loss = loss_dict["loss"]
        print(f"Epoch [{epoch}/{cfg.epochs}] - Loss: {current_loss:.4f}")

        # Save checkpoints
        runner.save_checkpoint(output_dir / "last.pth", extra_info={"epoch": epoch, "loss": current_loss})
        if current_loss < best_loss:
            best_loss = current_loss
            runner.save_checkpoint(output_dir / "best.pth", extra_info={"epoch": epoch, "loss": current_loss})

    print(f"=== Training Finished for {args.model}. Checkpoints saved to {output_dir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
