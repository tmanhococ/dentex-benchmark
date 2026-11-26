# scripts/train.py
import argparse
import json
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import cv2
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
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


def out_of_time(started_at: float, limit_min, epoch_seconds: list, now: float | None = None) -> bool:
    """True when the next epoch (estimated as the slowest one so far, x1.1) would not finish within limit_min."""
    if not limit_min:
        return False
    now = time.time() if now is None else now
    remaining = float(limit_min) * 60.0 - (now - started_at)
    needed = 1.1 * max(epoch_seconds) if epoch_seconds else 0.0
    return remaining <= 0 or needed > remaining


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
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Alias for --batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing raw/ and processed/")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory where checkpoints are saved")
    parser.add_argument("--augment-config", type=str, default=None, help="Path to online augmentation YAML config")
    parser.add_argument("--train-ann", type=str, default=None, help="Path to custom training annotations file (e.g. offline augmented COCO JSON)")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pth if found in output directory")
    parser.add_argument("--eval-interval", type=int, default=1, help="Run validation evaluation every N epochs")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (0 avoids fork deadlock)")
    parser.add_argument("--time-limit-min", type=float, default=None,
                        help="Wall-clock budget in minutes; stops cleanly before an epoch that would not finish in time")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps (default: 16 // batch_size)")
    return parser.parse_args(args)


def resolve_ann_paths(args) -> tuple[Path, Path]:
    """
    Resolve (train_ann, val_ann) and enforce INV-2: validation set is never augmented
    and cannot be replaced by training annotations.
    """
    data_dir = Path(args.data_dir)
    train_ann = Path(args.train_ann) if getattr(args, "train_ann", None) else data_dir / "processed" / "instances_train.json"
    val_ann = data_dir / "processed" / "instances_val.json"
    if train_ann.resolve() == val_ann.resolve():
        raise AssertionError(
            f"INV-2 VIOLATION: val_ann ({val_ann}) must never be replaced by or equal to train_ann ({train_ann})."
        )
    return train_ann, val_ann


def get_augment_metadata(train_ann: Path | None, augment_config_path: str | None = None) -> dict:
    """Extract augmentation metadata and manifest hash for logging into checkpoints."""
    metadata = {}
    if train_ann is not None:
        train_ann = Path(train_ann)
        metadata["custom_train_ann"] = str(train_ann)
        candidates = [
            train_ann.parent / "manifest.json",
            train_ann.parent / "state" / "manifest.json",
            train_ann.parent.parent / "state" / "manifest.json",
        ]
        for c in candidates:
            if c.is_file():
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    m_hash = m_data.get("manifest_hash")
                    if m_hash:
                        metadata["manifest_hash"] = m_hash
                    metadata["manifest_path"] = str(c)
                    break
                except Exception:
                    pass

    if augment_config_path:
        from dentex_bench.augment.config import load_config
        aug_cfg = load_config(augment_config_path)
        metadata["augment_config"] = str(augment_config_path)
        metadata["augment_run_name"] = aug_cfg.run_name

    return metadata


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
        cfg.gradient_accumulation_steps = max(1, 16 // max(1, cfg.batch_size))
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if getattr(args, "grad_accum", None):
        cfg.gradient_accumulation_steps = max(1, int(args.grad_accum))
    if getattr(args, "num_workers", None) is not None:
        cfg.num_workers = args.num_workers

    print(f"=== Starting Training: {args.model.upper()} ===")
    print(f"Device: {cfg.device} ({cfg.vram_gb} GB VRAM) | Batch size: {cfg.batch_size} (Grad Accum: {cfg.gradient_accumulation_steps})", flush=True)

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
    train_ann, val_ann = resolve_ann_paths(args)
    if not train_ann.exists():
        print(f"[dentex_bench] Error: Training annotations not found at {train_ann}.")
        print("Please run `python scripts/download_and_prepare.py` first.")
        return 1

    # Images are looked up recursively; a missing or mismatched image raises instead of being faked
    val_img_dir = data_dir / "raw" if (data_dir / "raw").is_dir() else data_dir
    train_img_dir = val_img_dir
    if train_ann.parent.parent.joinpath("images").is_dir():
        train_img_dir = train_ann.parent.parent / "images"
    elif train_ann.parent.joinpath("images").is_dir():
        train_img_dir = train_ann.parent / "images"

    dataset = DentexDataset(img_dir=train_img_dir, ann_file=train_ann, target_short=cfg.img_size, max_long=cfg.max_size)

    # Online Augmentation or Offline metadata tracking
    augment_metadata = get_augment_metadata(
        train_ann=Path(args.train_ann) if getattr(args, "train_ann", None) else None,
        augment_config_path=getattr(args, "augment_config", None),
    )
    if "manifest_hash" in augment_metadata:
        print(f"[Augment] Found offline augmentation manifest at {augment_metadata.get('manifest_path')} (hash: {augment_metadata['manifest_hash']})")

    if getattr(args, "augment_config", None):
        from dentex_bench.augment.config import load_config
        aug_cfg = load_config(args.augment_config)
        print(f"[Augment] Loaded online augmentation config: {args.augment_config} (run: {aug_cfg.run_name})")
        try:
            from dentex_bench.augment.online import SwapAugmentDataset
            from dentex_bench.augment.bank import ToothBank
            bank_dir = Path(aug_cfg.paths.out_root) / "bank"
            if ToothBank.is_cached(bank_dir):
                bank = ToothBank.load(bank_dir)
            else:
                bank = ToothBank([])
            dataset = SwapAugmentDataset(base=dataset, cfg=aug_cfg, bank=bank)
            print("[Augment] Wrapped training dataset with SwapAugmentDataset")
        except ImportError:
            print("[Augment] SwapAugmentDataset not yet available; proceeding with standard dataset")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers
    )

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_score = float("-inf")
    history = []

    last_pth = output_dir / "last.pth"
    if getattr(args, "resume", False) and last_pth.is_file():
        # load_checkpoint returns the extra_info dict saved with the checkpoint (the epoch record)
        saved_info = runner.load_checkpoint(last_pth) or {}
        if "extra_info" in saved_info:
            saved_info = saved_info["extra_info"] or {}
        saved_epoch = int(saved_info.get("epoch", 0))
        start_epoch = saved_epoch + 1
        history_path = output_dir / "history.json"
        if history_path.is_file():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass
        # Keep only epochs the restored weights have seen, so history never repeats an epoch
        history = [h for h in history if int(h.get("epoch", 0)) <= saved_epoch]
        if history:
            if val_ann.exists():
                best_score = max(h.get("val_mAP50", float("-inf")) for h in history)
            else:
                best_score = max(-h["loss"] for h in history)
        print(f"[dentex_bench] Resumed from {last_pth} at epoch {start_epoch}/{cfg.epochs}", flush=True)

    if val_ann.exists():
        print(f"best.pth is selected by validation mAP50 on {val_ann}", flush=True)
    else:
        print(f"[dentex_bench] Warning: {val_ann} not found; best.pth falls back to the lowest training loss.", flush=True)

    eval_interval = getattr(args, "eval_interval", 1) or 1
    started_at = time.time()
    epoch_seconds: list = [float(h["seconds"]) for h in history if "seconds" in h]  # durations seen before a resume
    for epoch in range(start_epoch, cfg.epochs + 1):
        if out_of_time(started_at, getattr(args, "time_limit_min", None), epoch_seconds):
            print(f"[dentex_bench] TIME LIMIT: stopping before epoch {epoch}/{cfg.epochs} "
                  f"(budget {args.time_limit_min:.0f} min). Resume later with --resume.", flush=True)
            break
        epoch_t0 = time.time()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        elif hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch)

        loss_dict = runner.train_epoch(dataloader, epoch)
        record = {"epoch": epoch, "loss": loss_dict["loss"]}
        if augment_metadata:
            record["augmentation"] = augment_metadata

        # Evaluate validation only on scheduled intervals or final epoch to maximize training speed
        should_eval = (epoch % eval_interval == 0) or (epoch == cfg.epochs)
        if val_ann.exists() and should_eval:
            val_metrics = evaluate_runner(runner, val_ann, val_img_dir)
            record.update({"val_mAP50": val_metrics["mAP50"], "val_mAP50_95": val_metrics["mAP50_95"]})
            score = val_metrics["mAP50"]
        elif val_ann.exists():
            # Carry forward previous best/known score or estimate
            prev_map = history[-1].get("val_mAP50", 0.0) if history else 0.0
            record.update({"val_mAP50": prev_map, "val_mAP50_95": history[-1].get("val_mAP50_95", 0.0) if history else 0.0})
            score = prev_map
        else:
            score = -loss_dict["loss"]

        record["seconds"] = round(time.time() - epoch_t0, 1)
        epoch_seconds.append(record["seconds"])
        history.append(record)
        format_val = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"Epoch [{epoch}/{cfg.epochs}] - " + ", ".join(f"{k}: {format_val(v)}" for k, v in record.items() if k not in ("epoch", "augmentation", "seconds")), flush=True)

        # Always update last.pth and history at the current epoch, so a disconnect loses at most one epoch
        if score > best_score:
            best_score = score
            runner.save_checkpoint(output_dir / "best.pth", extra_info=record)
        runner.save_checkpoint(output_dir / "last.pth", extra_info=record)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Save history json and curve plot
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if history:
        plot_training_history(history, output_dir / "training_curves.png")

    print(f"=== Training Finished for {args.model}. Checkpoints saved to {output_dir} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
