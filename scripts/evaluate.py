# scripts/evaluate.py
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import torch

# Add project root so it runs standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.eval.coco_eval import evaluate_coco_predictions, predict_dataset
from dentex_bench.eval.reporter import save_reports
from dentex_bench.models import (
    FasterRCNNRunner,
    DINORunner,
    HierarchicalDetRunner,
    HierarchicalNetRunner,
    DiffusionNetRunner
)

RUNNERS = {
    "hierarchical_net": HierarchicalNetRunner,
    "diffusion_net": DiffusionNetRunner,
    "faster_rcnn": FasterRCNNRunner,
    "dino": DINORunner,
    "hierarchical_det": HierarchicalDetRunner
}
NAME_MAP = {
    "hierarchical_net": "HierarchicalNet",
    "diffusion_net": "DiffusionNet",
    "faster_rcnn": "Faster R-CNN",
    "dino": "DINO",
    "hierarchical_det": "HierarchicalDet"
}


def evaluate_checkpoint(model_key: str, checkpoint: Path, val_ann: Path, img_dir: Path, output_dir: Path) -> dict:
    cfg = BenchmarkConfig.from_auto_env()
    runner = RUNNERS[model_key](cfg)
    runner.build_model(pretrained=False)
    runner.load_checkpoint(checkpoint)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    predictions, latencies = predict_dataset(runner, val_ann, img_dir)
    metrics = evaluate_coco_predictions(val_ann, predictions)
    metrics["latency_ms"] = round(float(sorted(latencies)[len(latencies) // 2]), 1) if latencies else 0.0
    metrics["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

    coco_dt = [
        {"image_id": p["image_id"], "category_id": int(l), "score": float(s),
         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
        for p in predictions for b, l, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist())
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"predictions_{model_key}.json").write_text(json.dumps(coco_dt), encoding="utf-8")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints on the validation split (COCO metrics)")
    parser.add_argument("--compare_all", action="store_true", help="Evaluate every model that has <checkpoints_dir>/<model>/best.pth")
    parser.add_argument("--model", type=str, default=None, choices=list(RUNNERS), help="Evaluate a single model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for --model (default <checkpoints_dir>/<model>/best.pth)")
    parser.add_argument("--checkpoints_dir", type=str, default="outputs", help="Directory holding <model>/best.pth")
    parser.add_argument("--val_ann", type=str, default="data/processed/instances_val.json", help="Path to val COCO json")
    parser.add_argument("--img_dir", type=str, default="data/raw", help="Directory searched (recursively) for the val images")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save markdown and latex reports")
    args = parser.parse_args(argv)

    if not args.compare_all and not args.model:
        parser.error("pass --model <name> or --compare_all")

    val_ann = Path(args.val_ann)
    if not val_ann.exists():
        print(f"[dentex_bench] Error: validation annotations not found at {val_ann}")
        return 1

    if args.compare_all:
        jobs = [(key, Path(args.checkpoints_dir) / key / "best.pth") for key in RUNNERS]
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.checkpoints_dir) / args.model / "best.pth"
        jobs = [(args.model, ckpt)]

    results = {}
    for key, ckpt in jobs:
        if not ckpt.is_file():
            print(f"[dentex_bench] Skipping {NAME_MAP[key]}: no checkpoint at {ckpt}")
            continue
        print(f"=== Evaluating {NAME_MAP[key]} from {ckpt} ===")
        results[NAME_MAP[key]] = evaluate_checkpoint(key, ckpt, val_ann, Path(args.img_dir), Path(args.output_dir))
        for k, v in results[NAME_MAP[key]].items():
            print(f"  {k}: {v}")

    if not results:
        print("[dentex_bench] Error: no checkpoint was evaluated, so no report was written.")
        return 1

    save_reports(results, Path(args.output_dir))
    print(f"Reports for {list(results)} saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
