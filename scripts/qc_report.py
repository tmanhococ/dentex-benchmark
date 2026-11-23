# scripts/qc_report.py
"""Generate Quality Control & Rejection Analysis Report for DENTEX Augmentation Runs.

Reads:
- <run_dir>/state/manifest.json
- <run_dir>/state/recipes.jsonl
- Optional: <run_dir>/dataset/coco/instances_train_aug.json

Produces:
- <run_dir>/qc/summary.json
- <run_dir>/qc/seam_boxplot.png (if seam scores present)
- <run_dir>/qc/rejections_bar.png (if rejections present)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root so it runs standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from dentex_bench.augment.qc import compute_seam_score


def generate_report(run_dir: Path) -> Dict[str, Any]:
    """Generate QC summary report, seam statistics, and rejection analysis."""
    run_dir = Path(run_dir)
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    state_dir = run_dir / "state"

    # 1. Read manifest
    manifest_file = state_dir / "manifest.json"
    manifest_data = {}
    if manifest_file.is_file():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            print(f"[QC] Warning: Failed to parse manifest: {e}")

    # 2. Read recipes.jsonl
    recipes_file = state_dir / "recipes.jsonl"
    recipes: List[Dict[str, Any]] = []
    if recipes_file.is_file():
        with open(recipes_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recipes.append(json.loads(line))
                    except Exception:
                        pass

    # 3. Analyze rejections and extract or compute seam scores
    rejected_by_reason: Dict[str, int] = {}
    total_rejections = 0
    seam_scores: List[float] = []
    seam_scores_by_blender: Dict[str, List[float]] = {}
    total_ops = 0

    for r in recipes:
        # Rejections
        for rej in r.get("rejections", []):
            code = rej.get("code") or rej.get("reason", "UNKNOWN")
            rejected_by_reason[code] = rejected_by_reason.get(code, 0) + 1
            total_rejections += 1

        # Operations & Seam scores
        for op in r.get("ops", []):
            total_ops += 1
            blender = op.get("blend_mode", "unknown")
            s = op.get("seam_score")
            if s is not None:
                try:
                    s_float = float(s)
                    seam_scores.append(s_float)
                    seam_scores_by_blender.setdefault(blender, []).append(s_float)
                except (ValueError, TypeError):
                    pass

    # 4. If seam scores are missing from recipes, attempt computing them from synthetic images
    if not seam_scores:
        coco_path = run_dir / "dataset" / "coco" / "instances_train_aug.json"
        if not coco_path.is_file():
            coco_path = run_dir / "dataset" / "coco" / "instances.json"

        if coco_path.is_file():
            try:
                import cv2
                with open(coco_path, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)

                img_lookup = {img["id"]: img for img in coco_data.get("images", [])}
                img_dir = run_dir / "dataset" / "images"
                if not img_dir.is_dir():
                    img_dir = run_dir / "synthetic"

                for ann in coco_data.get("annotations", []):
                    if ann.get("synthetic"):
                        iid = ann.get("image_id")
                        img_info = img_lookup.get(iid)
                        if not img_info:
                            continue
                        img_file = img_dir / img_info.get("file_name", "")
                        if not img_file.is_file():
                            continue

                        syn_img = cv2.imread(str(img_file))
                        if syn_img is None:
                            continue

                        h, w = syn_img.shape[:2]
                        mask = np.zeros((h, w), dtype=np.uint8)
                        bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
                        x1, y1 = max(0, int(bx)), max(0, int(by))
                        x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
                        if x2 > x1 and y2 > y1:
                            mask[y1:y2, x1:x2] = 255
                            s_val = compute_seam_score(syn_img, mask)
                            seam_scores.append(s_val)
            except Exception as e:
                print(f"[QC] Warning: Failed to compute seam scores from dataset: {e}")

    # 5. Compute statistics
    if seam_scores:
        seam_mean = float(np.mean(seam_scores))
        seam_std = float(np.std(seam_scores))
        seam_median = float(np.median(seam_scores))
        seam_min = float(np.min(seam_scores))
        seam_max = float(np.max(seam_scores))
    else:
        seam_mean = 0.0
        seam_std = 0.0
        seam_median = 0.0
        seam_min = 0.0
        seam_max = 0.0

    summary: Dict[str, Any] = {
        "status": "success",
        "run_name": manifest_data.get("run_name", run_dir.name),
        "total_recipes": len(recipes),
        "total_ops": total_ops,
        "total_rejections": total_rejections,
        "seam_scores_count": len(seam_scores),
        "seam_scores_mean": round(seam_mean, 4),
        "seam_scores_std": round(seam_std, 4),
        "seam_scores_median": round(seam_median, 4),
        "seam_scores_min": round(seam_min, 4),
        "seam_scores_max": round(seam_max, 4),
        "rejected_by_reason": rejected_by_reason,
    }

    # 6. Save summary JSON
    summary_path = qc_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[QC] Summary report generated at {summary_path}")

    # 7. Generate diagnostic plots
    try:
        # Seam score boxplot
        if seam_scores:
            plt.figure(figsize=(6, 4))
            if seam_scores_by_blender and len(seam_scores_by_blender) > 1:
                labels = list(seam_scores_by_blender.keys())
                data = [seam_scores_by_blender[k] for k in labels]
                plt.boxplot(data, tick_labels=labels)
                plt.ylabel("Seam Score")
                plt.title("Seam Score Distribution by Blender")
            else:
                plt.boxplot(seam_scores)
                plt.ylabel("Seam Score")
                plt.title(f"Seam Score Distribution (Mean: {seam_mean:.2f})")
            plt.tight_layout()
            plt.savefig(str(qc_dir / "seam_boxplot.png"), dpi=150)
            plt.close()

        # Rejections bar chart
        if rejected_by_reason:
            plt.figure(figsize=(8, 4))
            reasons = list(rejected_by_reason.keys())
            counts = [rejected_by_reason[r] for r in reasons]
            plt.bar(reasons, counts, color="salmon")
            plt.ylabel("Count")
            plt.title("Rejections by Reason")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(str(qc_dir / "rejections_bar.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"[QC] Warning: Failed to generate plots: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate Quality Control Report")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    args = parser.parse_args()
    generate_report(Path(args.run))


if __name__ == "__main__":
    main()
