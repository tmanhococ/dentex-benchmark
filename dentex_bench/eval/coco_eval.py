# dentex_bench/eval/coco_eval.py
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _to_list(val: Any) -> list:
    """Helper to convert tensor/numpy/iterable to a python list."""
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def evaluate_coco_predictions(ground_truth_json: Union[str, Path], predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluates predictions against COCO ground truth annotations.

    Args:
        ground_truth_json: Path to ground truth COCO json file.
        predictions: List of prediction dictionaries, where each element contains:
            - "image_id": int
            - "boxes": tensor or array of shape (N, 4) in [x1, y1, x2, y2] format
            - "scores": tensor or array of shape (N,)
            - "labels": tensor or array of shape (N,)

    Returns:
        Dictionary with mAP50, mAP50_95, and per-class AP scores.
    """
    default_metrics = {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "AP_Caries": 0.0,
        "AP_DeepCaries": 0.0,
        "AP_PeriapicalLesion": 0.0,
        "AP_ImpactedTooth": 0.0
    }

    if not predictions:
        return default_metrics

    # Format predictions to COCO detection format: [x, y, width, height]
    coco_dt = []
    for pred in predictions:
        img_id = pred.get("image_id", 0)
        boxes = _to_list(pred.get("boxes", []))
        scores = _to_list(pred.get("scores", []))
        labels = _to_list(pred.get("labels", []))

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            coco_dt.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

    if not coco_dt:
        return default_metrics

    coco_gt = COCO(str(ground_truth_json))
    coco_res = coco_gt.loadRes(coco_dt)
    coco_eval = COCOeval(coco_gt, coco_res, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    mAP50_95 = max(0.0, float(stats[0]))
    mAP50 = max(0.0, float(stats[1]))

    # Per-class AP at IoU=0.50 for standard 4 DENTEX abnormal classes
    class_names = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]
    per_class = {}
    for cid, cname in enumerate(class_names, start=1):
        if not coco_gt.getAnnIds(catIds=[cid]):
            # No lesion of this class in the ground truth: AP is undefined, not 0
            per_class[f"AP_{cname.replace(' ', '')}"] = float("nan")
            continue
        cat_idx = coco_eval.params.catIds.index(cid)
        # precisions shape: (T, R, K, A, M) where T is IoU threshold (0 is 0.50)
        prec = coco_eval.eval["precision"][0, :, cat_idx, 0, -1]
        prec = prec[prec > -1]
        ap = float(prec.mean()) if len(prec) > 0 else 0.0
        per_class[f"AP_{cname.replace(' ', '')}"] = round(max(0.0, ap), 4)

    res = {
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        **per_class
    }
    return res


def predict_dataset(runner, ann_file: Union[str, Path], img_dir: Union[str, Path]) -> tuple:
    """
    Runs the runner over every annotated image at original resolution.
    Returns (predictions with boxes in original pixel coordinates, per-image latencies in ms).
    """
    import time
    from dentex_bench.data.dataset import DentexDataset

    dataset = DentexDataset(img_dir=img_dir, ann_file=ann_file, resize=False)
    predictions, latencies = [], []
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = runner.predict([img])[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
        pred["image_id"] = int(target["image_id"].item())
        predictions.append(pred)
    return predictions, latencies


def evaluate_runner(runner, ann_file: Union[str, Path], img_dir: Union[str, Path]) -> Dict[str, float]:
    """COCO metrics of a runner on an annotation file, plus its median per-image latency."""
    predictions, latencies = predict_dataset(runner, ann_file, img_dir)
    metrics = evaluate_coco_predictions(ann_file, predictions)
    if latencies:
        # The first call includes CUDA/cuDNN warm-up, so report the median
        metrics["latency_ms"] = round(float(sorted(latencies)[len(latencies) // 2]), 1)
    return metrics


# Interface alias
evaluate_coco = evaluate_coco_predictions
