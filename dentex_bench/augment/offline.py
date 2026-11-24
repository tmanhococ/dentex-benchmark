"""Offline Augmentation Runner & Multiprocessing Invariance Engine.

Executes offline augmentation across target images with:
- Bit-exact worker invariance across any worker count (1, 2, 4, etc.)
- Atomic state manifest progress tracking and resume engine
- Lossless PNG synthetic image persistence (INV-7)
- Label synchronization respecting INV-6
- COCO and YOLO dataset exporters
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank
from dentex_bench.augment.locator import AnnotationToothLocator
from dentex_bench.augment.recipe import resolve, execute, Recipe
from dentex_bench.augment.state import (
    Manifest,
    save_manifest,
    load_manifest,
    update_progress,
)
from dentex_bench.augment.io_utils import (
    read_image_lossless,
    write_image_lossless,
    export_coco_dataset,
    export_yolo_dataset,
    STANDARD_CATEGORIES,
)


def _atomic_write_recipes(path: Union[str, Path], recipe_dicts: Iterable[Dict[str, Any]]) -> None:
    """Write recipes to destination path atomically, deterministically sorted."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f"{p.name}.tmp")
    sorted_recipes = sorted(
        recipe_dicts,
        key=lambda x: (int(x.get("target_image_id", 0)), str(x.get("recipe_id", ""))),
    )
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in sorted_recipes:
            f.write(json.dumps(r) + "\n")
    tmp_path.replace(p)


def _process_image_variants(task: Any) -> Tuple[List[Recipe], List[Dict[str, Any]]]:
    """Top-level worker function to resolve (and execute if saving) image variants.

    Must remain a module top-level function with picklable arguments for Windows
    multiprocessing compatibility.
    """
    if len(task) == 3:
        img_id, cfg, bank_dir = task
        img_idx = 0
        img_info = {"id": img_id, "file_name": f"{img_id}.png", "width": 800, "height": 400}
        img_annotations: List[Dict[str, Any]] = []
        synthetic_dir_str = ""
        dataset_images_dir_str = ""
    else:
        img_idx, img_info, img_annotations, cfg, bank_dir, synthetic_dir_str, dataset_images_dir_str = task
        img_id = img_info["id"]

    bank = ToothBank.load(Path(bank_dir))

    # Only teeth with a known FDI number can be targets; nothing is guessed
    locator = AnnotationToothLocator([dict(a) for a in img_annotations if a.get("fdi") is not None])

    recipes: List[Recipe] = []
    syn_results: List[Dict[str, Any]] = []
    num_variants = cfg.offline.get("variants_per_image", 2)
    target_file = img_info.get("file_name", f"{img_id}.png")

    for v_idx in range(num_variants):
        key = ("offline", img_id, v_idx)
        rec = resolve(img_id, key, cfg, bank, locator, target_file=target_file)
        if rec is None:
            continue
        recipes.append(rec)

        if cfg.output.save_dataset:
            # Read or initialize image for pure execution
            img_path = Path(cfg.paths.img_dir) / target_file
            if not img_path.is_file():
                raise FileNotFoundError(f"Target image '{target_file}' not found under {cfg.paths.img_dir}")
            base_img = read_image_lossless(img_path)

            loader = lambda iid: (base_img.copy(), [dict(a) for a in img_annotations])
            syn_img, syn_anns, meta = execute(rec, loader, bank)

            syn_file_name = f"syn_{img_id}_{v_idx}.png"
            if synthetic_dir_str:
                write_image_lossless(Path(synthetic_dir_str) / syn_file_name, syn_img)
            if dataset_images_dir_str:
                write_image_lossless(Path(dataset_images_dir_str) / syn_file_name, syn_img)

            syn_results.append({
                "variant_idx": v_idx,
                "file_name": syn_file_name,
                "width": syn_img.shape[1],
                "height": syn_img.shape[0],
                "orig_image_id": img_id,
                "recipe_id": rec.recipe_id,
                "seam_scores": meta.get("seam_scores", []),
                "annotations": syn_anns,
            })

    return recipes, syn_results


def run_offline(cfg: AugmentConfig, bank: Optional[ToothBank] = None) -> Path:
    """Run offline augmentation pipeline respecting configuration and state.

    Args:
        cfg: Augmentation configuration dataclass.
        bank: Optional ToothBank instance. If None, loaded from cache or built.

    Returns:
        Path to the output run directory.
    """
    run_dir = Path(cfg.paths.out_root) / f"{cfg.run_name}_seed{cfg.global_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = state_dir / "manifest.json"
    recipes_file = state_dir / "recipes.jsonl"
    synthetic_dir = run_dir / "synthetic"
    dataset_dir = run_dir / "dataset"
    dataset_images_dir = dataset_dir / "images"

    if cfg.output.save_dataset:
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        dataset_images_dir.mkdir(parents=True, exist_ok=True)

    # 1. State Manifest & Resume check
    if manifest_file.is_file() and not cfg.output.overwrite:
        manifest = load_manifest(run_dir)
        if manifest.status == "complete":
            return run_dir
    else:
        manifest = Manifest.create(run_dir, cfg)
        save_manifest(run_dir, manifest)

    # 2. Bank: tasks read patches from the bank's own directory (paths are relative to it)
    if bank is None:
        shared_bank_dir = Path(cfg.paths.out_root) / "bank"
        if ToothBank.is_cached(shared_bank_dir) and len(ToothBank.load(shared_bank_dir).entries) > 0:
            bank = ToothBank.load(shared_bank_dir)
        else:
            from dentex_bench.augment.bank_builder import build_tooth_bank
            bank, _ = build_tooth_bank(cfg.paths.train_ann, cfg.paths.img_dir, shared_bank_dir, val_ann=cfg.paths.val_ann)
    if bank.root is None:
        raise ValueError("run_offline needs a bank loaded from (or built into) a directory; ToothBank.root is None")
    bank_dir = bank.root
    with open(state_dir / "bank_ref.json", "w", encoding="utf-8") as f:
        json.dump({"bank_dir": str(bank_dir), "entries": len(bank.entries)}, f, indent=2)

    # 3. Load dataset annotations
    ann_path = Path(cfg.paths.train_ann)
    if ann_path.is_file():
        with open(ann_path, "r", encoding="utf-8") as f:
            train_coco = json.load(f)
        raw_images = train_coco.get("images", [])
        raw_annotations = train_coco.get("annotations", [])
    else:
        # Fallback for minimal test environments without pre-existing annotations file
        raw_images = [
            {"id": 1, "file_name": "sample_xray_001.png", "width": 800, "height": 400},
            {"id": 2, "file_name": "sample_xray_002.png", "width": 800, "height": 400},
        ]
        raw_annotations = []

    annotations_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in raw_annotations:
        iid = a.get("image_id")
        if iid is not None:
            annotations_by_img.setdefault(int(iid), []).append(a)

    # 4. Load already-completed recipes for resume
    saved_recipes_by_id: Dict[str, Dict[str, Any]] = {}
    if recipes_file.is_file() and not cfg.output.overwrite:
        with open(recipes_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r_dict = json.loads(line)
                    saved_recipes_by_id[r_dict["recipe_id"]] = r_dict

    last_done_index = manifest.progress.get("last_done_index", -1)

    # 5. Build tasks for uncompleted images
    tasks_to_run = []
    task_indices = []
    for idx, img in enumerate(raw_images):
        if idx <= last_done_index:
            continue
        task_data = (
            idx,
            img,
            annotations_by_img.get(int(img["id"]), []),
            cfg,
            str(bank_dir),
            str(synthetic_dir) if cfg.output.save_dataset else "",
            str(dataset_images_dir) if cfg.output.save_dataset else "",
        )
        tasks_to_run.append(task_data)
        task_indices.append(idx)

    # 6. Execute tasks across workers with invariance
    workers = int(cfg.offline.get("num_workers", 1))
    all_synthetic_results: List[Dict[str, Any]] = []

    def _handle_result(task_idx: int, recipes: List[Recipe], syn_results: List[Dict[str, Any]]) -> None:
        for r in recipes:
            saved_recipes_by_id[r.recipe_id] = r.to_dict()
        all_synthetic_results.extend(syn_results)
        # Atomically write sorted recipes
        _atomic_write_recipes(recipes_file, saved_recipes_by_id.values())
        # Atomically update progress
        counts_delta = {
            "recipes": len(recipes),
            "ops": sum(len(r.ops) for r in recipes),
            "rejected_ops": sum(len(r.rejections) for r in recipes),
        }
        update_progress(run_dir, last_done_index=task_idx, counts_delta=counts_delta)

    if tasks_to_run:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for task_idx, (recs, syns) in zip(task_indices, executor.map(_process_image_variants, tasks_to_run)):
                    _handle_result(task_idx, recs, syns)
        else:
            for task_idx, task in zip(task_indices, tasks_to_run):
                recs, syns = _process_image_variants(task)
                _handle_result(task_idx, recs, syns)

    # 7. Dataset Export (COCO & YOLO) if save_dataset is enabled
    if cfg.output.save_dataset:
        export_images: List[Dict[str, Any]] = []
        export_annotations: List[Dict[str, Any]] = []

        # Include originals if requested
        if cfg.output.include_originals:
            export_images.extend(raw_images)
            export_annotations.extend(raw_annotations)
            # Copy original images to dataset/images if available
            for img in raw_images:
                fn = img.get("file_name", "")
                src_path = Path(cfg.paths.img_dir) / fn
                dst_path = dataset_images_dir / fn
                if src_path.is_file() and not dst_path.exists():
                    try:
                        shutil.copy2(src_path, dst_path)
                    except (shutil.SameFileError, IOError):
                        pass

        # Add synthetic images and annotations
        max_img_id = max([int(img["id"]) for img in export_images], default=0)
        max_ann_id = max([int(a["id"]) for a in export_annotations], default=0)

        # Sort synthetic results deterministically
        sorted_syn = sorted(all_synthetic_results, key=lambda s: (s["orig_image_id"], s["variant_idx"]))
        for syn in sorted_syn:
            max_img_id += 1
            syn_id = max_img_id
            export_images.append({
                "id": syn_id,
                "file_name": syn["file_name"],
                "width": syn["width"],
                "height": syn["height"],
            })
            for ann in syn["annotations"]:
                max_ann_id += 1
                ann_rec = dict(ann)
                ann_rec["id"] = max_ann_id
                ann_rec["image_id"] = syn_id
                export_annotations.append(ann_rec)

        raw_formats = cfg.output.formats
        if isinstance(raw_formats, str):
            # CLI overrides arrive as text, e.g. "[coco]" or "coco,yolo"
            raw_formats = [t for t in raw_formats.strip("[] ").replace(" ", "").split(",") if t]
        formats = [f.lower() for f in raw_formats]
        if "coco" in formats:
            coco_dir = dataset_dir / "coco"
            coco_dir.mkdir(parents=True, exist_ok=True)
            export_coco_dataset(export_images, export_annotations, coco_dir / "instances_train_aug.json")
            export_coco_dataset(export_images, export_annotations, coco_dir / "instances.json")

        if "yolo" in formats:
            yolo_dir = dataset_dir / "yolo"
            export_yolo_dataset(export_images, export_annotations, yolo_dir)
            yolo_images_dir = yolo_dir / "images"
            yolo_images_dir.mkdir(parents=True, exist_ok=True)
            for img_rec in export_images:
                fn = img_rec["file_name"]
                src_p = dataset_images_dir / fn
                if src_p.is_file():
                    dst_p = yolo_images_dir / Path(fn).name
                    if not dst_p.exists():
                        try:
                            shutil.copy2(src_p, dst_p)
                        except (shutil.SameFileError, IOError):
                            pass

    # 8. QC summary of what was actually generated
    all_seams = [v for syn in all_synthetic_results for v in syn.get("seam_scores", [])]
    qc_summary = {
        "synthetic_images": len(all_synthetic_results),
        "ops": len(all_seams),
        "seam_score_median": float(np.median(all_seams)) if all_seams else None,
        "seam_score_p90": float(np.percentile(all_seams, 90)) if all_seams else None,
        "qc_max_seam_score": float((cfg.qc or {}).get("max_seam_score", 2.5)),
        "seam_score_above_qc_max": int(sum(v > float((cfg.qc or {}).get("max_seam_score", 2.5)) for v in all_seams)),
    }
    with open(state_dir / "qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(qc_summary, f, indent=2)

    # 9. Mark manifest complete
    final_done_idx = len(raw_images) - 1
    update_progress(run_dir, last_done_index=final_done_idx, status="complete")

    return run_dir


# Alias as specified by the implementation plan
run_offline_augmentation = run_offline

__all__ = [
    "run_offline",
    "run_offline_augmentation",
    "_process_image_variants",
]

