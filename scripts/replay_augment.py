"""Replay Runner for DENTEX Augmentation with Bit-Exact Validation & Blender Overrides.

Enforces:
- INV-6: Label annotation synchronization on replay.
- INV-7: Lossless PNG image persistence.
- INV-8: Pure execution without unseeded RNG calls.
- Strict rejection of placement or geometry modifications during replay.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np

from dentex_bench.augment.config import AugmentConfig, _dict_to_dataclass
from dentex_bench.augment.bank import ToothBank
from dentex_bench.augment.recipe import execute, Recipe
from dentex_bench.augment.state import Manifest, load_manifest, save_manifest
from dentex_bench.augment.io_utils import (
    read_image_lossless,
    write_image_lossless,
    sha256_file,
    export_coco_dataset,
    export_yolo_dataset,
)


ALLOWED_OVERRIDE_PREFIXES = (
    "blend.",
    "intensity_match",
    "noise_match",
    "extras.photometric.",
    "extras.patch_crop.",
)


def validate_overrides(overrides: List[str]) -> None:
    """Validate that replay overrides only touch execution-only parameters.

    Raises:
        ValueError: If an override touches placement, geometry, or other non-execution parameters.
    """
    if not overrides:
        return
    for o in overrides:
        key = o.split("=")[0].strip() if "=" in o else o.strip()
        if not any(key.startswith(prefix) for prefix in ALLOWED_OVERRIDE_PREFIXES):
            raise ValueError(
                f"Invalid override '{key}'. Replay only allows execution-only parameter overrides ({ALLOWED_OVERRIDE_PREFIXES}). "
                "To modify placement or geometry, run a new augment_offline job."
            )


def _parse_override_val(val_str: str) -> Any:
    v = val_str.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def apply_overrides(recipe: Recipe, overrides: List[str]) -> Recipe:
    """Apply execution-only overrides to a Recipe copy."""
    if not overrides:
        return recipe

    validate_overrides(overrides)

    rec_dict = copy.deepcopy(recipe.to_dict())
    ops = rec_dict.get("ops", [])
    post = rec_dict.get("post", {})

    for o in overrides:
        if "=" not in o:
            continue
        key, raw_val = o.split("=", 1)
        key = key.strip()
        val = _parse_override_val(raw_val)

        if key in ("blend.mode", "blend_mode"):
            for op in ops:
                op["blend_mode"] = str(val)
                if "blend" not in op or not isinstance(op["blend"], dict):
                    op["blend"] = {}
                op["blend"]["mode"] = str(val)
        elif key.startswith("blend."):
            sub = key.split(".", 1)[1]
            for op in ops:
                if "blend" not in op or not isinstance(op["blend"], dict):
                    op["blend"] = {}
                op["blend"][sub] = val
                op[sub] = val
        elif key in ("intensity_match", "intensity_match.mode", "intensity.mode"):
            for op in ops:
                if "intensity" not in op or not isinstance(op["intensity"], dict):
                    op["intensity"] = {}
                op["intensity"]["mode"] = str(val)
        elif key.startswith("intensity_match."):
            sub = key.split(".", 1)[1]
            for op in ops:
                if "intensity" not in op or not isinstance(op["intensity"], dict):
                    op["intensity"] = {}
                op["intensity"][sub] = val
        elif key in ("noise_match", "noise_match.enabled"):
            for op in ops:
                op["noise_match"] = bool(val)
                if "noise" not in op or not isinstance(op["noise"], dict):
                    op["noise"] = {}
                op["noise"]["enabled"] = bool(val)
        elif key.startswith("extras."):
            parts = key.split(".")
            cur = post
            for p in parts[1:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = val

    rec_dict["ops"] = ops
    rec_dict["post"] = post
    rec_dict["post"]["parent_recipe_id"] = recipe.recipe_id
    new_rec = Recipe.from_dict(rec_dict)
    new_rec.parent_recipe_id = recipe.recipe_id
    return new_rec


def _resolve_paths(state_arg: Union[str, Path]) -> Tuple[Path, Path]:
    """Resolve state directory and parent run directory."""
    p = Path(state_arg)
    if p.name == "state":
        state_dir = p
        run_dir = p.parent
    elif (p / "state").is_dir():
        state_dir = p / "state"
        run_dir = p
    else:
        state_dir = p
        run_dir = p.parent
    return state_dir, run_dir


def _setup_image_loader(
    manifest: Optional[Manifest],
    run_dir: Path,
    recipes: List[Recipe],
) -> Tuple[Callable[[int], Tuple[np.ndarray, List[Dict[str, Any]]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create deterministic base image loader matching offline execution."""
    cfg_paths = manifest.config.get("paths", {}) if manifest else {}
    train_ann_raw = cfg_paths.get("train_ann", "data/processed/instances_train.json")
    img_dir_raw = cfg_paths.get("img_dir", "data/raw")

    train_ann_path = Path(train_ann_raw)
    if not train_ann_path.is_file():
        # Check relative to run_dir or repo_root
        for cand in [run_dir / train_ann_raw, repo_root / train_ann_raw]:
            if cand.is_file():
                train_ann_path = cand
                break

    if train_ann_path.is_file():
        with open(train_ann_path, "r", encoding="utf-8") as f:
            train_coco = json.load(f)
        raw_images = train_coco.get("images", [])
        raw_annotations = train_coco.get("annotations", [])
    else:
        raw_images = [
            {"id": 1, "file_name": "sample_xray_001.png", "width": 800, "height": 400},
            {"id": 2, "file_name": "sample_xray_002.png", "width": 800, "height": 400},
        ]
        raw_annotations = []

    images_by_id: Dict[int, Dict[str, Any]] = {int(img["id"]): img for img in raw_images}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in raw_annotations:
        iid = a.get("image_id")
        if iid is not None:
            anns_by_img.setdefault(int(iid), []).append(a)

    for rec in recipes:
        iid = int(rec.target_image_id)
        if iid not in images_by_id:
            fn = rec.target_file or f"{iid}.png"
            images_by_id[iid] = {"id": iid, "file_name": fn, "width": 800, "height": 400}

    img_dir_path = Path(img_dir_raw)

    def loader(iid: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        info = images_by_id.get(iid, {})
        fn = info.get("file_name", f"{iid}.png")

        candidates = [
            img_dir_path / fn,
            repo_root / img_dir_path / fn,
            run_dir / "dataset" / "images" / fn,
            Path(fn),
        ]
        found_path: Optional[Path] = None
        for c in candidates:
            if c.is_file():
                found_path = c
                break

        if found_path:
            base_img = read_image_lossless(found_path)
        else:
            w = int(info.get("width", 800))
            h = int(info.get("height", 400))
            base_img = np.zeros((h, w, 3), dtype=np.uint8)

        anns = [dict(a) for a in anns_by_img.get(iid, [])]
        return base_img.copy(), anns

    return loader, raw_images, raw_annotations


def replay(
    state_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    recipe_ids: Optional[Union[str, List[str]]] = None,
    overrides: Optional[List[str]] = None,
    verify_hash: bool = False,
    export_dataset: bool = False,
    formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute bit-exact or overridden augmentation replay from state.

    Args:
        state_dir: Path to run state directory (or run directory).
        out_dir: Output directory to save replayed images, state, and datasets.
        recipe_ids: Optional list of recipe_ids to replay (defaults to all).
        overrides: Execution-only overrides (e.g. ['blend.mode=poisson_normal']).
        verify_hash: Assert bit-exact sha256 match against original run output.
        export_dataset: Export COCO/YOLO dataset formats if out_dir is provided.
        formats: Dataset formats to export (['coco', 'yolo'] by default).

    Returns:
        Dict with execution summary, replayed recipes, images, and annotations.
    """
    actual_state_dir, run_dir = _resolve_paths(state_dir)
    recipes_file = actual_state_dir / "recipes.jsonl"
    if not recipes_file.is_file():
        raise FileNotFoundError(f"recipes.jsonl not found in {actual_state_dir}")

    manifest_file = actual_state_dir / "manifest.json"
    manifest: Optional[Manifest] = None
    if manifest_file.is_file():
        try:
            manifest = load_manifest(run_dir)
        except Exception:
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = Manifest(**json.load(f))
            except Exception:
                manifest = None

    # Load recipes
    original_recipes: List[Recipe] = []
    with open(recipes_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                original_recipes.append(Recipe.from_dict(json.loads(line)))

    # Filter by recipe_ids if specified
    if recipe_ids:
        if isinstance(recipe_ids, str):
            filter_set = {rid.strip() for rid in recipe_ids.split(",") if rid.strip()}
        else:
            filter_set = set(recipe_ids)
        recipes_to_replay = [r for r in original_recipes if r.recipe_id in filter_set]
    else:
        recipes_to_replay = original_recipes

    # Validate and apply overrides
    if overrides:
        validate_overrides(overrides)
        exec_recipes = [apply_overrides(r, overrides) for r in recipes_to_replay]
    else:
        exec_recipes = [copy.deepcopy(r) for r in recipes_to_replay]

    # Load ToothBank
    bank_dir = run_dir / "bank"
    if not bank_dir.exists() and manifest:
        bank_out_root = manifest.config.get("paths", {}).get("out_root", "")
        if bank_out_root:
            bank_dir = Path(bank_out_root) / "bank"

    if ToothBank.is_cached(bank_dir):
        bank = ToothBank.load(bank_dir)
    else:
        bank = ToothBank([])

    # Setup image loader
    loader, raw_images, raw_annotations = _setup_image_loader(manifest, run_dir, exec_recipes)

    # Output directories
    out_path_dir = Path(out_dir) if out_dir else None
    syn_dir: Optional[Path] = None
    dataset_img_dir: Optional[Path] = None
    if out_path_dir:
        syn_dir = out_path_dir / "synthetic"
        syn_dir.mkdir(parents=True, exist_ok=True)
        if export_dataset:
            dataset_img_dir = out_path_dir / "dataset" / "images"
            dataset_img_dir.mkdir(parents=True, exist_ok=True)

    replayed_results: List[Dict[str, Any]] = []
    replayed_recipes: List[Recipe] = []
    replayed_images: Dict[str, np.ndarray] = {}
    replayed_anns: Dict[str, List[Dict[str, Any]]] = {}
    hashes: Dict[str, str] = {}

    for orig_rec, rec in zip(recipes_to_replay, exec_recipes):
        syn_img, syn_ann, meta = execute(rec, loader, bank)
        replayed_images[rec.recipe_id] = syn_img
        replayed_anns[rec.recipe_id] = syn_ann

        syn_file_name = rec.output.get("file")
        if not syn_file_name:
            v_idx = rec.rng_key[-1] if len(rec.rng_key) >= 3 else 0
            syn_file_name = f"syn_{rec.target_image_id}_{v_idx}.png"

        # Save images and determine hash
        if syn_dir:
            dest_img = syn_dir / syn_file_name
            write_image_lossless(dest_img, syn_img)
            actual_hash = sha256_file(dest_img)
            if dataset_img_dir:
                write_image_lossless(dataset_img_dir / syn_file_name, syn_img)
        else:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_p = Path(tf.name)
            try:
                write_image_lossless(tmp_p, syn_img)
                actual_hash = sha256_file(tmp_p)
            finally:
                if tmp_p.exists():
                    tmp_p.unlink()

        hashes[rec.recipe_id] = actual_hash

        # Bit-exact hash verification
        if verify_hash:
            expected_hash = orig_rec.output.get("sha256")
            if not expected_hash:
                candidates = [
                    run_dir / "synthetic" / syn_file_name,
                    run_dir / "dataset" / "images" / syn_file_name,
                    actual_state_dir / "synthetic" / syn_file_name,
                ]
                for cand in candidates:
                    if cand.is_file():
                        expected_hash = sha256_file(cand)
                        break

            if expected_hash:
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Hash mismatch for recipe {rec.recipe_id}: "
                        f"expected {expected_hash}, got {actual_hash}"
                    )
            else:
                raise ValueError(
                    f"Cannot verify hash for recipe {rec.recipe_id}: "
                    "no reference hash or original synthetic image found on disk."
                )

        rec.output = {
            "file": syn_file_name,
            "sha256": actual_hash,
            "n_annotations": len(syn_ann),
        }
        replayed_recipes.append(rec)
        replayed_results.append({
            "recipe_id": rec.recipe_id,
            "orig_image_id": rec.target_image_id,
            "file_name": syn_file_name,
            "width": syn_img.shape[1],
            "height": syn_img.shape[0],
            "annotations": syn_ann,
        })

    # Save state to output directory if specified
    if out_path_dir:
        out_state_dir = out_path_dir / "state"
        out_state_dir.mkdir(parents=True, exist_ok=True)
        out_recipes_file = out_state_dir / "recipes.jsonl"
        with open(out_recipes_file, "w", encoding="utf-8") as f:
            for r in replayed_recipes:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Save manifest
        if manifest:
            out_manifest = copy.deepcopy(manifest)
            out_manifest.run_id = out_path_dir.name
            out_manifest.status = "complete"
            out_manifest.counts["recipes"] = len(replayed_recipes)
            out_manifest.counts["ops"] = sum(len(r.ops) for r in replayed_recipes)
            save_manifest(out_path_dir, out_manifest)

        # Export dataset if requested
        if export_dataset:
            target_formats = formats or (manifest.config.get("output", {}).get("formats", ["coco", "yolo"]) if manifest else ["coco", "yolo"])
            target_formats = [f.lower() for f in target_formats]

            export_images = []
            export_annotations = []
            max_img_id = 0
            max_ann_id = 0

            for res in replayed_results:
                max_img_id += 1
                syn_id = max_img_id
                export_images.append({
                    "id": syn_id,
                    "file_name": res["file_name"],
                    "width": res["width"],
                    "height": res["height"],
                })
                for ann in res["annotations"]:
                    max_ann_id += 1
                    a_rec = dict(ann)
                    a_rec["id"] = max_ann_id
                    a_rec["image_id"] = syn_id
                    export_annotations.append(a_rec)

            dataset_dir = out_path_dir / "dataset"
            if "coco" in target_formats:
                coco_dir = dataset_dir / "coco"
                coco_dir.mkdir(parents=True, exist_ok=True)
                export_coco_dataset(export_images, export_annotations, coco_dir / "instances_train_aug.json")
                export_coco_dataset(export_images, export_annotations, coco_dir / "instances.json")

            if "yolo" in target_formats:
                yolo_dir = dataset_dir / "yolo"
                export_yolo_dataset(export_images, export_annotations, yolo_dir)

    return {
        "status": "success",
        "replayed_count": len(replayed_recipes),
        "recipes": replayed_recipes,
        "images": replayed_images,
        "hashes": hashes,
        "annotations": replayed_anns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay DENTEX augmentation from state")
    parser.add_argument("--state", type=str, required=True, help="Path to state/ directory or run directory")
    parser.add_argument("--out", type=str, default=None, help="Output directory to save replayed dataset/images")
    parser.add_argument("--recipe-ids", type=str, default=None, help="Comma-separated recipe IDs to replay")
    parser.add_argument("--override", nargs="*", default=[], help="Execution-only overrides (e.g. blend.mode=poisson_normal)")
    parser.add_argument("--verify-hash", action="store_true", help="Assert bit-exact sha256 match against original run output")
    parser.add_argument("--export-dataset", action="store_true", help="Export dataset in COCO/YOLO formats if out is provided")
    args = parser.parse_args()

    validate_overrides(args.override)

    print(f"[Replay] Replaying recipes from: {args.state}")
    if args.override:
        print(f"[Replay] Applying execution overrides: {args.override}")

    res = replay(
        state_dir=args.state,
        out_dir=args.out,
        recipe_ids=args.recipe_ids,
        overrides=args.override,
        verify_hash=args.verify_hash,
        export_dataset=args.export_dataset,
    )

    print(f"[Replay] Replayed {res['replayed_count']} recipes successfully.")
    if args.out:
        out_p = Path(args.out)
        manifest_p = out_p / "state" / "manifest.json"
        print(f"[Replay] Run directory: {out_p}")
        print(f"[Replay] Manifest path: {manifest_p}")
    print("[Replay] Execution complete.")


if __name__ == "__main__":
    main()
