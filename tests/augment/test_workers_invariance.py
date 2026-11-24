# tests/augment/test_workers_invariance.py
import json
from pathlib import Path
import subprocess
import sys
import pytest

from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from real_bank import make_real_bank, write_images_for_ann
from dentex_bench.augment.offline import run_offline


def _make_test_bank() -> ToothBank:
    entry = ToothEntry(
        entry_id="bank_1",
        image_id=99,
        file_name="99.png",
        fdi=16,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[10.0, 10.0, 20.0, 20.0],
        roi_xywh=[5, 5, 30, 30],
        annotations=[{"category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}],
        axis_angle_deg=0.0,
        mask_area_px=400,
        mask_h=20.0,
        mask_w=20.0,
        ring_stats={},
        noise_sigma=1.0,
        patch_path="",
        mask_path="",
        roi_mask_path="",
        mask_source="polygon",
    )
    return make_real_bank([entry])


def _make_test_ann_file(path: Path, num_images: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for i in range(1, num_images + 1):
        images.append({"id": i, "file_name": f"img_{i}.png", "width": 800, "height": 400})
        annotations.append({
            "id": i,
            "image_id": i,
            "category_id": 1,
            "fdi": 16,
            "bbox": [100.0, 100.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        })
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "Caries"}]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)


def test_multiprocessing_invariance_num_workers(tmp_path):
    """Running with workers=1 vs workers=2 must produce identical recipes."""
    cfg1 = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_w1'}",
            "offline.num_workers=1",
            "output.save_dataset=false",
        ],
    )
    cfg2 = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_w2'}",
            "offline.num_workers=2",
            "output.save_dataset=false",
        ],
    )
    run_offline(cfg1)
    run_offline(cfg2)

    rec1 = (tmp_path / "out_w1" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()
    rec2 = (tmp_path / "out_w2" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()
    assert sorted(rec1) == sorted(rec2)


def test_multiprocessing_invariance_with_populated_recipes(tmp_path):
    """Running with workers=1 vs workers=4 on real recipes produces bit-exact, identical recipes in order."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=6)
    bank = _make_test_bank()

    cfg1 = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_pop_w1'}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=1",
            "offline.variants_per_image=2",
            "output.save_dataset=false",
        ],
    )
    cfg2 = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_pop_w4'}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=4",
            "offline.variants_per_image=2",
            "output.save_dataset=false",
        ],
    )

    run_offline(cfg1, bank=bank)
    run_offline(cfg2, bank=bank)

    rec1 = (tmp_path / "out_pop_w1" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()
    rec2 = (tmp_path / "out_pop_w4" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()

    # Must have generated actual recipes
    assert len(rec1) == 12  # 6 images * 2 variants
    # Worker invariance: lines must be identical and in exact deterministic order
    assert rec1 == rec2


def test_offline_save_dataset_coco_and_yolo(tmp_path):
    """When save_dataset is True, offline runner exports lossless PNGs, COCO json, and YOLO labels."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=3)
    bank = _make_test_bank()

    out_root = tmp_path / "out_dataset"
    cfg = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={out_root}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=2",
            "offline.variants_per_image=1",
            "output.save_dataset=true",
            "output.include_originals=true",
        ],
    )

    run_offline(cfg, bank=bank)

    run_dir = out_root / "base_seed0"

    # Synthetic lossless PNGs exist
    synthetic_dir = run_dir / "synthetic"
    assert synthetic_dir.exists()
    syn_pngs = list(synthetic_dir.glob("*.png"))
    assert len(syn_pngs) == 3

    # Dataset images directory exists
    dataset_img_dir = run_dir / "dataset" / "images"
    assert dataset_img_dir.exists()
    assert len(list(dataset_img_dir.glob("syn_*.png"))) == 3

    # COCO dataset exported
    coco_file = run_dir / "dataset" / "coco" / "instances_train_aug.json"
    assert coco_file.exists()
    coco_data = json.loads(coco_file.read_text(encoding="utf-8"))
    assert len(coco_data["images"]) == 6  # 3 original + 3 synthetic
    assert any(a.get("synthetic") is True for a in coco_data["annotations"])

    # YOLO dataset exported
    yolo_dir = run_dir / "dataset" / "yolo"
    assert (yolo_dir / "data.yaml").exists()
    yolo_labels = list((yolo_dir / "labels").glob("*.txt"))
    assert len(yolo_labels) == 6


def test_augment_offline_cli_help():
    """Verify CLI help for scripts/augment_offline.py."""
    cmd = [sys.executable, "scripts/augment_offline.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--config" in res.stdout
    assert "--force" in res.stdout
    assert "--set" in res.stdout


def test_augment_offline_cli_execution(tmp_path):
    """Verify CLI execution of scripts/augment_offline.py including --force."""
    out_root = tmp_path / "out_cli"
    cmd = [
        sys.executable,
        "scripts/augment_offline.py",
        "--config",
        "configs/augment/base.yaml",
        "--set",
        f"paths.out_root={out_root}",
        "offline.num_workers=1",
        "output.save_dataset=false",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "[Offline] Augmentation complete" in res.stdout
    assert (out_root / "base_seed0" / "state" / "manifest.json").exists()

    # Test with --force flag
    cmd_force = cmd + ["--force"]
    res_force = subprocess.run(cmd_force, capture_output=True, text=True)
    assert res_force.returncode == 0
    assert "[Offline] Augmentation complete" in res_force.stdout
