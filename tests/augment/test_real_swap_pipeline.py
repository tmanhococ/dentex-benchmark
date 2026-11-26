"""End-to-end checks of the tooth-swap pipeline on DENTEX-format data with real polygons.

Guards the 2026-11 bug where every swap silently became a gray square on an inpainted box.
"""

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dentex_bench.augment.bank import ToothBank
from dentex_bench.augment.bank_builder import build_tooth_bank
from dentex_bench.augment.config import load_config
from dentex_bench.augment.io_utils import read_image_lossless, write_image_lossless
from dentex_bench.augment.offline import run_offline
from dentex_bench.augment.recipe import Recipe, execute
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

W, H = 640, 320
# (quadrant id, tooth id) as DENTEX stores them: 0-based ids whose names are the clinical numbers
TEETH = {(0, 5): (150, 60), (2, 5): (470, 210)}  # FDI 16 (upper right), FDI 36 (lower left)


def _tooth_polygon(cx, cy, w=40, h=80):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]


def _make_dentex_dataset(root: Path, n_images: int = 6) -> Path:
    """DENTEX hierarchical json + images whose teeth have a unique brightness per image."""
    img_dir = root / "xrays"
    img_dir.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    ann_id = 1
    for i in range(n_images):
        img = np.full((H, W), 70, dtype=np.uint8)
        for (q, t), (cx, cy) in TEETH.items():
            poly = _tooth_polygon(cx, cy)
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(img, [pts], 120 + 20 * i)
            # Two diagnoses on the same tooth for the first image
            for cat3 in ([1, 2] if i == 0 and q == 0 else [1]):
                x1, y1 = min(poly[0::2]), min(poly[1::2])
                anns.append({
                    "id": ann_id, "image_id": i + 1, "iscrowd": 0,
                    "category_id_1": q, "category_id_2": t, "category_id_3": cat3,
                    "bbox": [x1, y1, 40, 80], "area": 3200, "segmentation": [poly],
                })
                ann_id += 1
        cv2.imwrite(str(img_dir / f"train_{i}.png"), img)
        images.append({"id": i + 1, "file_name": f"train_{i}.png", "width": W, "height": H})
    data = {
        "images": images,
        "annotations": anns,
        "categories_1": [{"id": k, "name": str(k + 1)} for k in range(4)],
        "categories_2": [{"id": k, "name": str(k + 1)} for k in range(8)],
        "categories_3": [{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"},
                         {"id": 2, "name": "Periapical Lesion"}, {"id": 3, "name": "Deep Caries"}],
    }
    raw = root / "train_quadrant_enumeration_disease.json"
    raw.write_text(json.dumps(data), encoding="utf-8")
    return raw


def test_parser_keeps_exact_fdi_and_polygons(tmp_path):
    raw = _make_dentex_dataset(tmp_path, n_images=2)
    convert_dentex_annotations_to_coco(raw, tmp_path / "t.json", tmp_path / "v.json", split_ratio=1.0)
    coco = json.loads((tmp_path / "t.json").read_text())
    fdis = sorted({a["fdi"] for a in coco["annotations"]})
    assert fdis == [16, 36]
    assert all(a.get("segmentation") for a in coco["annotations"])
    # categories_3 id 2 is "Periapical Lesion" -> class 3
    assert {a["category_id"] for a in coco["annotations"]} == {1, 3}


def test_bank_groups_diagnoses_per_tooth_with_real_outline(tmp_path):
    raw = _make_dentex_dataset(tmp_path, n_images=2)
    convert_dentex_annotations_to_coco(raw, tmp_path / "t.json", tmp_path / "v.json", split_ratio=1.0)
    bank, report = build_tooth_bank(tmp_path / "t.json", tmp_path / "xrays", tmp_path / "bank")
    assert report["teeth_in_bank"] == 4  # 2 images x 2 teeth, the double diagnosis merged
    entry = bank.get("1_16")
    assert entry is not None and len(entry.annotations) == 2
    assert entry.roi_class == "periapical"
    mask = read_image_lossless(bank.resolve_path(entry.mask_path))
    patch = read_image_lossless(bank.resolve_path(entry.patch_path))
    # The tooth mask covers exactly the bright tooth in the patch
    assert patch[mask > 0].min() >= 115
    # Paths resolve against the bank directory from any working directory
    reloaded = ToothBank.load(tmp_path / "bank")
    assert reloaded.resolve_path(entry.patch_path).is_file()


def test_execute_raises_when_patch_missing(tmp_path):
    raw = _make_dentex_dataset(tmp_path, n_images=2)
    convert_dentex_annotations_to_coco(raw, tmp_path / "t.json", tmp_path / "v.json", split_ratio=1.0)
    bank, _ = build_tooth_bank(tmp_path / "t.json", tmp_path / "xrays", tmp_path / "bank")
    entry = bank.get("1_16")
    broken = ToothBank([dataclasses.replace(entry, patch_path="patches/missing.png")], root=bank.root)
    recipe = Recipe(
        recipe_id="r", schema_version=1, rng_key=[], target_image_id=2, target_file="train_1.png",
        target_file_sha256="", ops=[{"op": "swap", "source_entry_id": "1_16", "target_fdi": 16,
                                     "target_tooth_ref": {"bbox": [130, 20, 40, 80]}, "mirror": False}],
    )
    img = np.full((H, W), 70, dtype=np.uint8)
    with pytest.raises(FileNotFoundError):
        execute(recipe, lambda i: (img.copy(), []), broken)


def test_offline_swap_pastes_the_source_tooth_not_a_placeholder(tmp_path):
    raw = _make_dentex_dataset(tmp_path, n_images=6)
    train, val = tmp_path / "processed" / "t.json", tmp_path / "processed" / "v.json"
    convert_dentex_annotations_to_coco(raw, train, val, split_ratio=1.0)
    cfg = load_config("configs/augment/base.yaml", cli_overrides=[
        f"paths.train_ann={train}", f"paths.val_ann={val}", f"paths.img_dir={tmp_path / 'xrays'}",
        f"paths.out_root={tmp_path / 'out'}", "offline.variants_per_image=1", "output.formats=[coco]",
        "placement.min_ops=1", "placement.max_ops=1", "geometry.jitter_scale=0", "geometry.jitter_rotate_deg=0",
    ])
    run_dir = run_offline(cfg)
    coco = json.loads((run_dir / "dataset" / "coco" / "instances_train_aug.json").read_text())
    recipes = [json.loads(l) for l in (run_dir / "state" / "recipes.jsonl").read_text().splitlines()]
    assert len(recipes) == 6

    for rec in recipes:
        op = rec["ops"][0]
        src_img_id = int(op["source_entry_id"].split("_")[0])
        assert src_img_id != rec["target_image_id"]
        syn = read_image_lossless(run_dir / "synthetic" / f"syn_{rec['target_image_id']}_0.png")
        orig = read_image_lossless(tmp_path / "xrays" / rec["target_file"])
        x, y, w, h = [int(v) for v in op["target_tooth_ref"]["bbox"]]
        core = syn[y + 15:y + h - 15, x + 10:x + w - 10]
        # Background ring is identical in source and target, so the pasted tooth keeps the
        # source tooth brightness (120 + 20 * i), not the target's and never a flat 180 square
        expected = 120 + 20 * (src_img_id - 1)
        assert abs(float(core.mean()) - expected) <= 3, (rec["recipe_id"], core.mean(), expected)
        far = (slice(0, 20), slice(0, 20))
        assert np.array_equal(syn[far], orig[far])

    syn_anns = [a for a in coco["annotations"] if a.get("synthetic")]
    assert syn_anns and all(a["fdi"] in (16, 36) and a.get("segmentation") for a in syn_anns)
    summary = json.loads((run_dir / "state" / "qc_summary.json").read_text())
    assert summary["ops"] == 6


def test_offline_resume_after_interruption_exports_every_synthetic_image(tmp_path):
    raw = _make_dentex_dataset(tmp_path, n_images=6)
    train, val = tmp_path / "processed" / "t.json", tmp_path / "processed" / "v.json"
    convert_dentex_annotations_to_coco(raw, train, val, split_ratio=1.0)
    overrides = [
        f"paths.train_ann={train}", f"paths.val_ann={val}", f"paths.img_dir={tmp_path / 'xrays'}",
        f"paths.out_root={tmp_path / 'out'}", "offline.variants_per_image=1", "output.formats=[coco]",
    ]
    run_dir = run_offline(load_config("configs/augment/base.yaml", cli_overrides=overrides))
    full = json.loads((run_dir / "dataset" / "coco" / "instances_train_aug.json").read_text())

    # Simulate a kernel dying after image 3: manifest says running, last done index 2
    manifest_path = run_dir / "state" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest["progress"]["last_done_index"] = 2
    manifest_path.write_text(json.dumps(manifest))
    (run_dir / "dataset" / "coco" / "instances_train_aug.json").unlink()

    run_dir = run_offline(load_config("configs/augment/base.yaml", cli_overrides=overrides))
    resumed = json.loads((run_dir / "dataset" / "coco" / "instances_train_aug.json").read_text())
    assert resumed["images"] == full["images"]
    assert resumed["annotations"] == full["annotations"]
    assert json.loads((run_dir / "state" / "qc_summary.json").read_text())["synthetic_images"] == 6
