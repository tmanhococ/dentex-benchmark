# tests/augment/test_recipe_determinism.py
import json
import random
import numpy as np
import pytest

from dentex_bench.augment.recipe import resolve, execute, Recipe
from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.locator import AnnotationToothLocator
from dentex_bench.augment.io_utils import write_image_lossless


def test_execute_is_pure_deterministic(monkeypatch):
    recipe = Recipe(
        recipe_id="test_001",
        schema_version=1,
        rng_key=["offline", 10, 0],
        target_image_id=10,
        target_file="sample.png",
        target_file_sha256="",
        ops=[]
    )

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    loader = lambda fid: (dummy_img.copy(), [])
    bank = ToothBank([])

    # Calling execute twice produces bit-exact identical array
    img1, anns1, _ = execute(recipe, loader, bank)
    img2, anns2, _ = execute(recipe, loader, bank)
    np.testing.assert_array_equal(img1, img2)

    # Assert INV-8: execute never calls global random/np.random
    def forbidden_call(*args, **kwargs):
        raise RuntimeError("INV-8 VIOLATION: execute() called RNG!")

    monkeypatch.setattr(np.random, "rand", forbidden_call)
    monkeypatch.setattr(np.random, "randint", forbidden_call)
    execute(recipe, loader, bank)


def test_resolve_reproducibility():
    cfg = AugmentConfig(global_seed=42)
    src_entry = ToothEntry(
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
    bank = ToothBank([src_entry])

    target_anns = [
        {"image_id": 10, "fdi": 16, "category_id": 1, "bbox": [15.0, 15.0, 20.0, 20.0]}
    ]
    locator = AnnotationToothLocator(target_anns)

    # Calling resolve twice with same key yields identical recipe
    r1 = resolve(10, ("offline", 10, 0), cfg, bank, locator, target_file="img_10.png")
    r2 = resolve(10, ("offline", 10, 0), cfg, bank, locator, target_file="img_10.png")

    assert r1 is not None
    assert r2 is not None
    assert r1.recipe_id == r2.recipe_id
    assert r1.ops == r2.ops
    assert r1.target_image_id == 10

    # Calling with different key yields different recipe_id
    r3 = resolve(10, ("offline", 10, 1), cfg, bank, locator, target_file="img_10.png")
    assert r3 is not None
    assert r3.recipe_id != r1.recipe_id


def test_resolve_returns_none_when_no_teeth():
    cfg = AugmentConfig(global_seed=42)
    bank = ToothBank([])
    locator = AnnotationToothLocator([])

    recipe = resolve(10, ("offline", 10, 0), cfg, bank, locator)
    assert recipe is None


def test_execute_with_ops_pure_and_deterministic(monkeypatch):
    src_entry = ToothEntry(
        entry_id="bank_1",
        image_id=99,
        file_name="99.png",
        fdi=16,
        is_upper=True,
        labels=["deep_caries"],
        is_healthy=False,
        roi_class="deep_caries",
        bbox_xywh=[10.0, 10.0, 20.0, 20.0],
        roi_xywh=[5, 5, 30, 30],
        annotations=[{"category_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0]}],
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
    bank = ToothBank([src_entry])

    recipe = Recipe(
        recipe_id="rec_10_0",
        schema_version=1,
        rng_key=["offline", 10, 0],
        target_image_id=10,
        target_file="img_10.png",
        target_file_sha256="",
        ops=[
            {
                "op": "swap",
                "source_entry_id": "bank_1",
                "target_fdi": 16,
                "target_tooth_ref": {
                    "source": "annotations",
                    "bbox": [20.0, 20.0, 22.0, 22.0],
                    "has_mask": False,
                },
                "mirror": False,
                "scale_jitter": 1.02,
                "rotate_jitter": 1.5,
                "blend_mode": "core_preserving",
                "noise_seed": 123456,
            }
        ],
    )

    dummy_img = np.full((120, 120, 3), 100, dtype=np.uint8)
    orig_anns = [
        {"id": 1, "image_id": 10, "fdi": 16, "category_id": 1, "bbox": [20.0, 20.0, 22.0, 22.0]},
        {"id": 2, "image_id": 10, "fdi": 48, "category_id": 4, "bbox": [80.0, 80.0, 20.0, 20.0]},
    ]
    loader = lambda fid: (dummy_img.copy(), list(orig_anns))

    # Assert INV-8: execute never calls any RNG
    def forbidden(*args, **kwargs):
        raise RuntimeError("INV-8 VIOLATION: execute() called RNG!")

    for method in ["rand", "randn", "randint", "random", "uniform", "choice"]:
        if hasattr(np.random, method):
            monkeypatch.setattr(np.random, method, forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(random, "randint", forbidden)

    out_img1, out_anns1, meta1 = execute(recipe, loader, bank)
    out_img2, out_anns2, meta2 = execute(recipe, loader, bank)

    np.testing.assert_array_equal(out_img1, out_img2)
    assert out_anns1 == out_anns2
    assert meta1["status"] == "success"

    # INV-6: original target FDI 16 annotation deleted, bank annotation added with synthetic=True
    fdis = [a["fdi"] for a in out_anns1]
    assert 16 in fdis
    assert 48 in fdis
    ann_16 = [a for a in out_anns1 if a["fdi"] == 16][0]
    assert ann_16["category_id"] == 2
    assert ann_16["synthetic"] is True


def test_resolve_blend_mode_random_choice():
    cfg = AugmentConfig(global_seed=123)
    cfg.blend.mode = "random_choice"
    cfg.blend.choices = ["hard", "feather"]

    src_entry = ToothEntry(
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
    bank = ToothBank([src_entry])
    target_anns = [{"image_id": 10, "fdi": 16, "category_id": 1, "bbox": [15.0, 15.0, 20.0, 20.0]}]
    locator = AnnotationToothLocator(target_anns)

    r = resolve(10, ("offline", 10, 0), cfg, bank, locator)
    assert r is not None
    assert r.ops[0]["blend_mode"] in ["hard", "feather"]


def test_execute_with_saved_patch_file(tmp_path):
    # Create saved patch (60x60) and mask (60x60)
    patch_img = np.full((60, 60, 3), 200, dtype=np.uint8)
    mask_img = np.zeros((60, 60), dtype=np.uint8)
    mask_img[10:50, 10:50] = 255

    patch_path = str(tmp_path / "patch.png")
    mask_path = str(tmp_path / "mask.png")
    write_image_lossless(patch_path, patch_img)
    write_image_lossless(mask_path, mask_img)

    src_entry = ToothEntry(
        entry_id="bank_saved",
        image_id=99,
        file_name="99.png",
        fdi=16,
        is_upper=True,
        labels=["periapical_lesion"],
        is_healthy=False,
        roi_class="periapical",
        bbox_xywh=[10.0, 10.0, 40.0, 40.0],
        roi_xywh=[0, 0, 60, 60],
        annotations=[{"category_id": 3, "bbox": [10.0, 10.0, 40.0, 40.0]}],
        axis_angle_deg=0.0,
        mask_area_px=1600,
        mask_h=40.0,
        mask_w=40.0,
        ring_stats={},
        noise_sigma=1.0,
        patch_path=patch_path,
        mask_path=mask_path,
        roi_mask_path=mask_path,
        mask_source="polygon",
    )
    bank = ToothBank([src_entry])

    # Target image is 200x200 filled with 50
    # Target tooth bbox is at [80.0, 80.0, 40.0, 40.0], exactly matching 40x40 source tooth
    recipe = Recipe(
        recipe_id="rec_patch_test",
        schema_version=1,
        rng_key=["offline", 10, 0],
        target_image_id=10,
        target_file="target.png",
        target_file_sha256="",
        ops=[
            {
                "op": "swap",
                "source_entry_id": "bank_saved",
                "target_fdi": 16,
                "target_tooth_ref": {
                    "source": "annotations",
                    "bbox": [80.0, 80.0, 40.0, 40.0],
                    "has_mask": True,
                },
                "mirror": False,
                "scale_jitter": 1.0,
                "rotate_jitter": 0.0,
                "blend_mode": "hard",
                "noise_seed": 42,
                "intensity": {"mode": "meanstd", "a": 1.0, "b": 10.0},
            }
        ],
    )

    target_img = np.full((200, 200, 3), 50, dtype=np.uint8)
    target_anns = [
        {"id": 1, "image_id": 10, "fdi": 16, "category_id": 1, "bbox": [80.0, 80.0, 40.0, 40.0]}
    ]
    loader = lambda fid: (target_img.copy(), list(target_anns))

    out_img, out_anns, meta = execute(recipe, loader, bank)
    assert meta["status"] == "success"

    # Verify intensity match: 200 * 1.0 + 10 = 210
    # Center of target tooth is at (100, 100)
    assert np.all(out_img[100, 100] == 210)
    # Outside target tooth remains background 50
    assert np.all(out_img[10, 10] == 50)

    # Transformed annotation aligns with [80.0, 80.0, 40.0, 40.0]
    assert len(out_anns) == 1
    ann = out_anns[0]
    assert ann["fdi"] == 16
    assert ann["category_id"] == 3
    assert ann["synthetic"] is True
    np.testing.assert_allclose(ann["bbox"], [80.0, 80.0, 40.0, 40.0], atol=1e-3)

