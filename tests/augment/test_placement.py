import numpy as np
import pytest

from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.locator import LocatedTooth, AnnotationToothLocator
from dentex_bench.augment.placement import (
    generate_swap_plan,
    SwapOp,
    SwapPlan,
    check_scale_rule,
    check_angle_rule,
    check_border_rule,
)
from dentex_bench.augment.rng import rng_for


def _make_dummy_entry(
    entry_id: str = "1_16",
    image_id: int = 1,
    fdi: int = 16,
    roi_class: str = "caries",
    is_healthy: bool = False,
    bbox_xywh: list = None,
    axis_angle_deg: float = 0.0,
) -> ToothEntry:
    if bbox_xywh is None:
        bbox_xywh = [10.0, 20.0, 30.0, 40.0]
    return ToothEntry(
        entry_id=entry_id,
        image_id=image_id,
        file_name=f"{image_id}.png",
        fdi=fdi,
        is_upper=(fdi // 10 in (1, 2)),
        labels=[roi_class],
        is_healthy=is_healthy,
        roi_class=roi_class,
        bbox_xywh=bbox_xywh,
        roi_xywh=[0, 10, 50, 60],
        annotations=[],
        axis_angle_deg=axis_angle_deg,
        mask_area_px=1000,
        mask_h=float(bbox_xywh[3]),
        mask_w=float(bbox_xywh[2]),
        ring_stats={},
        noise_sigma=1.0,
        patch_path="",
        mask_path="",
        roi_mask_path="",
        mask_source="polygon",
    )


def test_placement_rejects_impacted_on_non_8():
    cfg = load_config("configs/augment/base.yaml")
    # Entry is impacted on tooth 48
    entry_impacted = ToothEntry(
        entry_id="1_48",
        image_id=1,
        file_name="1.png",
        fdi=48,
        is_upper=False,
        labels=["impacted_tooth"],
        is_healthy=False,
        roi_class="impacted",
        bbox_xywh=[10, 20, 30, 40],
        roi_xywh=[0, 10, 50, 60],
        annotations=[],
        axis_angle_deg=0.0,
        mask_area_px=1000,
        mask_h=40.0,
        mask_w=30.0,
        ring_stats={},
        noise_sigma=1.0,
        patch_path="",
        mask_path="",
        roi_mask_path="",
        mask_source="polygon",
    )
    bank = ToothBank([entry_impacted])
    # Target tooth is tooth 36 (not an 8)
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 3, "category_id_2": 6, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_impacted")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    # Plan must contain 0 ops due to R_IMPACTED rejection
    assert len(plan.ops) == 0
    assert any(r["code"] == "R_IMPACTED" for r in plan.rejections)


def test_placement_valid_fdi_swap_same_quadrant():
    cfg = load_config("configs/augment/base.yaml")
    entry = _make_dummy_entry(entry_id="1_16", image_id=1, fdi=16, roi_class="caries")
    bank = ToothBank([entry])
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 1, "category_id_2": 6, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_valid_swap")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    assert len(plan.ops) >= 1
    op = plan.ops[0]
    assert op.source_entry_id == "1_16"
    assert op.target_image_id == 2
    assert op.target_fdi == 16
    assert op.mirror is False
    assert op.target_tooth_ref["source"] == "annotations"


def test_placement_valid_fdi_swap_mirrored():
    cfg = load_config("configs/augment/base.yaml")
    entry = _make_dummy_entry(entry_id="1_16", image_id=1, fdi=16, roi_class="caries")
    bank = ToothBank([entry])
    # Target tooth is FDI 26 (contralateral quadrant in upper jaw)
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 2, "category_id_2": 6, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_mirror_swap")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    assert len(plan.ops) >= 1
    op = plan.ops[0]
    assert op.target_fdi == 26
    assert op.mirror is True


def test_placement_disallow_mirror_rejects():
    cfg = load_config("configs/augment/base.yaml")
    cfg.placement.allow_mirror = False
    entry = _make_dummy_entry(entry_id="1_16", image_id=1, fdi=16, roi_class="caries")
    bank = ToothBank([entry])
    # Target tooth is FDI 26 (requires mirror, but mirror disallowed)
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 2, "category_id_2": 6, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_disallow_mirror")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    assert len(plan.ops) == 0
    assert any(r["code"] == "R_FDI_MISMATCH" for r in plan.rejections)


def test_placement_rejects_same_image():
    cfg = load_config("configs/augment/base.yaml")
    entry = _make_dummy_entry(entry_id="2_16", image_id=2, fdi=16, roi_class="caries")
    bank = ToothBank([entry])
    # Target is also image 2
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 1, "category_id_2": 6, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_same_image")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    assert len(plan.ops) == 0
    assert any(r["code"] == "R_SAME_IMAGE" for r in plan.rejections)


def test_placement_symmetric_allows_same_image():
    cfg = load_config("configs/augment/base.yaml")
    cfg.placement.mode = "symmetric"
    entry = _make_dummy_entry(entry_id="2_16", image_id=2, fdi=16, roi_class="caries")
    bank = ToothBank([entry])
    # Target is same image 2, contralateral tooth 26
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 2, "category_id_2": 6, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_symmetric")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    assert len(plan.ops) >= 1
    assert plan.ops[0].mirror is True


def test_placement_no_target_teeth():
    cfg = load_config("configs/augment/base.yaml")
    entry = _make_dummy_entry(entry_id="1_16", image_id=1, fdi=16)
    bank = ToothBank([entry])
    locator = AnnotationToothLocator([])
    rng = rng_for(42, "test_no_teeth")
    plan = generate_swap_plan(999, cfg, bank, locator, rng)
    assert len(plan.ops) == 0
    assert len(plan.rejections) == 0


def test_placement_impacted_on_tooth_8_succeeds():
    cfg = load_config("configs/augment/base.yaml")
    entry_impacted = _make_dummy_entry(
        entry_id="1_48", image_id=1, fdi=48, roi_class="impacted"
    )
    bank = ToothBank([entry_impacted])
    # Target tooth is tooth 38 (lower tooth 8, contralateral)
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 3, "category_id_2": 8, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_impacted_8")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    assert len(plan.ops) >= 1
    assert plan.ops[0].target_fdi == 38
    assert plan.ops[0].mirror is True


def test_rejection_rule_helpers():
    # Test check_scale_rule
    assert check_scale_rule([0, 0, 100, 100], [0, 0, 100, 100], 0.85, 1.15) is True
    assert check_scale_rule([0, 0, 100, 100], [0, 0, 200, 200], 0.85, 1.15) is False

    # Test check_angle_rule
    assert check_angle_rule(5.0, 10.0, 10.0) is True
    assert check_angle_rule(5.0, 25.0, 10.0) is False

    # Test check_border_rule
    # Bbox fully inside 1000x1000 image
    assert check_border_rule([100, 100, 50, 50], 1000, 1000, 0.1) is True
    # Bbox extending 50% outside image
    assert check_border_rule([975, 100, 50, 50], 1000, 1000, 0.1) is False
