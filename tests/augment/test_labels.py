# tests/augment/test_labels.py
import numpy as np
import pytest
from dentex_bench.augment.bank import ToothEntry
from dentex_bench.augment.placement import SwapOp
from dentex_bench.augment.labels import update_annotations_for_swap


def test_inv6_label_replacement():
    # Target has an existing annotation on tooth FDI 36
    target_anns = [
        {"id": 1, "image_id": 10, "fdi": 36, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 10, "fdi": 48, "category_id": 4, "bbox": [80, 80, 30, 30]},
    ]

    op = SwapOp(
        op="swap",
        source_entry_id="99_36",
        target_image_id=10,
        target_fdi=36,
        target_tooth_ref={"source": "annotations", "bbox": [10, 10, 20, 20], "has_mask": True},
        mirror=False,
    )

    src_entry = ToothEntry(
        entry_id="99_36",
        image_id=99,
        file_name="99.png",
        fdi=36,
        is_upper=False,
        labels=["periapical_lesion"],
        is_healthy=False,
        roi_class="periapical",
        bbox_xywh=[0, 0, 25, 25],
        roi_xywh=[0, 0, 35, 35],
        annotations=[{"category_id": 3, "bbox": [5.0, 5.0, 15.0, 15.0]}],
        axis_angle_deg=0.0,
        mask_area_px=500,
        mask_h=25.0,
        mask_w=25.0,
        ring_stats={},
        noise_sigma=1.0,
        patch_path="",
        mask_path="",
        roi_mask_path="",
        mask_source="polygon",
    )

    # Simple identity affine
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    updated = update_annotations_for_swap(target_anns, op, src_entry, M, 200, 200)

    # Tooth 36's old annotation (category 1) MUST be deleted (INV-6)
    # Tooth 48's annotation MUST remain untouched
    # Tooth 36's new annotation (category 3) from source MUST be added
    assert len(updated) == 2
    fdis = [a["fdi"] for a in updated]
    assert 36 in fdis and 48 in fdis
    new_36_ann = [a for a in updated if a["fdi"] == 36][0]
    assert new_36_ann["category_id"] == 3
    assert new_36_ann.get("synthetic") is True
    assert new_36_ann["category_id_1"] == 3  # Quadrant 3
    assert new_36_ann["category_id_2"] == 6  # Tooth 6


def test_erase_disease_and_paste_negative():
    target_anns = [
        {"id": 1, "image_id": 10, "fdi": 36, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 10, "fdi": 48, "category_id": 4, "bbox": [80, 80, 30, 30]},
    ]
    src_entry = ToothEntry(
        entry_id="99_36",
        image_id=99,
        file_name="99.png",
        fdi=36,
        is_upper=False,
        labels=["healthy"],
        is_healthy=True,
        roi_class=None,
        bbox_xywh=[0, 0, 25, 25],
        roi_xywh=[0, 0, 35, 35],
        annotations=[{"category_id": 0, "bbox": [5.0, 5.0, 15.0, 15.0]}],
        axis_angle_deg=0.0,
        mask_area_px=500,
        mask_h=25.0,
        mask_w=25.0,
        ring_stats={},
        noise_sigma=1.0,
        patch_path="",
        mask_path="",
        roi_mask_path="",
        mask_source="polygon",
    )
    M = np.eye(2, 3, dtype=np.float32)

    # Test erase_disease
    op_erase = SwapOp(
        op="erase_disease",
        source_entry_id="99_36",
        target_image_id=10,
        target_fdi=36,
        target_tooth_ref=None,
        mirror=False,
    )
    updated_erase = update_annotations_for_swap(target_anns, op_erase, src_entry, M, 200, 200)
    assert len(updated_erase) == 1
    assert updated_erase[0]["fdi"] == 48

    # Test paste_negative
    op_paste_neg = SwapOp(
        op="paste_negative",
        source_entry_id="99_36",
        target_image_id=10,
        target_fdi=36,
        target_tooth_ref=None,
        mirror=False,
    )
    updated_paste_neg = update_annotations_for_swap(target_anns, op_paste_neg, src_entry, M, 200, 200)
    assert len(updated_paste_neg) == 1
    assert updated_paste_neg[0]["fdi"] == 48


def test_warp_segmentation_polygons():
    target_anns = []
    op = SwapOp(
        op="swap",
        source_entry_id="99_11",
        target_image_id=10,
        target_fdi=11,
        target_tooth_ref=None,
        mirror=False,
    )
    src_entry = ToothEntry(
        entry_id="99_11",
        image_id=99,
        file_name="99.png",
        fdi=11,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[10, 10, 20, 20],
        roi_xywh=[10, 10, 20, 20],
        annotations=[
            {
                "category_id": 2,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "segmentation": [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]],
            }
        ],
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
    # Translate by +5, +10
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 10.0]], dtype=np.float32)

    updated = update_annotations_for_swap(target_anns, op, src_entry, M, 200, 200)
    assert len(updated) == 1
    new_ann = updated[0]
    assert new_ann["bbox"] == [15.0, 20.0, 20.0, 20.0]
    assert new_ann["segmentation"] == [[15.0, 20.0, 35.0, 20.0, 35.0, 40.0, 15.0, 40.0]]


def test_drop_when_out_of_bounds():
    target_anns = []
    op = SwapOp(
        op="swap",
        source_entry_id="99_11",
        target_image_id=10,
        target_fdi=11,
        target_tooth_ref=None,
        mirror=False,
    )
    src_entry = ToothEntry(
        entry_id="99_11",
        image_id=99,
        file_name="99.png",
        fdi=11,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[10, 10, 20, 20],
        roi_xywh=[10, 10, 20, 20],
        annotations=[
            {"category_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0]}
        ],
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
    # Translate so 75% of bbox is outside image of size (20, 20)
    # New box: x: 15..35 -> inside is 15..20 (w=5), y: 15..35 -> inside is 15..20 (h=5)
    # Area inside: 25 / 400 = 0.0625 < 0.5 -> dropped
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]], dtype=np.float32)

    updated = update_annotations_for_swap(target_anns, op, src_entry, M, 20, 20)
    assert len(updated) == 0
