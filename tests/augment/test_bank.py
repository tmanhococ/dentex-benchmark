import json
import numpy as np
from pathlib import Path
from dentex_bench.augment.bank import ToothEntry, ToothBank
from dentex_bench.augment.config import AugmentConfig


def test_tooth_bank_save_load(tmp_path):
    entry = ToothEntry(
        entry_id="1_16",
        image_id=1,
        file_name="sample_1.png",
        fdi=16,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[10.0, 20.0, 30.0, 40.0],
        roi_xywh=[0, 10, 50, 60],
        annotations=[{"category_id": 1, "bbox": [10.0, 10.0, 30.0, 40.0]}],
        axis_angle_deg=5.0,
        mask_area_px=1000,
        mask_h=40.0,
        mask_w=30.0,
        ring_stats={"mean": 128.0, "std": 20.0},
        noise_sigma=1.5,
        patch_path="patches/1_16.png",
        mask_path="masks/1_16_tooth.png",
        roi_mask_path="masks/1_16_roi.png",
        mask_source="polygon",
        healthy_score=None,
    )

    bank = ToothBank(entries=[entry])
    bank_dir = tmp_path / "bank"
    bank.save(bank_dir)

    assert (bank_dir / "bank_index.jsonl").exists()
    assert (bank_dir / "bank_meta.json").exists()

    loaded_bank = ToothBank.load(bank_dir)
    assert len(loaded_bank.entries) == 1
    assert loaded_bank.entries[0].entry_id == "1_16"


def test_tooth_bank_queries():
    entry1 = ToothEntry(
        entry_id="1_16",
        image_id=1,
        file_name="sample_1.png",
        fdi=16,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[10.0, 20.0, 30.0, 40.0],
        roi_xywh=[0, 10, 50, 60],
        annotations=[{"category_id": 1, "bbox": [10.0, 10.0, 30.0, 40.0]}],
        axis_angle_deg=5.0,
        mask_area_px=1000,
        mask_h=40.0,
        mask_w=30.0,
        ring_stats={"mean": 128.0, "std": 20.0},
        noise_sigma=1.5,
        patch_path="patches/1_16.png",
        mask_path="masks/1_16_tooth.png",
        roi_mask_path="masks/1_16_roi.png",
        mask_source="polygon",
        healthy_score=None,
    )
    entry2 = ToothEntry(
        entry_id="2_26",
        image_id=2,
        file_name="sample_2.png",
        fdi=26,
        is_upper=True,
        labels=["deep_caries"],
        is_healthy=False,
        roi_class="deep_caries",
        bbox_xywh=[50.0, 20.0, 30.0, 40.0],
        roi_xywh=[40, 10, 50, 60],
        annotations=[{"category_id": 2, "bbox": [50.0, 10.0, 30.0, 40.0]}],
        axis_angle_deg=-3.0,
        mask_area_px=1100,
        mask_h=40.0,
        mask_w=30.0,
        ring_stats={"mean": 130.0, "std": 18.0},
        noise_sigma=1.2,
        patch_path="patches/2_26.png",
        mask_path="masks/2_26_tooth.png",
        roi_mask_path="masks/2_26_roi.png",
        mask_source="polygon",
        healthy_score=None,
    )

    bank = ToothBank(entries=[entry1, entry2])

    assert bank.get("1_16") == entry1
    assert bank.get("nonexistent") is None

    assert bank.by_fdi(16) == [entry1]
    assert bank.by_fdi(26) == [entry2]
    assert bank.by_fdi(36) == []

    assert bank.by_class("caries") == [entry1]
    assert bank.by_class("deep_caries") == [entry2]
    assert bank.by_class("periapical") == []


def test_tooth_bank_is_cached(tmp_path):
    bank_dir = tmp_path / "bank"
    assert ToothBank.is_cached(bank_dir) is False

    bank_dir.mkdir(parents=True)
    assert ToothBank.is_cached(bank_dir) is False

    (bank_dir / "bank_index.jsonl").touch()
    assert ToothBank.is_cached(bank_dir) is False

    (bank_dir / "bank_meta.json").touch()
    assert ToothBank.is_cached(bank_dir) is True
