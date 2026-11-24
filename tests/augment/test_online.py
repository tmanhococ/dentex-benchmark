import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from pathlib import Path

from dentex_bench.augment.online import SwapAugmentDataset
from dentex_bench.augment.config import load_config, AugmentConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from real_bank import make_real_bank
from dentex_bench.augment.locator import ToothLocator, LocatedTooth, AnnotationToothLocator


class DummyDictDataset(Dataset):
    def __init__(self, count=5):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return {
            "id": idx,
            "image": np.full((100, 100, 3), 50, dtype=np.uint8),
            "target": [
                {"id": 1, "image_id": idx, "category_id": 1, "bbox": [20.0, 20.0, 30.0, 40.0], "fdi": 16}
            ]
        }


class MockDentexDataset(Dataset):
    def __init__(self, count=5):
        self.images = [{"id": i, "file_name": f"{i}.png", "width": 100, "height": 100} for i in range(count)]
        self.img_to_anns = {
            i: [{"id": i, "image_id": i, "category_id": 1, "bbox": [20.0, 20.0, 30.0, 40.0], "fdi": 16}]
            for i in range(count)
        }
        self.coco = {"images": self.images, "annotations": [a for anns in self.img_to_anns.values() for a in anns]}
        self.resize = False
        self.target_short = None
        self.max_long = None
        self.transforms = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor = torch.full((3, 100, 100), 50.0 / 255.0, dtype=torch.float32)
        target = {
            "boxes": torch.tensor([[20.0, 20.0, 50.0, 60.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([1200.0], dtype=torch.float32),
        }
        return img_tensor, target


def _create_mock_bank():
    entry = ToothEntry(
        entry_id="src_1",
        image_id=999,  # Different from target image IDs to satisfy R_SAME_IMAGE
        file_name="999.png",
        fdi=16,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[20.0, 20.0, 30.0, 40.0],
        roi_xywh=[10, 10, 50, 60],
        annotations=[{"id": 99, "image_id": 999, "category_id": 2, "bbox": [20.0, 20.0, 30.0, 40.0], "fdi": 16}],
        axis_angle_deg=0.0,
        mask_area_px=1000,
        mask_h=40.0,
        mask_w=30.0,
        ring_stats={"mean": 128.0, "std": 20.0},
        noise_sigma=1.5,
        patch_path="",
        mask_path="",
        roi_mask_path="",
        mask_source="polygon",
        healthy_score=None,
    )
    return make_real_bank([entry])


def _create_mock_locator():
    class _Loc:
        def teeth(self, image_id: int):
            return [LocatedTooth(fdi=16, bbox=[20.0, 20.0, 30.0, 40.0], source="annotations", score=1.0)]
    return _Loc()


def test_online_dataset_epoch_determinism():
    cfg = load_config("configs/augment/base.yaml")
    bank = _create_mock_bank()
    locator = _create_mock_locator()

    ds1 = SwapAugmentDataset(base=DummyDictDataset(), cfg=cfg, bank=bank, locator=locator)
    ds1.set_epoch(1)
    assert ds1._epoch == 1

    ds2 = SwapAugmentDataset(base=DummyDictDataset(), cfg=cfg, bank=bank, locator=locator)
    ds2.set_epoch(1)
    assert ds2._epoch == 1

    # Determinism across identical instances with same epoch
    item1 = ds1[0]
    item2 = ds2[0]
    assert np.array_equal(item1["image"], item2["image"])
    assert item1["target"] == item2["target"]

    # Different epochs produce different stochastic resolution (e.g. jitter)
    ds1.set_epoch(2)
    item_epoch2 = ds1[0]
    # Box coordinates vary across epochs due to jitter_scale / jitter_rotate
    box_e1 = item1["target"][0]["bbox"]
    box_e2 = item_epoch2["target"][0]["bbox"]
    assert box_e1 != box_e2


def test_online_dataset_probability_zero_and_one():
    cfg = load_config("configs/augment/base.yaml")
    bank = _create_mock_bank()
    locator = _create_mock_locator()
    base_ds = DummyDictDataset()

    # Case 1: p = 0.0 -> Must return exact unmodified base item
    cfg.online = {"p": 0.0}
    ds_zero = SwapAugmentDataset(base=base_ds, cfg=cfg, bank=bank, locator=locator)
    ds_zero.set_epoch(1)
    item_zero = ds_zero[0]
    raw_item = base_ds[0]
    assert np.array_equal(item_zero["image"], raw_item["image"])
    assert item_zero["target"] == raw_item["target"]

    # Case 2: p = 1.0 -> Must execute augmentation
    cfg.online = {"p": 1.0}
    ds_one = SwapAugmentDataset(base=base_ds, cfg=cfg, bank=bank, locator=locator)
    ds_one.set_epoch(1)
    item_one = ds_one[0]
    # In raw_item, category_id is 1. Swapped tooth from src_1 has category_id 2!
    assert item_one["target"][0]["category_id"] == 2
    assert item_one["target"][0].get("synthetic") is True


def test_online_dataset_with_dentex_dataset():
    cfg = load_config("configs/augment/base.yaml")
    cfg.online = {"p": 1.0}
    bank = _create_mock_bank()
    base_ds = MockDentexDataset()

    # Automatically builds locator from base_ds annotations if locator is None
    ds = SwapAugmentDataset(base=base_ds, cfg=cfg, bank=bank)
    ds.set_epoch(1)
    assert len(ds) == len(base_ds)

    img_tensor, target = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 100, 100)
    assert img_tensor.dtype == torch.float32

    assert isinstance(target, dict)
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target
    assert "area" in target

    # Verify boxes are xyxy
    boxes = target["boxes"]
    assert boxes.shape == (1, 4)
    # Box should be [20, 20, 50, 60] (with small jitter possible)
    assert boxes[0, 2] > boxes[0, 0]
    assert boxes[0, 3] > boxes[0, 1]

    # Verify swapped label from bank (category_id was 2 in bank entry)
    assert target["labels"][0].item() == 2


def test_online_dataset_p_none_or_missing_recipe():
    cfg = load_config("configs/augment/base.yaml")
    cfg.online = {"p": 1.0}
    # Empty bank -> recipe resolution returns None
    bank = ToothBank([])
    base_ds = DummyDictDataset()

    ds = SwapAugmentDataset(base=base_ds, cfg=cfg, bank=bank)
    ds.set_epoch(1)
    # When recipe is None, returns base item
    item = ds[0]
    raw_item = base_ds[0]
    assert np.array_equal(item["image"], raw_item["image"])
    assert item["target"] == raw_item["target"]
