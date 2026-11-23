# tests/augment/test_hard_negative.py
import json
from pathlib import Path
import numpy as np
import pytest

from dentex_bench.augment.locator import (
    LocatedTooth,
    PseudoLabelToothLocator,
    ToothLocator,
)
from dentex_bench.augment.extras import (
    plan_hard_negative_ops,
    plan_disease_erasing_ops,
)


def test_pseudo_label_locator_and_hard_neg(tmp_path: Path):
    pseudo_file = tmp_path / "pseudo.json"
    dummy = {
        "annotations": [
            {"image_id": 10, "fdi": 16, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.85}
        ]
    }
    pseudo_file.write_text(json.dumps(dummy), encoding="utf-8")

    loc = PseudoLabelToothLocator(pseudo_file, min_score=0.5)
    teeth = loc.teeth(10)
    assert len(teeth) == 1
    assert teeth[0].fdi == 16
    assert teeth[0].score == 0.85
    assert teeth[0].source == "pseudo_label"


def test_pseudo_label_locator_basic(tmp_path: Path):
    pseudo_file = tmp_path / "pseudo_coco.json"
    dummy = {
        "images": [{"id": 1, "file_name": "img1.png"}, {"id": 2, "file_name": "img2.png"}],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "fdi": 16,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "score": 0.9,
                "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            },
            {
                "id": 102,
                "image_id": 1,
                "fdi": 17,
                "bbox": [50.0, 20.0, 30.0, 40.0],
                "score": 0.35,  # Below threshold
            },
            {
                "id": 103,
                "image_id": 1,
                "category_id_1": 2,
                "category_id_2": 1,  # FDI 21
                "bbox": [100.0, 20.0, 30.0, 40.0],
                "score": 0.75,
            },
            {
                "id": 104,
                "image_id": 2,
                "fdi": 46,
                "bbox": [200.0, 200.0, 35.0, 45.0],
                "score": 0.95,
            },
        ],
    }
    pseudo_file.write_text(json.dumps(dummy), encoding="utf-8")

    locator = PseudoLabelToothLocator(pseudo_file, min_score=0.5)
    assert isinstance(locator, ToothLocator)

    teeth_img1 = locator.teeth(1)
    assert len(teeth_img1) == 2
    assert teeth_img1[0].fdi == 16
    assert teeth_img1[0].score == 0.9
    assert teeth_img1[0].bbox == [10.0, 20.0, 30.0, 40.0]
    assert teeth_img1[0].segmentation == [[10, 20, 40, 20, 40, 60, 10, 60]]
    assert teeth_img1[0].source == "pseudo_label"

    assert teeth_img1[1].fdi == 21
    assert teeth_img1[1].score == 0.75

    teeth_img2 = locator.teeth(2)
    assert len(teeth_img2) == 1
    assert teeth_img2[0].fdi == 46

    # Non-existent image returns empty list
    assert locator.teeth(999) == []


def test_pseudo_label_locator_empty_and_missing_file(tmp_path: Path):
    # Non-existent file raises FileNotFoundError
    missing = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        PseudoLabelToothLocator(missing)

    # Empty JSON dict
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("{}", encoding="utf-8")
    loc_empty = PseudoLabelToothLocator(empty_json)
    assert loc_empty.teeth(1) == []

    # Empty annotations list
    empty_anns = tmp_path / "empty_anns.json"
    empty_anns.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    loc_anns = PseudoLabelToothLocator(empty_anns)
    assert loc_anns.teeth(1) == []

    # 0-byte file
    zero_file = tmp_path / "zero_bytes.json"
    zero_file.write_text("", encoding="utf-8")
    loc_zero = PseudoLabelToothLocator(zero_file)
    assert loc_zero.teeth(1) == []


class DummyEntry:
    def __init__(self, entry_id: str, image_id: int, fdi: int, is_healthy: bool = True):
        self.entry_id = entry_id
        self.image_id = image_id
        self.fdi = fdi
        self.is_healthy = is_healthy


def test_plan_hard_negative_ops():
    healthy_entries = [
        DummyEntry(entry_id="bank_16_a", image_id=1, fdi=16),
        DummyEntry(entry_id="bank_21_a", image_id=2, fdi=21),
        DummyEntry(entry_id="bank_36_a", image_id=3, fdi=36),
    ]

    target_teeth = [
        LocatedTooth(fdi=16, bbox=[10.0, 20.0, 30.0, 40.0], source="pseudo_label", score=0.9),
        LocatedTooth(fdi=21, bbox=[50.0, 60.0, 30.0, 40.0], source="pseudo_label", score=0.88),
        LocatedTooth(fdi=48, bbox=[100.0, 100.0, 30.0, 40.0], source="pseudo_label", score=0.92),  # No bank match
    ]

    # Test basic op generation
    ops = plan_hard_negative_ops(
        target_image_id=10,
        healthy_bank_entries=healthy_entries,
        locator_teeth=target_teeth,
        max_per_image=2,
    )

    assert len(ops) == 2
    assert ops[0]["op"] == "paste_negative"
    assert ops[0]["target_image_id"] == 10
    assert ops[0]["target_fdi"] == 16
    assert ops[0]["source_entry_id"] == "bank_16_a"
    assert ops[0]["target_tooth_ref"]["source"] == "pseudo_label"
    assert ops[0]["target_tooth_ref"]["bbox"] == [10.0, 20.0, 30.0, 40.0]

    assert ops[1]["op"] == "paste_negative"
    assert ops[1]["target_image_id"] == 10
    assert ops[1]["target_fdi"] == 21
    assert ops[1]["source_entry_id"] == "bank_21_a"

    # Test max_per_image constraint
    ops_1 = plan_hard_negative_ops(
        target_image_id=10,
        healthy_bank_entries=healthy_entries,
        locator_teeth=target_teeth,
        max_per_image=1,
    )
    assert len(ops_1) == 1

    # Test with RNG
    rng = np.random.default_rng(42)
    ops_rng = plan_hard_negative_ops(
        target_image_id=10,
        healthy_bank_entries=healthy_entries,
        locator_teeth=target_teeth,
        rng=rng,
        max_per_image=2,
    )
    assert len(ops_rng) == 2


def test_plan_disease_erasing_ops():
    healthy_entries = [
        DummyEntry(entry_id="bank_16_h", image_id=101, fdi=16, is_healthy=True),
        DummyEntry(entry_id="bank_36_h", image_id=102, fdi=36, is_healthy=True),
    ]

    diseased_target_teeth = [
        {"fdi": 16, "bbox": [10.0, 20.0, 30.0, 40.0], "category_id": 1, "source": "annotations"},
        {"fdi": 36, "bbox": [70.0, 80.0, 30.0, 40.0], "category_id": 2, "source": "annotations"},
        {"fdi": 25, "bbox": [120.0, 130.0, 30.0, 40.0], "category_id": 1, "source": "annotations"},  # No healthy bank match
    ]

    ops = plan_disease_erasing_ops(
        target_image_id=20,
        diseased_target_teeth=diseased_target_teeth,
        healthy_bank_entries=healthy_entries,
    )

    assert len(ops) == 2
    assert ops[0]["op"] == "erase_disease"
    assert ops[0]["target_image_id"] == 20
    assert ops[0]["target_fdi"] == 16
    assert ops[0]["source_entry_id"] == "bank_16_h"

    assert ops[1]["op"] == "erase_disease"
    assert ops[1]["target_image_id"] == 20
    assert ops[1]["target_fdi"] == 36
    assert ops[1]["source_entry_id"] == "bank_36_h"
