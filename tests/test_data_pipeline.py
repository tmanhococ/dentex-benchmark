# tests/test_data_pipeline.py
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import numpy as np
import torch
from dentex_bench.data.parser import convert_dentex_annotations_to_coco, TARGET_CLASSES
from dentex_bench.data.dataset import DentexDataset

def test_target_classes():
    assert len(TARGET_CLASSES) == 4
    assert TARGET_CLASSES[1] == "Caries"
    assert TARGET_CLASSES[2] == "Deep Caries"
    assert TARGET_CLASSES[3] == "Periapical Lesion"
    assert TARGET_CLASSES[4] == "Impacted Tooth"

def test_parse_dentex_mock(tmp_path):
    mock_dentex = [
        {
            "image_id": 1,
            "file_name": "test_01.png",
            "width": 1000,
            "height": 500,
            "annotations": [
                {"bbox": [100, 100, 50, 50], "category": "Caries"},
                {"bbox": [200, 200, 60, 60], "category": "Deep Caries"},
                {"bbox": [300, 300, 40, 40], "category": "Normal"}  # should be ignored
            ]
        },
        {
            "image_id": 2,
            "file_name": "test_02.png",
            "width": 1000,
            "height": 500,
            "annotations": [
                {"bbox": [150, 150, 50, 50], "category": "Impacted Tooth"},
                {"bbox": [250, 250, 70, 70], "category": "Periapical Lesion"}
            ]
        }
    ]
    raw_path = tmp_path / "mock_raw.json"
    with open(raw_path, "w") as f:
        json.dump(mock_dentex, f)
        
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    
    summary = convert_dentex_annotations_to_coco(
        raw_ann_file=raw_path,
        train_out=train_path,
        val_out=val_path,
        split_ratio=0.5,
        seed=42
    )
    
    assert train_path.exists()
    assert val_path.exists()
    with open(train_path) as f:
        coco_train = json.load(f)
    assert len(coco_train["categories"]) == 4
    assert len(coco_train["images"]) == 1
    assert len(coco_train["annotations"]) == 2  # excludes Normal

def test_parse_coco_format_with_normal_exclusion(tmp_path):
    mock_coco = {
        "images": [
            {
                "id": 10,
                "file_name": "tooth_01.png",
                "width": 1000,
                "height": 500
            }
        ],
        "categories": [
            {"id": 1, "name": "Normal"},
            {"id": 2, "name": "Caries"}
        ],
        "annotations": [
            {
                "id": 100,
                "image_id": 10,
                "category_id": 1,  # "Normal" in raw_cat_map -> must be excluded!
                "bbox": [50, 50, 40, 40]
            },
            {
                "id": 101,
                "image_id": 10,
                "category_id": 2,  # "Caries" in raw_cat_map -> must be remapped to 1 (Caries in TARGET_CLASSES)
                "bbox": [150, 150, 60, 60]
            }
        ]
    }
    raw_path = tmp_path / "mock_coco_raw.json"
    with open(raw_path, "w") as f:
        json.dump(mock_coco, f)
        
    train_path = tmp_path / "train_coco.json"
    val_path = tmp_path / "val_coco.json"
    
    summary = convert_dentex_annotations_to_coco(
        raw_ann_file=raw_path,
        train_out=train_path,
        val_out=val_path,
        split_ratio=1.0,
        seed=42
    )
    
    assert train_path.exists()
    with open(train_path) as f:
        train_data = json.load(f)
        
    assert len(train_data["images"]) == 1
    assert len(train_data["annotations"]) == 1
    ann = train_data["annotations"][0]
    # Caries must map to TARGET_CLASSES ID 1
    assert ann["category_id"] == 1
    assert ann["bbox"] == [150, 150, 60, 60]

def test_dentex_dataset(tmp_path):
    mock_dentex = [
        {
            "image_id": 1,
            "file_name": "test_01.png",
            "width": 100,
            "height": 50,
            "annotations": [
                {"bbox": [10, 10, 20, 20], "category": "Caries"},
                {"bbox": [20, 20, 30, 30], "category": "Deep Caries"}
            ]
        }
    ]
    raw_path = tmp_path / "mock_raw.json"
    with open(raw_path, "w") as f:
        json.dump(mock_dentex, f)
        
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    
    convert_dentex_annotations_to_coco(
        raw_ann_file=raw_path,
        train_out=train_path,
        val_out=val_path,
        split_ratio=1.0,
        seed=42
    )
    
    ds = DentexDataset(img_dir=tmp_path, ann_file=train_path, resize=False)
    assert len(ds) == 1
    img_tensor, target = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 50, 100)
    assert target["boxes"].shape == (2, 4)
    assert list(target["labels"].numpy()) == [1, 2]
    assert target["image_id"].item() == 1

def test_dentex_dataset_resizing(tmp_path):
    mock_dentex = [
        {
            "image_id": 1,
            "file_name": "large_01.png",
            "width": 2600,
            "height": 1300,
            "annotations": [
                {"bbox": [260, 130, 260, 260], "category": "Caries"}
            ]
        }
    ]
    raw_path = tmp_path / "mock_raw.json"
    with open(raw_path, "w") as f:
        json.dump(mock_dentex, f)
        
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    
    convert_dentex_annotations_to_coco(
        raw_ann_file=raw_path,
        train_out=train_path,
        val_out=val_path,
        split_ratio=1.0,
        seed=42
    )
    
    # Default dataset should resize large image to safe dimensions
    ds = DentexDataset(img_dir=tmp_path, ann_file=train_path, target_short=800, max_long=1333)
    img_tensor, target = ds[0]
    h, w = img_tensor.shape[1], img_tensor.shape[2]
    assert max(h, w) <= 1333
    assert min(h, w) <= 800
    # Original aspect ratio 2:1 -> scaled down by scale factor
    expected_scale = min(800.0 / 1300.0, 1333.0 / 2600.0)
    assert abs(w - round(2600 * expected_scale)) <= 1
    assert abs(h - round(1300 * expected_scale)) <= 1
    # Check boxes scaled proportionally
    box = target["boxes"][0]
    assert abs(box[0].item() - 260 * expected_scale) < 1.0
    assert abs(box[1].item() - 130 * expected_scale) < 1.0

