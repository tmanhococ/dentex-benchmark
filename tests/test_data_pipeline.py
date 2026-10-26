# tests/test_data_pipeline.py
import json
import pytest
import numpy as np
import torch
from pathlib import Path
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
    
    ds = DentexDataset(img_dir=tmp_path, ann_file=train_path)
    assert len(ds) == 1
    img_tensor, target = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 50, 100)
    assert target["boxes"].shape == (2, 4)
    assert list(target["labels"].numpy()) == [1, 2]
    assert target["image_id"].item() == 1
