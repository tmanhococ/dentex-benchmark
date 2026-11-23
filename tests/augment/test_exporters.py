# tests/augment/test_exporters.py
import json
from pathlib import Path
import pytest
from dentex_bench.augment.io_utils import (
    STANDARD_CATEGORIES,
    export_coco_dataset,
    export_yolo_dataset,
)


def test_standard_categories_definition():
    """Verify STANDARD_CATEGORIES matches specification."""
    expected = [
        {"id": 1, "name": "Caries"},
        {"id": 2, "name": "Deep Caries"},
        {"id": 3, "name": "Periapical Lesion"},
        {"id": 4, "name": "Impacted Tooth"},
    ]
    assert STANDARD_CATEGORIES == expected


def test_coco_export_structure_and_categories(tmp_path):
    """Test COCO export creates valid json with proper images, annotations, and categories."""
    images = [
        {"id": 1, "file_name": "syn_1.png", "width": 1000, "height": 500},
        {"id": 2, "file_name": "nested/syn_2.png", "width": 800, "height": 600},
    ]
    annotations = [
        {"id": 101, "image_id": 1, "category_id": 1, "bbox": [100.0, 100.0, 50.0, 50.0], "area": 2500.0, "iscrowd": 0},
        {"id": 102, "image_id": 1, "category_id": 3, "bbox": [200.0, 150.0, 40.0, 60.0], "area": 2400.0, "iscrowd": 0},
    ]

    out_file = tmp_path / "deeply" / "nested" / "coco" / "instances.json"
    export_coco_dataset(images, annotations, out_file)

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))

    assert "images" in data
    assert "annotations" in data
    assert "categories" in data

    assert len(data["images"]) == 2
    assert len(data["annotations"]) == 2
    assert data["categories"] == STANDARD_CATEGORIES

    # Also test with str path
    out_file_str = str(tmp_path / "str_path" / "coco.json")
    export_coco_dataset(images, annotations, out_file_str)
    assert Path(out_file_str).exists()


def test_yolo_export_normalization_and_indexing(tmp_path):
    """Test YOLO export 0-indexes category IDs and normalizes bbox coordinates correctly."""
    images = [
        {"id": 1, "file_name": "syn_sample_01.png", "width": 1000, "height": 500}
    ]
    annotations = [
        # category 1 -> YOLO class 0
        # bbox: x=100, y=100, w=50, h=50 -> cx=(100+25)/1000=0.125, cy=(100+25)/500=0.25, nw=50/1000=0.05, nh=50/500=0.1
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100.0, 100.0, 50.0, 50.0]},
        # category 2 -> YOLO class 1
        # bbox: x=0, y=0, w=200, h=100 -> cx=100/1000=0.1, cy=50/500=0.1, nw=200/1000=0.2, nh=100/500=0.2
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0.0, 0.0, 200.0, 100.0]},
        # category 3 -> YOLO class 2
        {"id": 3, "image_id": 1, "category_id": 3, "bbox": [500.0, 250.0, 100.0, 50.0]},
        # category 4 -> YOLO class 3
        {"id": 4, "image_id": 1, "category_id": 4, "bbox": [800.0, 400.0, 100.0, 50.0]},
    ]

    yolo_dir = tmp_path / "yolo_export"
    export_yolo_dataset(images, annotations, yolo_dir)

    label_file = yolo_dir / "labels" / "syn_sample_01.txt"
    assert label_file.exists()

    lines = label_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    assert lines[0] == "0 0.125000 0.250000 0.050000 0.100000"
    assert lines[1] == "1 0.100000 0.100000 0.200000 0.200000"
    assert lines[2] == "2 0.550000 0.550000 0.100000 0.100000"
    assert lines[3] == "3 0.850000 0.850000 0.100000 0.100000"

    # Verify data.yaml
    yaml_file = yolo_dir / "data.yaml"
    assert yaml_file.exists()
    yaml_content = yaml_file.read_text(encoding="utf-8")
    assert "0: Caries" in yaml_content
    assert "1: Deep Caries" in yaml_content
    assert "2: Periapical Lesion" in yaml_content
    assert "3: Impacted Tooth" in yaml_content


def test_yolo_export_empty_annotations_and_multiple_images(tmp_path):
    """Test YOLO export handles images without annotations cleanly."""
    images = [
        {"id": 1, "file_name": "annotated.png", "width": 800, "height": 600},
        {"id": 2, "file_name": "background_clean.png", "width": 800, "height": 600},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 2, "bbox": [400.0, 300.0, 80.0, 60.0]}
    ]

    yolo_dir = tmp_path / "yolo_empty_test"
    export_yolo_dataset(images, annotations, str(yolo_dir))

    lbl_annotated = yolo_dir / "labels" / "annotated.txt"
    lbl_clean = yolo_dir / "labels" / "background_clean.txt"

    assert lbl_annotated.exists()
    assert len(lbl_annotated.read_text(encoding="utf-8").strip().splitlines()) == 1

    assert lbl_clean.exists()
    assert lbl_clean.read_text(encoding="utf-8").strip() == ""
