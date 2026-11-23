import json
import subprocess
import sys
from pathlib import Path

def test_inspect_tool_generates_valid_report(tmp_path):
    ann_file = tmp_path / "sample_ann.json"
    dummy_data = {
        "images": [{"id": 1, "file_name": "img1.png", "width": 800, "height": 400}],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 150.0, 50.0, 60.0],
                "segmentation": [[100, 150, 150, 150, 150, 210, 100, 210]],
                "area": 3000.0,
                "iscrowd": 0
            }
        ],
        "categories": [{"id": 1, "name": "Caries"}]
    }
    ann_file.write_text(json.dumps(dummy_data))
    
    out_report = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "scripts/inspect_dentex_json.py",
        "--ann", str(ann_file),
        "--out", str(out_report)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert out_report.exists()
    
    report = json.loads(out_report.read_text())
    assert "top_level_keys" in report
    assert "total_images" in report
    assert "total_annotations" in report
    assert "polygon_valid_ratio" in report
    assert report["polygon_valid_ratio"] == 1.0
