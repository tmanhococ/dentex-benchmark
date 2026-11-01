# tests/test_analysis_fixtures.py
import json
from pathlib import Path
import pytest
from scripts.create_test_fixtures import create_sample_fixtures

def test_create_sample_fixtures(tmp_path):
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    
    stats = create_sample_fixtures(data_dir=data_dir, outputs_dir=outputs_dir, num_samples=4)
    assert stats["train_images"] >= 2
    assert stats["val_images"] >= 2
    assert stats["annotations"] >= 4
    
    train_json = data_dir / "processed" / "instances_train.json"
    val_json = data_dir / "processed" / "instances_val.json"
    assert train_json.exists()
    assert val_json.exists()
    
    with open(train_json, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    assert len(t_data["images"]) >= 2
    assert len(t_data["categories"]) == 4
    
    # Check dummy model checkpoints
    assert (outputs_dir / "hierarchical_net" / "best.pth").exists()
    assert (outputs_dir / "diffusion_net" / "best.pth").exists()
