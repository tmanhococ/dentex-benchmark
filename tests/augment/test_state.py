from pathlib import Path
import json
from dentex_bench.augment.state import (
    Manifest,
    save_manifest,
    load_manifest,
    update_progress,
    atomic_write_json,
)
from dentex_bench.augment.config import load_config


def test_manifest_roundtrip_and_atomic_update(tmp_path):
    cfg = load_config("configs/augment/base.yaml")
    run_dir = tmp_path / "run_base"
    run_dir.mkdir()
    
    manifest = Manifest.create(run_dir, cfg)
    save_manifest(run_dir, manifest)
    
    loaded = load_manifest(run_dir)
    assert loaded.global_seed == 0
    assert loaded.status == "in_progress"
    
    update_progress(run_dir, last_done_index=15, counts_delta={"ops": 3})
    reloaded = load_manifest(run_dir)
    assert reloaded.progress["last_done_index"] == 15
    assert reloaded.counts["ops"] == 3


def test_manifest_create_metadata_and_environment(tmp_path):
    cfg = load_config("configs/augment/base.yaml")
    run_dir = tmp_path / "run_metadata"
    manifest = Manifest.create(run_dir, cfg)
    
    assert manifest.run_id == "run_metadata"
    assert manifest.spec_version == "1.0"
    assert manifest.schema_version == 1
    assert manifest.mode == cfg.mode
    assert manifest.global_seed == cfg.global_seed
    assert manifest.config_hash is not None and len(manifest.config_hash) == 64
    assert "python" in manifest.environment
    assert "numpy" in manifest.environment
    assert "opencv" in manifest.environment
    assert "torch" in manifest.environment
    assert "platform" in manifest.environment
    assert manifest.counts == {"recipes": 0, "ops": 0, "rejected_ops": 0}
    assert manifest.progress == {"last_done_index": -1}
    
    d = manifest.to_dict()
    assert d["run_id"] == "run_metadata"
    assert isinstance(d, dict)


def test_atomic_persistence_and_no_tmp_leak(tmp_path):
    cfg = load_config("configs/augment/base.yaml")
    run_dir = tmp_path / "run_atomic"
    manifest = Manifest.create(run_dir, cfg)
    
    save_manifest(run_dir, manifest)
    manifest_file = run_dir / "state" / "manifest.json"
    tmp_file = run_dir / "state" / "manifest.json.tmp"
    
    assert manifest_file.exists()
    assert not tmp_file.exists()
    
    # Verify valid JSON
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == "run_atomic"
    assert data["status"] == "in_progress"


def test_atomic_write_json_helper(tmp_path):
    target = tmp_path / "subdir" / "data.json"
    sample = {"key": "value", "numbers": [1, 2, 3]}
    atomic_write_json(target, sample)
    
    assert target.exists()
    assert not (tmp_path / "subdir" / "data.json.tmp").exists()
    with open(target, "r", encoding="utf-8") as f:
        assert json.load(f) == sample


def test_multiple_progress_updates(tmp_path):
    cfg = load_config("configs/augment/base.yaml")
    run_dir = tmp_path / "run_multi"
    manifest = Manifest.create(run_dir, cfg)
    save_manifest(run_dir, manifest)
    
    update_progress(run_dir, last_done_index=5, counts_delta={"recipes": 5, "ops": 8})
    update_progress(run_dir, last_done_index=12, counts_delta={"recipes": 7, "ops": 10, "rejected_ops": 2})
    update_progress(run_dir, last_done_index=12, status="completed")
    
    reloaded = load_manifest(run_dir)
    assert reloaded.progress["last_done_index"] == 12
    assert reloaded.counts["recipes"] == 12
    assert reloaded.counts["ops"] == 18
    assert reloaded.counts["rejected_ops"] == 2
    assert reloaded.status == "completed"


def test_string_path_support(tmp_path):
    cfg = load_config("configs/augment/base.yaml")
    run_dir_str = str(tmp_path / "run_str")
    
    manifest = Manifest.create(run_dir_str, cfg)
    save_manifest(run_dir_str, manifest)
    loaded = load_manifest(run_dir_str)
    assert loaded.run_id == "run_str"
    
    update_progress(run_dir_str, last_done_index=1)
    reloaded = load_manifest(run_dir_str)
    assert reloaded.progress["last_done_index"] == 1
