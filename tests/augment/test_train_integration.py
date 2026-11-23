import json
import subprocess
import sys
from pathlib import Path
import pytest

from scripts.train import parse_args, resolve_ann_paths, get_augment_metadata


def test_train_cli_accepts_augment_flags():
    """Verifies that --help includes --augment-config and --train-ann."""
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "train.py"
    cmd = [sys.executable, str(script), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--augment-config" in res.stdout
    assert "--train-ann" in res.stdout


def test_train_args_custom_train_ann():
    """Tests parse_args with custom --train-ann and --augment-config."""
    args = parse_args([
        "--model", "faster_rcnn",
        "--train-ann", "outputs/augment/E6_core/augmented_train.json",
        "--augment-config", "configs/augment/base.yaml",
    ])
    assert args.train_ann == "outputs/augment/E6_core/augmented_train.json"
    assert args.augment_config == "configs/augment/base.yaml"


def test_train_inv2_val_never_augmented(tmp_path):
    """Tests that val_ann is always the pristine instances_val.json and never touched by augment flags."""
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    val_file = processed_dir / "instances_val.json"
    val_file.write_text('{"images": [], "annotations": []}')
    train_file = processed_dir / "instances_train.json"
    train_file.write_text('{"images": [], "annotations": []}')

    custom_train = tmp_path / "custom_augmented_train.json"
    custom_train.write_text('{"images": [], "annotations": []}')

    # Case 1: Custom --train-ann provided
    args = parse_args([
        "--model", "faster_rcnn",
        "--data_dir", str(data_dir),
        "--train-ann", str(custom_train),
        "--augment-config", "configs/augment/base.yaml",
    ])
    train_ann, val_ann = resolve_ann_paths(args)
    assert Path(train_ann).resolve() == custom_train.resolve()
    assert Path(val_ann).resolve() == val_file.resolve()
    assert Path(train_ann).resolve() != Path(val_ann).resolve()
    assert "val" in str(val_ann).lower()

    # Case 2: Attempting to point --train-ann to validation set violates INV-2
    bad_args = parse_args([
        "--model", "faster_rcnn",
        "--data_dir", str(data_dir),
        "--train-ann", str(val_file),
    ])
    with pytest.raises(AssertionError, match="INV-2 VIOLATION"):
        resolve_ann_paths(bad_args)

    # Case 3: Default mode without custom --train-ann
    default_args = parse_args([
        "--model", "faster_rcnn",
        "--data_dir", str(data_dir),
    ])
    def_train_ann, def_val_ann = resolve_ann_paths(default_args)
    assert Path(def_train_ann).resolve() == train_file.resolve()
    assert Path(def_val_ann).resolve() == val_file.resolve()
    assert Path(def_train_ann).resolve() != Path(def_val_ann).resolve()


def test_get_augment_metadata_discovers_manifest(tmp_path):
    """Verifies that offline manifest hash and metadata are discovered and logged."""
    run_dir = tmp_path / "E3_fdi_hard"
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True)

    train_ann_path = run_dir / "augmented_train.json"
    train_ann_path.write_text('{"images": [], "annotations": []}')

    manifest_file = state_dir / "manifest.json"
    manifest_data = {"manifest_hash": "abc123hash", "count": 100}
    manifest_file.write_text(json.dumps(manifest_data))

    meta = get_augment_metadata(train_ann=train_ann_path, augment_config_path="configs/augment/base.yaml")
    assert meta["custom_train_ann"] == str(train_ann_path)
    assert meta["manifest_hash"] == "abc123hash"
    assert meta["manifest_path"] == str(manifest_file)
    assert meta["augment_config"] == "configs/augment/base.yaml"
    assert "augment_run_name" in meta
