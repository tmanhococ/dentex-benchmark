# tests/augment/test_replay.py
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.offline import run_offline
from dentex_bench.augment.io_utils import sha256_file, read_image_lossless


def _make_test_bank() -> ToothBank:
    entry = ToothEntry(
        entry_id="bank_1",
        image_id=99,
        file_name="99.png",
        fdi=16,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[10.0, 10.0, 20.0, 20.0],
        roi_xywh=[5, 5, 30, 30],
        annotations=[{"category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}],
        axis_angle_deg=0.0,
        mask_area_px=400,
        mask_h=20.0,
        mask_w=20.0,
        ring_stats={},
        noise_sigma=1.0,
        patch_path="",
        mask_path="",
        roi_mask_path="",
        mask_source="polygon",
    )
    return ToothBank([entry])


def _make_test_ann_file(path: Path, num_images: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for i in range(1, num_images + 1):
        images.append({"id": i, "file_name": f"img_{i}.png", "width": 800, "height": 400})
        annotations.append({
            "id": i,
            "image_id": i,
            "category_id": 1,
            "fdi": 16,
            "bbox": [100.0, 100.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        })
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "Caries"}]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)


def test_replay_cli_help():
    cmd = [sys.executable, "scripts/replay_augment.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--state" in res.stdout
    assert "--override" in res.stdout


def test_replay_bit_exact(tmp_path):
    """Asserting sha256 match between offline output and replayed output."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=2)
    bank = _make_test_bank()

    out_root = tmp_path / "out_offline"
    cfg = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={out_root}",
            f"paths.train_ann={ann_file}",
            "offline.num_workers=1",
            "offline.variants_per_image=1",
            "output.save_dataset=true",
        ],
    )
    run_dir = run_offline(cfg, bank=bank)
    state_dir = run_dir / "state"
    assert (state_dir / "recipes.jsonl").is_file()

    # Locate original offline synthetic image
    orig_syn_dir = run_dir / "synthetic"
    orig_files = list(orig_syn_dir.glob("*.png"))
    assert len(orig_files) > 0, "Offline runner must generate synthetic images when save_dataset=true"
    orig_hashes = {f.name: sha256_file(f) for f in orig_files}

    # Replay via Python API
    from scripts.replay_augment import replay

    replay_out = tmp_path / "replay_out"
    res = replay(state_dir=state_dir, out_dir=replay_out, verify_hash=True)
    assert res["status"] == "success"
    assert res["replayed_count"] == len(orig_files)

    replayed_syn_dir = replay_out / "synthetic"
    replayed_files = list(replayed_syn_dir.glob("*.png"))
    assert len(replayed_files) == len(orig_files)

    for rf in replayed_files:
        assert rf.name in orig_hashes
        replayed_hash = sha256_file(rf)
        assert replayed_hash == orig_hashes[rf.name], f"Hash mismatch for {rf.name}"
        # Also assert bit-exact pixel array equality
        img_orig = read_image_lossless(orig_syn_dir / rf.name)
        img_replayed = read_image_lossless(rf)
        np.testing.assert_array_equal(img_orig, img_replayed)

    # Also replay via CLI
    replay_cli_out = tmp_path / "replay_cli_out"
    cmd = [
        sys.executable,
        "scripts/replay_augment.py",
        "--state",
        str(state_dir),
        "--out",
        str(replay_cli_out),
        "--verify-hash",
    ]
    sub_res = subprocess.run(cmd, capture_output=True, text=True)
    assert sub_res.returncode == 0, f"Replay CLI failed: {sub_res.stderr}\n{sub_res.stdout}"

    cli_syn_dir = replay_cli_out / "synthetic"
    for rf in cli_syn_dir.glob("*.png"):
        assert sha256_file(rf) == orig_hashes[rf.name]


def test_replay_recipe_ids_filter(tmp_path):
    """Replaying with --recipe-ids filters execution to only specified recipes."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=2)
    bank = _make_test_bank()

    out_root = tmp_path / "out_offline_filter"
    cfg = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={out_root}",
            f"paths.train_ann={ann_file}",
            "offline.num_workers=1",
            "offline.variants_per_image=2",
            "output.save_dataset=true",
        ],
    )
    run_dir = run_offline(cfg, bank=bank)
    state_dir = run_dir / "state"

    # Read recipes
    recipes_lines = (state_dir / "recipes.jsonl").read_text().splitlines()
    all_recipes = [json.loads(line) for line in recipes_lines if line.strip()]
    assert len(all_recipes) >= 2
    selected_id = all_recipes[0]["recipe_id"]

    from scripts.replay_augment import replay

    replay_out = tmp_path / "replay_filtered"
    res = replay(state_dir=state_dir, out_dir=replay_out, recipe_ids=[selected_id])
    assert res["replayed_count"] == 1
    assert res["recipes"][0].recipe_id == selected_id

    replayed_files = list((replay_out / "synthetic").glob("*.png"))
    assert len(replayed_files) == 1
