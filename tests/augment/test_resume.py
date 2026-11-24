# tests/augment/test_resume.py
import json
from pathlib import Path
import pytest

from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from real_bank import make_real_bank, write_images_for_ann
from dentex_bench.augment.state import load_manifest
from dentex_bench.augment.offline import run_offline


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
    return make_real_bank([entry])


def _make_test_ann_file(path: Path, num_images: int = 4) -> None:
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


def test_resume_offline_interrupted(tmp_path, monkeypatch):
    """When an offline run is interrupted, resuming skips existing completed recipes and completes manifest."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=4)
    bank = _make_test_bank()

    out_root = tmp_path / "out_resume"
    cfg = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={out_root}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=1",
            "offline.variants_per_image=1",
            "output.save_dataset=false",
        ],
    )

    run_dir = out_root / "base_seed0"

    # Simulate an interruption during run 1 after processing image index 1 (second image)
    from dentex_bench.augment import offline

    orig_process = offline._process_image_variants
    call_count = {"count": 0}

    def failing_process(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] > 2:
            raise KeyboardInterrupt("Simulated interruption")
        return orig_process(*args, **kwargs)

    monkeypatch.setattr(offline, "_process_image_variants", failing_process)

    # First run fails halfway
    with pytest.raises(KeyboardInterrupt):
        run_offline(cfg, bank=bank)

    # Verify manifest reflects incomplete state
    manifest = load_manifest(run_dir)
    assert manifest.status == "in_progress"
    assert manifest.progress["last_done_index"] == 1  # 2 images (idx 0 and 1) completed
    recipes_after_fail = (run_dir / "state" / "recipes.jsonl").read_text().splitlines()
    assert len(recipes_after_fail) == 2

    # Restore normal processing
    monkeypatch.setattr(offline, "_process_image_variants", orig_process)

    # Track which images are processed during resume
    resumed_calls = []

    def tracking_process(*args, **kwargs):
        # args[0] is task tuple (img_info, ...)
        resumed_calls.append(args[0])
        return orig_process(*args, **kwargs)

    monkeypatch.setattr(offline, "_process_image_variants", tracking_process)

    # Resume the run
    run_offline(cfg, bank=bank)

    # Check that already completed images (indices 0 and 1) were NOT re-processed
    # Only remaining 2 images should have been processed
    assert len(resumed_calls) == 2

    # Manifest should now be complete
    manifest_final = load_manifest(run_dir)
    assert manifest_final.status == "complete"
    assert manifest_final.progress["last_done_index"] == 3

    final_recipes = (run_dir / "state" / "recipes.jsonl").read_text().splitlines()
    assert len(final_recipes) == 4


def test_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    """An interrupted-and-resumed run produces identical recipes to an uninterrupted run."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=3)
    bank = _make_test_bank()

    # Uninterrupted run
    cfg_uninterrupted = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_uninterrupted'}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=1",
            "offline.variants_per_image=2",
            "output.save_dataset=false",
        ],
    )
    run_offline(cfg_uninterrupted, bank=bank)
    rec_clean = (tmp_path / "out_uninterrupted" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()

    # Interrupted run
    cfg_interrupted = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_interrupted'}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=1",
            "offline.variants_per_image=2",
            "output.save_dataset=false",
        ],
    )

    from dentex_bench.augment import offline
    orig_process = offline._process_image_variants
    call_count = {"count": 0}

    def failing_process(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] > 1:
            raise RuntimeError("Simulated crash")
        return orig_process(*args, **kwargs)

    monkeypatch.setattr(offline, "_process_image_variants", failing_process)
    with pytest.raises(RuntimeError):
        run_offline(cfg_interrupted, bank=bank)

    # Resume run
    monkeypatch.setattr(offline, "_process_image_variants", orig_process)
    run_offline(cfg_interrupted, bank=bank)

    rec_resumed = (tmp_path / "out_interrupted" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()
    assert rec_clean == rec_resumed


def test_resume_already_complete_no_op(tmp_path):
    """Calling run_offline on an already completed run without overwrite is a no-op."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=2)
    bank = _make_test_bank()

    cfg = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_noop'}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=1",
            "output.save_dataset=false",
        ],
    )
    run_offline(cfg, bank=bank)
    rec_orig = (tmp_path / "out_noop" / "base_seed0" / "state" / "recipes.jsonl").read_text()

    # Run again
    run_offline(cfg, bank=bank)
    rec_second = (tmp_path / "out_noop" / "base_seed0" / "state" / "recipes.jsonl").read_text()
    assert rec_orig == rec_second


def test_force_overwrite_reprocesses_run(tmp_path, monkeypatch):
    """Calling run_offline with overwrite=True re-processes images even if completed."""
    ann_file = tmp_path / "train_ann.json"
    _make_test_ann_file(ann_file, num_images=2)
    bank = _make_test_bank()

    cfg = load_config(
        "configs/augment/base.yaml",
        cli_overrides=[
            f"paths.out_root={tmp_path / 'out_force'}",
            f"paths.train_ann={ann_file}",
            f"paths.img_dir={write_images_for_ann(ann_file)}",
            "offline.num_workers=1",
            "output.save_dataset=false",
        ],
    )
    run_offline(cfg, bank=bank)

    from dentex_bench.augment import offline
    orig_process = offline._process_image_variants
    call_count = {"count": 0}

    def counting_process(*args, **kwargs):
        call_count["count"] += 1
        return orig_process(*args, **kwargs)

    monkeypatch.setattr(offline, "_process_image_variants", counting_process)

    # With overwrite=True, it should re-run all images
    cfg.output.overwrite = True
    run_offline(cfg, bank=bank)
    assert call_count["count"] == 2

