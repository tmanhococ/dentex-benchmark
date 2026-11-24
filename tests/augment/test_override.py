# tests/augment/test_override.py
import numpy as np
import pytest

from dentex_bench.augment.recipe import execute, Recipe
from dentex_bench.augment.bank import ToothBank, ToothEntry
from real_bank import make_real_bank


def _make_dummy_bank() -> ToothBank:
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


def _make_test_recipe(blend_mode: str = "hard") -> Recipe:
    return Recipe(
        recipe_id="test_rec_001",
        schema_version=1,
        rng_key=["offline", 1, 0],
        target_image_id=1,
        target_file="sample.png",
        target_file_sha256="",
        ops=[
            {
                "op": "swap",
                "source_entry_id": "bank_1",
                "target_fdi": 16,
                "target_tooth_ref": {"bbox": [50.0, 50.0, 40.0, 40.0]},
                "mirror": False,
                "scale_jitter": 1.0,
                "rotate_jitter": 0.0,
                "blend_mode": blend_mode,
                "blend": {"mode": blend_mode, "feather_sigma": 5.0, "core_erode_px": 6},
                "inpaint": {"enabled": False},
                "intensity": {"mode": "none"},
            }
        ],
    )


def test_override_disallows_placement_change():
    """Attempting to override placement.* or geometry.* in replay must raise ValueError."""
    forbidden_placement = "placement.mode=random"
    with pytest.raises(ValueError, match="execution-only"):
        from scripts.replay_augment import validate_overrides
        validate_overrides([forbidden_placement])

    forbidden_geometry = "geometry.jitter_scale=0.1"
    with pytest.raises(ValueError, match="execution-only"):
        from scripts.replay_augment import validate_overrides
        validate_overrides([forbidden_geometry])

    forbidden_bank = "bank.crop_margin_px=10"
    with pytest.raises(ValueError, match="execution-only"):
        from scripts.replay_augment import validate_overrides
        validate_overrides([forbidden_bank])


def test_override_allowed_prefixes():
    """Execution-only overrides must pass validation without error."""
    from scripts.replay_augment import validate_overrides

    allowed = [
        "blend.mode=feather",
        "blend.feather_sigma=3.0",
        "blend.core_erode_px=4",
        "intensity_match=histogram",
        "intensity_match.mode=meanstd",
        "noise_match=false",
        "extras.photometric.clahe=true",
        "extras.patch_crop.sizes=1024",
    ]
    # Should not raise
    validate_overrides(allowed)


def test_override_blend_mode():
    """Applying blend.mode override modifies recipe and executes cleanly."""
    from scripts.replay_augment import apply_overrides

    recipe = _make_test_recipe(blend_mode="hard")
    bank = _make_dummy_bank()

    dummy_img = np.full((120, 120, 3), 100, dtype=np.uint8)
    loader = lambda iid: (dummy_img.copy(), [])

    # Execute original with hard blend
    img_hard, anns_hard, meta_hard = execute(recipe, loader, bank)
    assert meta_hard["status"] == "success"

    # Override blend.mode to feather
    overridden_rec = apply_overrides(recipe, ["blend.mode=feather"])
    assert overridden_rec.ops[0]["blend_mode"] == "feather"
    assert overridden_rec.ops[0]["blend"]["mode"] == "feather"

    img_feather, anns_feather, meta_feather = execute(overridden_rec, loader, bank)
    assert meta_feather["status"] == "success"
    assert img_feather.shape == img_hard.shape
    assert img_feather.dtype == img_hard.dtype

    # Due to feather vs hard blending boundary, the pixels along boundary should differ
    assert not np.array_equal(img_hard, img_feather)


def test_override_intensity_match():
    """Applying intensity_match override modifies recipe and execution cleanly."""
    from scripts.replay_augment import apply_overrides

    recipe = _make_test_recipe(blend_mode="hard")
    assert recipe.ops[0]["intensity"]["mode"] == "none"

    overridden = apply_overrides(recipe, ["intensity_match=meanstd"])
    assert overridden.ops[0]["intensity"]["mode"] == "meanstd"
