# tests/augment/test_sampling.py
import math
import numpy as np
import pytest

from dentex_bench.augment.sampling import (
    compute_repeat_factors,
    suggest_t,
    compute_image_repeat_factors,
    sample_rfs_indices,
)
from dentex_bench.augment.rng import rng_for


def test_compute_repeat_factors():
    """Verify r_c calculation: r_c = max(1.0, sqrt(t / f_c))."""
    cat_counts = {"caries": 500, "periapical": 50, "empty": 0}
    total_imgs = 1000
    factors = compute_repeat_factors(cat_counts, total_imgs, t=0.1)

    # f_periapical = 50 / 1000 = 0.05 -> r = sqrt(0.1 / 0.05) = sqrt(2) ~ 1.4142
    assert abs(factors["periapical"] - math.sqrt(2.0)) < 1e-4

    # f_caries = 500 / 1000 = 0.5 -> sqrt(0.1 / 0.5) = sqrt(0.2) ~ 0.447 < 1.0 -> r = 1.0
    assert factors["caries"] == 1.0

    # f_empty = 0 -> clamped to 1e-6 -> sqrt(0.1 / 1e-6) = sqrt(100000) ~ 316.227
    assert factors["empty"] > 300.0


def test_suggest_t_inversion():
    """Verify inverse calculation solving for t: t = (target_repeat ** 2) * f_c."""
    cat_counts = {"periapical": 100, "caries": 600}
    total_imgs = 1000  # f_periapical = 0.1

    # Solve for target_repeat = 2.0 -> t = 4.0 * 0.1 = 0.4
    t = suggest_t("periapical", target_repeat=2.0, category_counts=cat_counts, total_images=total_imgs)
    assert abs(t - 0.4) < 1e-4

    # Confirm consistency: computing repeat factor with this t yields exactly target_repeat
    factors = compute_repeat_factors(cat_counts, total_imgs, t=t)
    assert abs(factors["periapical"] - 2.0) < 1e-4

    # Solve for target_repeat = 2.5
    t2 = suggest_t("periapical", target_repeat=2.5, category_counts=cat_counts, total_images=total_imgs)
    factors2 = compute_repeat_factors(cat_counts, total_imgs, t=t2)
    assert abs(factors2["periapical"] - 2.5) < 1e-4


def test_compute_image_repeat_factors():
    """Verify image-level repeat factor: R_i = max_{c in image} r_c (or 1.0 if empty)."""
    images = {
        101: ["caries"],
        102: ["periapical"],
        103: ["caries", "periapical"],
        104: [],  # Healthy/no pathology
    }
    # When category_counts is passed explicitly:
    cat_counts = {"caries": 500, "periapical": 50}
    total_imgs = 1000  # r_caries = 1.0, r_periapical = sqrt(2) ~ 1.4142

    img_factors = compute_image_repeat_factors(
        images_with_categories=images,
        total_images=total_imgs,
        t=0.1,
        category_counts=cat_counts,
    )

    assert abs(img_factors[101] - 1.0) < 1e-4
    assert abs(img_factors[102] - math.sqrt(2.0)) < 1e-4
    # max(1.0, sqrt(2)) = sqrt(2)
    assert abs(img_factors[103] - math.sqrt(2.0)) < 1e-4
    # Empty annotations defaults to 1.0
    assert img_factors[104] == 1.0

    # Also test auto-calculating category_counts from image annotations
    auto_factors = compute_image_repeat_factors(
        images_with_categories=images,
        total_images=len(images),
        t=0.5,
    )
    # caries appears in 2 images (101, 103) -> f = 2/4 = 0.5 -> r = sqrt(0.5/0.5) = 1.0
    # periapical appears in 2 images (102, 103) -> f = 2/4 = 0.5 -> r = 1.0
    assert auto_factors[101] == 1.0
    assert auto_factors[104] == 1.0


def test_sample_rfs_indices_determinism():
    """Verify deterministic sampling with seeded RNG."""
    image_repeat_factors = {
        1: 1.0,   # floor=1, frac=0.0 -> always 1
        2: 2.0,   # floor=2, frac=0.0 -> always 2
        3: 1.4,   # floor=1, frac=0.4 -> 1 or 2
        4: 2.7,   # floor=2, frac=0.7 -> 2 or 3
    }

    rng1 = rng_for(42, "rfs", 0)
    rng2 = rng_for(42, "rfs", 0)

    indices1 = sample_rfs_indices(image_repeat_factors, rng1)
    indices2 = sample_rfs_indices(image_repeat_factors, rng2)

    # Identical seed must yield bit-exact identical indices
    assert indices1 == indices2
    assert indices1.count(1) == 1
    assert indices1.count(2) == 2
    assert indices1.count(3) in (1, 2)
    assert indices1.count(4) in (2, 3)

    # Different seed sequence should yield independent valid results
    rng3 = rng_for(99, "rfs", 1)
    indices3 = sample_rfs_indices(image_repeat_factors, rng3)
    assert indices3.count(1) == 1
    assert indices3.count(2) == 2


def test_sample_rfs_indices_statistical_property():
    """Verify stochastic floor/ceil Bernoulli expectation over many trials."""
    image_repeat_factors = {10: 1.4}
    rng = np.random.default_rng(2026)
    
    total_samples = 5000
    counts = 0
    for _ in range(total_samples):
        res = sample_rfs_indices(image_repeat_factors, rng)
        counts += res.count(10)
        
    empirical_mean = counts / total_samples
    # Expected value is 1.4
    assert abs(empirical_mean - 1.4) < 0.05
