"""Repeat Factor Sampling (RFS) calculations and utilities.

Implements category-level repeat factor sampling (Gupta et al., CVPR 2019 / LVIS),
image-level repeat factor aggregation, inverse threshold solver (suggest_t),
and deterministic/stochastic sample index generation.
"""

import math
from typing import Any, Dict, List, Optional


def compute_repeat_factors(
    category_counts: Dict[str, int],
    total_images: int,
    t: float,
) -> Dict[str, float]:
    """Compute Repeat Factor Sampling (RFS) multipliers for categories:

    Formula:
        f_c = max(1e-6, count / max(1, total_images))
        r_c = max(1.0, math.sqrt(max(0.0, t) / f_c))

    Args:
        category_counts: Mapping from category name (or id) to image occurrence count.
        total_images: Total number of training images.
        t: Repeat factor threshold parameter (e.g. 0.12).

    Returns:
        Dict mapping category name to repeat factor r_c (>= 1.0).
    """
    factors: Dict[str, float] = {}
    denom = max(1, total_images)
    for cat, count in category_counts.items():
        f_c = max(1e-6, float(count) / denom)
        factors[cat] = max(1.0, math.sqrt(max(0.0, t) / f_c))
    return factors


def suggest_t(
    target_class: str,
    target_repeat: float,
    category_counts: Dict[str, int],
    total_images: int,
) -> float:
    """Invert target repeat factor to solve for required threshold t.

    Since r_c = sqrt(t / f_c), we have:
        t = (target_repeat ** 2) * f_c

    Args:
        target_class: Target category identifier.
        target_repeat: Desired repeat factor for the target category (e.g. 2.0 or 2.5).
        category_counts: Mapping from category name to occurrence count.
        total_images: Total number of training images.

    Returns:
        Calculated threshold t as a float.
    """
    count = category_counts.get(target_class, 1)
    denom = max(1, total_images)
    f_c = max(1e-6, float(count) / denom)
    return float((target_repeat ** 2) * f_c)


def compute_image_repeat_factors(
    images_with_categories: Dict[int, List[str]],
    total_images: int,
    t: float,
    category_counts: Optional[Dict[str, int]] = None,
) -> Dict[int, float]:
    """Compute image-level repeat factors R_i across all categories in each image:

    For image i:
        R_i = max_{c in image} r_c (or 1.0 if empty / no categories).

    Args:
        images_with_categories: Mapping from image_id to list of category names on the image.
        total_images: Total number of images in dataset.
        t: Repeat factor threshold parameter.
        category_counts: Optional precomputed category counts. If None, computed from
            images_with_categories as the number of unique images containing category c.

    Returns:
        Dict mapping image_id to float repeat factor R_i (>= 1.0).
    """
    if category_counts is None:
        cat_counts: Dict[str, int] = {}
        for cats in images_with_categories.values():
            for c in set(cats):
                cat_counts[c] = cat_counts.get(c, 0) + 1
    else:
        cat_counts = category_counts

    factors = compute_repeat_factors(cat_counts, total_images, t)

    image_factors: Dict[int, float] = {}
    for img_id, cats in images_with_categories.items():
        if not cats:
            image_factors[img_id] = 1.0
        else:
            image_factors[img_id] = float(max((factors.get(c, 1.0) for c in cats), default=1.0))

    return image_factors


def sample_rfs_indices(
    image_repeat_factors: Dict[int, float],
    rng: Any,
) -> List[int]:
    """Stochastic floor/ceil sampling of image indices based on repeat factors.

    For each image i with repeat factor R_i:
        repeat count is floor(R_i) + (1 if rng.random() < (R_i - floor(R_i)) else 0)

    Iterates over image_id in sorted key order for cross-platform deterministic reproducibility.

    Args:
        image_repeat_factors: Mapping from image_id to repeat factor R_i.
        rng: Random number generator (np.random.Generator, random.Random, or seed int).

    Returns:
        List of image IDs with repeated indices according to stochastic RFS.
    """
    if isinstance(rng, int):
        import numpy as np
        rng = np.random.default_rng(rng)

    sampled: List[int] = []
    # Sort keys for deterministic execution across dict implementations
    for img_id in sorted(image_repeat_factors.keys()):
        r = float(image_repeat_factors[img_id])
        floor_r = math.floor(r)
        frac = r - floor_r

        rand_val = rng.random() if hasattr(rng, "random") else rng.uniform(0.0, 1.0)
        count = floor_r + (1 if rand_val < frac else 0)
        sampled.extend([img_id] * count)

    return sampled
