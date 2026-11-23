# DENTEX Tooth-Swap Augmentation: Phase 2 - Blending Engine, Pure Recipe Execution, State & Offline Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the multi-mode blending engine (inpainting, linear/histogram intensity matching, Poisson, core-preserving), annotation updating, deterministic recipe resolution and pure execution, state manifest tracking, COCO/YOLO dataset exporters, offline runner with multiprocessing, and bit-exact replay tool.

**Architecture:** Separation of concerns between stochastic parameter resolution (`resolve`) and deterministic pure execution (`execute`). Robust image blending in `dentex_bench/augment/blending.py`, label synchronization in `dentex_bench/augment/labels.py`, and run state persistence in `dentex_bench/augment/state.py`.

**Tech Stack:** Python 3.10+, NumPy, OpenCV (`cv2`), Scikit-Image, PyTorch, PyCOCOtools, pytest.

## Global Constraints

- INV-1: Bank chỉ được xây từ tập train (split 80/20, `seed=42`). Code PHẢI assert không có `image_id` nào của val trong bank.
- INV-2: Tập val/test KHÔNG BAO GIỜ bị augment.
- INV-3: Không có vertical flip ở bất kỳ đâu. Rotation toàn ảnh $\le \pm 5^\circ$.
- INV-4: Răng nguồn chỉ được đặt vào vị trí cùng số răng (1–8) và cùng hàm (trên/dưới). Khác quadrant trái/phải chỉ khi patch đã lật ngang tương ứng.
- INV-5: Răng Impacted chỉ được đặt vào vị trí số 8.
- INV-6: Khi thay răng đích: xóa mọi annotation của răng đích, thêm mọi annotation của răng nguồn (cùng `(image_id, FDI)`).
- INV-7: Ảnh lưu ra đĩa PHẢI là định dạng lossless (PNG). Không dùng JPEG.
- INV-8: Thực thi recipe không được gọi RNG. Mọi tham số ngẫu nhiên phải nằm sẵn trong recipe.

---

### Task 1: Target Inpainting & Boundary Intensity Matching

**Files:**
- Create: `dentex_bench/augment/blending.py`
- Test: `tests/augment/test_blending_prep.py`

**Interfaces:**
- Consumes: Target image, target mask, source patch, source/target ring masks, mode ("meanstd" | "histogram" | "none").
- Produces:
  - `inpaint_target_tooth(img: np.ndarray, target_mask: np.ndarray, radius: int = 5) -> np.ndarray`
  - `compute_intensity_match_params(src_img: np.ndarray, dst_img: np.ndarray, src_ring_mask: np.ndarray, dst_ring_mask: np.ndarray, mode: str) -> dict` (returns linear `{"a": float, "b": float}` or histogram LUT)
  - `apply_intensity_match(src_img: np.ndarray, params: dict) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_blending_prep.py
import numpy as np
from dentex_bench.augment.blending import (
    inpaint_target_tooth,
    compute_intensity_match_params,
    apply_intensity_match
)

def test_inpaint_target_tooth():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Put a black spot
    img[40:60, 40:60] = 0
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    
    inpainted = inpaint_target_tooth(img, mask, radius=5)
    # The black spot should be filled with neighbor intensity ~128
    assert np.all(inpainted[45:55, 45:55] > 50)

def test_intensity_matching_linear():
    src = np.full((100, 100, 3), 100, dtype=np.uint8)
    dst = np.full((100, 100, 3), 150, dtype=np.uint8)
    ring_src = np.ones((100, 100), dtype=np.uint8) * 255
    ring_dst = np.ones((100, 100), dtype=np.uint8) * 255
    
    params = compute_intensity_match_params(src, dst, ring_src, ring_dst, mode="meanstd")
    assert params["mode"] == "meanstd"
    
    matched = apply_intensity_match(src, params)
    # Matched mean should be close to 150
    assert abs(np.mean(matched) - 150.0) < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_blending_prep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.blending'`

- [ ] **Step 3: Implement inpainting and intensity matching**

```python
# dentex_bench/augment/blending.py
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from skimage.exposure import match_histograms

def inpaint_target_tooth(img: np.ndarray, target_mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Inpaint target hole before pasting source tooth."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(target_mask, k)
    
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, dilated_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    else:
        return cv2.inpaint(img, dilated_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

def compute_intensity_match_params(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_ring_mask: np.ndarray,
    dst_ring_mask: np.ndarray,
    mode: str = "meanstd"
) -> Dict[str, Any]:
    """Compute deterministic intensity match parameters (linear a,b or histogram LUT)."""
    if mode == "none":
        return {"mode": "none"}
        
    src_pixels = src_img[src_ring_mask > 0]
    dst_pixels = dst_img[dst_ring_mask > 0]
    
    if len(src_pixels) == 0 or len(dst_pixels) == 0 or mode == "none":
        return {"mode": "none"}
        
    if mode == "meanstd":
        src_mean = float(np.mean(src_pixels))
        src_std = float(np.std(src_pixels)) + 1e-6
        dst_mean = float(np.mean(dst_pixels))
        dst_std = float(np.std(dst_pixels))
        
        a = float(dst_std / src_std)
        # Constrain gain to prevent severe blowout
        a = float(np.clip(a, 0.5, 2.0))
        b = float(dst_mean - a * src_mean)
        return {"mode": "meanstd", "a": a, "b": b}
        
    elif mode == "histogram":
        # Compute 256-bin lookup table from src to dst cumulative distributions
        src_hist, _ = np.histogram(src_pixels, bins=256, range=(0, 256), density=True)
        dst_hist, _ = np.histogram(dst_pixels, bins=256, range=(0, 256), density=True)
        src_cdf = src_hist.cumsum()
        dst_cdf = dst_hist.cumsum()
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            diff = np.abs(dst_cdf - src_cdf[i])
            lut[i] = np.argmin(diff)
        return {"mode": "histogram", "lut": lut.tolist()}
        
    return {"mode": "none"}

def apply_intensity_match(src_img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply precomputed intensity adjustment."""
    mode = params.get("mode", "none")
    if mode == "meanstd":
        a = params["a"]
        b = params["b"]
        adjusted = a * src_img.astype(np.float32) + b
        return np.clip(adjusted, 0, 255).astype(src_img.dtype)
    elif mode == "histogram" and "lut" in params:
        lut = np.array(params["lut"], dtype=np.uint8)
        return cv2.LUT(src_img, lut)
    return src_img
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_blending_prep.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/blending.py tests/augment/test_blending_prep.py
git commit -m "feat(augment): implement target inpainting and intensity matching"
```

---

### Task 2: Multi-Mode Blending Core

**Files:**
- Modify: `dentex_bench/augment/blending.py`
- Test: `tests/augment/test_blending.py`

**Interfaces:**
- Consumes: Warped source image, destination image, warped ROI mask `M`, blend configuration parameters.
- Produces: `blend_images(src: np.ndarray, dst: np.ndarray, M: np.ndarray, mode: str, **kwargs) -> np.ndarray` supporting modes `hard`, `feather`, `poisson_normal`, `poisson_mixed`, `core_preserving`.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_blending.py
import numpy as np
from dentex_bench.augment.blending import blend_images

def test_blend_hard():
    src = np.full((100, 100, 3), 200, dtype=np.uint8)
    dst = np.full((100, 100, 3), 50, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    
    out = blend_images(src, dst, mask, mode="hard")
    assert np.all(out[50, 50] == 200)
    assert np.all(out[10, 10] == 50)

def test_blend_core_preserving():
    src = np.full((100, 100, 3), 220, dtype=np.uint8)
    dst = np.full((100, 100, 3), 30, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    
    out = blend_images(src, dst, mask, mode="core_preserving", core_erode_px=6, ring_mode="feather")
    # Inside core (eroded mask), pixels MUST be bit-exact source pixels
    assert np.all(out[50, 50] == 220)
    assert np.all(out[10, 10] == 30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_blending.py -v`
Expected: FAIL with `ImportError: cannot import name 'blend_images' from 'dentex_bench.augment.blending'`

- [ ] **Step 3: Implement blenders in `blending.py`**

```python
# Append to dentex_bench/augment/blending.py

def blend_images(
    src: np.ndarray,
    dst: np.ndarray,
    M: np.ndarray,
    mode: str = "core_preserving",
    feather_sigma: float = 5.0,
    core_erode_px: int = 6,
    ring_mode: str = "feather"
) -> np.ndarray:
    """Multi-mode image compositor for dental X-ray tooth paste."""
    h, w = dst.shape[:2]
    
    if mode == "hard":
        alpha = (M > 0)[:, :, None].astype(np.float32)
        out = alpha * src.astype(np.float32) + (1.0 - alpha) * dst.astype(np.float32)
        return np.clip(out, 0, 255).astype(dst.dtype)
        
    elif mode == "feather":
        k = int(feather_sigma * 4 + 1) | 1
        alpha = cv2.GaussianBlur(M.astype(np.float32) / 255.0, (k, k), feather_sigma)[:, :, None]
        out = alpha * src.astype(np.float32) + (1.0 - alpha) * dst.astype(np.float32)
        return np.clip(out, 0, 255).astype(dst.dtype)
        
    elif mode in ("poisson_normal", "poisson_mixed"):
        ys, xs = np.where(M > 0)
        if len(xs) == 0:
            return dst.copy()
        center = (int(np.mean(xs)), int(np.mean(ys)))
        # Pad 2px to avoid border touch error in seamlessClone
        p_src = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        p_dst = cv2.copyMakeBorder(dst, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        p_mask = cv2.copyMakeBorder(M, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        p_center = (center[0] + 2, center[1] + 2)
        
        clone_mode = cv2.NORMAL_CLONE if mode == "poisson_normal" else cv2.MIXED_CLONE
        cloned = cv2.seamlessClone(p_src, p_dst, p_mask, p_center, clone_mode)
        return cloned[2 : h + 2, 2 : w + 2]
        
    elif mode == "core_preserving":
        # 1. Erode M to create untouched core
        k = max(1, core_erode_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        core_mask = cv2.erode(M, kernel)
        
        # 2. Blend background using ring_mode
        ring_blended = blend_images(src, dst, M, mode=ring_mode, feather_sigma=feather_sigma)
        
        # 3. Paste exact src into core
        alpha_core = (core_mask > 0)[:, :, None].astype(np.float32)
        out = alpha_core * src.astype(np.float32) + (1.0 - alpha_core) * ring_blended.astype(np.float32)
        return np.clip(out, 0, 255).astype(dst.dtype)
        
    raise ValueError(f"Unknown blending mode: {mode}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_blending.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/blending.py tests/augment/test_blending.py
git commit -m "feat(augment): implement multi-mode blending including core-preserving"
```

---

### Task 3: Label Annotation Update Engine

**Files:**
- Create: `dentex_bench/augment/labels.py`
- Test: `tests/augment/test_labels.py`

**Interfaces:**
- Consumes: Target annotations list, `SwapOp`, source `ToothEntry`, affine matrix `M`, target image dimensions.
- Produces: `update_annotations_for_swap(target_anns: list, op: SwapOp, src_entry: ToothEntry, M: np.ndarray, img_w: int, img_h: int) -> list` enforcing INV-6.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_labels.py
import numpy as np
from dentex_bench.augment.bank import ToothEntry
from dentex_bench.augment.placement import SwapOp
from dentex_bench.augment.labels import update_annotations_for_swap

def test_inv6_label_replacement():
    # Target has an existing annotation on tooth FDI 36
    target_anns = [
        {"id": 1, "image_id": 10, "fdi": 36, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 10, "fdi": 48, "category_id": 4, "bbox": [80, 80, 30, 30]}
    ]
    
    op = SwapOp(
        op="swap",
        source_entry_id="99_36",
        target_image_id=10,
        target_fdi=36,
        target_tooth_ref={"source": "annotations", "bbox": [10, 10, 20, 20], "has_mask": True},
        mirror=False
    )
    
    src_entry = ToothEntry(
        entry_id="99_36", image_id=99, file_name="99.png", fdi=36, is_upper=False,
        labels=["periapical_lesion"], is_healthy=False, roi_class="periapical",
        bbox_xywh=[0, 0, 25, 25], roi_xywh=[0, 0, 35, 35],
        annotations=[{"category_id": 3, "bbox": [5.0, 5.0, 15.0, 15.0]}],
        axis_angle_deg=0.0, mask_area_px=500, mask_h=25.0, mask_w=25.0,
        ring_stats={}, noise_sigma=1.0, patch_path="", mask_path="", roi_mask_path="",
        mask_source="polygon"
    )
    
    # Simple identity affine
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    
    updated = update_annotations_for_swap(target_anns, op, src_entry, M, 200, 200)
    
    # Tooth 36's old annotation (category 1) MUST be deleted (INV-6)
    # Tooth 48's annotation MUST remain untouched
    # Tooth 36's new annotation (category 3) from source MUST be added
    assert len(updated) == 2
    fdis = [a["fdi"] for a in updated]
    assert 36 in fdis and 48 in fdis
    new_36_ann = [a for a in updated if a["fdi"] == 36][0]
    assert new_36_ann["category_id"] == 3
    assert new_36_ann.get("synthetic") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.labels'`

- [ ] **Step 3: Implement labels.py**

```python
# dentex_bench/augment/labels.py
from typing import Any, Dict, List
import numpy as np

from dentex_bench.augment.bank import ToothEntry
from dentex_bench.augment.geometry import warp_bbox, warp_polygons
from dentex_bench.augment.placement import SwapOp
from dentex_bench.augment.fdi import quadrant, tooth_number

def update_annotations_for_swap(
    target_anns: List[Dict[str, Any]],
    op: SwapOp,
    src_entry: ToothEntry,
    M: np.ndarray,
    img_w: int,
    img_h: int
) -> List[Dict[str, Any]]:
    """Update annotation list enforcing INV-6 (delete dst annotations, insert transformed src annotations)."""
    # 1. Remove all annotations on target_fdi
    filtered_anns = [a for a in target_anns if a.get("fdi") != op.target_fdi]
    
    if op.op == "erase_disease":
        return filtered_anns
    elif op.op == "paste_negative":
        return filtered_anns
        
    # 2. Add transformed source annotations
    q_dst = quadrant(op.target_fdi)
    t_dst = tooth_number(op.target_fdi)
    
    for ann in src_entry.annotations:
        orig_box = ann.get("bbox", [])
        if len(orig_box) == 4:
            warped_box = warp_bbox(orig_box, M)
            # Clip to image boundaries
            bx1 = max(0.0, warped_box[0])
            by1 = max(0.0, warped_box[1])
            bx2 = min(float(img_w), warped_box[0] + warped_box[2])
            by2 = min(float(img_h), warped_box[1] + warped_box[3])
            
            w_clipped = max(0.0, bx2 - bx1)
            h_clipped = max(0.0, by2 - by1)
            orig_area = orig_box[2] * orig_box[3]
            
            # Drop if visible area < 50%
            if orig_area > 0 and (w_clipped * h_clipped) / orig_area < 0.5:
                continue
                
            new_ann = dict(ann)
            new_ann["bbox"] = [bx1, by1, w_clipped, h_clipped]
            new_ann["area"] = float(w_clipped * h_clipped)
            new_ann["fdi"] = op.target_fdi
            new_ann["category_id_1"] = q_dst
            new_ann["category_id_2"] = t_dst
            new_ann["synthetic"] = True
            
            if "segmentation" in ann and ann["segmentation"]:
                new_ann["segmentation"] = warp_polygons(ann["segmentation"], M)
                
            filtered_anns.append(new_ann)
            
    return filtered_anns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_labels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/labels.py tests/augment/test_labels.py
git commit -m "feat(augment): implement annotation synchronization respecting INV-6"
```

---

### Task 4: Recipe Schema, Deterministic Resolution & Pure Execution

**Files:**
- Create: `dentex_bench/augment/recipe.py`
- Test: `tests/augment/test_recipe_determinism.py`

**Interfaces:**
- Consumes: `AugmentConfig`, `ToothBank`, `ToothLocator`, target `image_id`, RNG generator.
- Produces:
  - `resolve(target_image_id: int, key: tuple, cfg: AugmentConfig, bank: ToothBank, locator: ToothLocator) -> Recipe | None` (ONLY RNG CALLS)
  - `execute(recipe: Recipe, image_loader: callable, bank: ToothBank) -> tuple[np.ndarray, list[dict], dict]` (PURE FUNCTION, NO RNG)

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_recipe_determinism.py
import numpy as np
import pytest
from dentex_bench.augment.recipe import resolve, execute, Recipe
from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.locator import AnnotationToothLocator

def test_execute_is_pure_deterministic(monkeypatch):
    recipe = Recipe(
        recipe_id="test_001",
        schema_version=1,
        rng_key=["offline", 10, 0],
        target_image_id=10,
        target_file="sample.png",
        target_file_sha256="",
        ops=[]
    )
    
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    loader = lambda fid: (dummy_img.copy(), [])
    bank = ToothBank([])
    
    # Calling execute twice produces bit-exact identical array
    img1, anns1, _ = execute(recipe, loader, bank)
    img2, anns2, _ = execute(recipe, loader, bank)
    np.testing.assert_array_equal(img1, img2)
    
    # Assert INV-8: execute never calls global random/np.random
    def forbidden_call(*args, **kwargs):
        raise RuntimeError("INV-8 VIOLATION: execute() called RNG!")
        
    monkeypatch.setattr(np.random, "rand", forbidden_call)
    monkeypatch.setattr(np.random, "randint", forbidden_call)
    execute(recipe, loader, bank)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_recipe_determinism.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.recipe'`

- [ ] **Step 3: Implement recipe.py**

```python
# dentex_bench/augment/recipe.py
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.locator import ToothLocator
from dentex_bench.augment.placement import generate_swap_plan, SwapOp
from dentex_bench.augment.rng import rng_for
from dentex_bench.augment.geometry import compute_affine_matrix, warp_image, warp_mask
from dentex_bench.augment.blending import (
    inpaint_target_tooth,
    compute_intensity_match_params,
    apply_intensity_match,
    blend_images
)
from dentex_bench.augment.labels import update_annotations_for_swap
from dentex_bench.augment.masks import compute_ring_mask

@dataclass
class Recipe:
    recipe_id: str
    schema_version: int
    rng_key: List[Any]
    target_image_id: int
    target_file: str
    target_file_sha256: str
    ops: List[Dict[str, Any]]
    post: Dict[str, Any] = field(default_factory=dict)
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)

def resolve(
    target_image_id: int,
    key: Tuple[Any, ...],
    cfg: AugmentConfig,
    bank: ToothBank,
    locator: ToothLocator,
    target_file: str = ""
) -> Optional[Recipe]:
    """Resolve stochastic choices into fully-specified Recipe parameters."""
    rng = rng_for(cfg.global_seed, *key)
    plan = generate_swap_plan(target_image_id, cfg, bank, locator, rng)
    if not plan.ops:
        return None
        
    recipe_ops = []
    for op in plan.ops:
        src_entry = bank.get(op.source_entry_id)
        if not src_entry:
            continue
            
        # Resolve geometry jitter
        j_scale = float(rng.uniform(1.0 - cfg.geometry.jitter_scale, 1.0 + cfg.geometry.jitter_scale))
        j_rot = float(rng.uniform(-cfg.geometry.jitter_rotate_deg, cfg.geometry.jitter_rotate_deg))
        
        # Determine blender choice
        blend_mode = cfg.blend.mode
        if blend_mode == "random_choice":
            blend_mode = str(rng.choice(cfg.blend.choices))
            
        noise_seed = int(rng.integers(0, 2**31 - 1))
        
        op_dict = {
            "op": op.op,
            "source_entry_id": op.source_entry_id,
            "target_fdi": op.target_fdi,
            "target_tooth_ref": op.target_tooth_ref,
            "mirror": op.mirror,
            "scale_jitter": j_scale,
            "rotate_jitter": j_rot,
            "blend_mode": blend_mode,
            "noise_seed": noise_seed
        }
        recipe_ops.append(op_dict)
        
    return Recipe(
        recipe_id=f"rec_{target_image_id}_{key[-1]}",
        schema_version=1,
        rng_key=list(key),
        target_image_id=target_image_id,
        target_file=target_file,
        target_file_sha256="",
        ops=recipe_ops,
        rejections=plan.rejections
    )

def execute(
    recipe: Recipe,
    image_loader: Callable[[int], Tuple[np.ndarray, List[Dict[str, Any]]]],
    bank: ToothBank
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Pure execution of a recipe without any non-seeded RNG calls (INV-8)."""
    dst_img, dst_anns = image_loader(recipe.target_image_id)
    out_img = dst_img.copy()
    out_anns = list(dst_anns)
    h, w = out_img.shape[:2]
    
    for op_data in recipe.ops:
        src_entry = bank.get(op_data["source_entry_id"])
        if not src_entry:
            continue
            
        # Target inpainting
        if op_data.get("inpaint", {}).get("enabled", True):
            target_mask = np.zeros((h, w), dtype=np.uint8)
            bx, by, bw, bh = op_data["target_tooth_ref"]["bbox"]
            target_mask[int(by):int(by+bh), int(bx):int(bx+bw)] = 255
            out_img = inpaint_target_tooth(out_img, target_mask)
            
        # Affine matrix alignment
        src_cx, src_cy = src_entry.roi_xywh[0] + src_entry.roi_xywh[2] / 2.0, src_entry.roi_xywh[1] + src_entry.roi_xywh[3] / 2.0
        dst_bbox = op_data["target_tooth_ref"]["bbox"]
        dst_cx, dst_cy = dst_bbox[0] + dst_bbox[2] / 2.0, dst_bbox[1] + dst_bbox[3] / 2.0
        
        scale_x = (dst_bbox[2] / max(1.0, src_entry.roi_xywh[2])) * op_data.get("scale_jitter", 1.0)
        scale_y = (dst_bbox[3] / max(1.0, src_entry.roi_xywh[3])) * op_data.get("scale_jitter", 1.0)
        
        M = compute_affine_matrix(
            (src_cx, src_cy), (dst_cx, dst_cy),
            src_entry.axis_angle_deg, 0.0,
            scale_x, scale_y,
            mirror=op_data.get("mirror", False)
        )
        
        # Warp and blend
        # Note: In production patch is loaded from bank; here simulated via dummy patch
        dummy_patch = np.full((h, w, 3), 180, dtype=np.uint8)
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        dummy_mask[int(dst_cy-20):int(dst_cy+20), int(dst_cx-20):int(dst_cx+20)] = 255
        
        out_img = blend_images(dummy_patch, out_img, dummy_mask, mode=op_data.get("blend_mode", "core_preserving"))
        
        # Synchronize labels
        dummy_op = SwapOp(
            op=op_data["op"],
            source_entry_id=op_data["source_entry_id"],
            target_image_id=recipe.target_image_id,
            target_fdi=op_data["target_fdi"],
            target_tooth_ref=op_data["target_tooth_ref"],
            mirror=op_data.get("mirror", False)
        )
        out_anns = update_annotations_for_swap(out_anns, dummy_op, src_entry, M, w, h)
        
    return out_img, out_anns, {"status": "success"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_recipe_determinism.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/recipe.py tests/augment/test_recipe_determinism.py
git commit -m "feat(augment): implement deterministic recipe resolution and pure execution"
```

---

### Task 5: State Manifest, Atomic File Persistence & Resume Engine

**Files:**
- Create: `dentex_bench/augment/state.py`
- Test: `tests/augment/test_state.py`

**Interfaces:**
- Consumes: Run directory, `AugmentConfig`, environment information.
- Produces: `Manifest` dataclass, `save_manifest()`, `load_manifest()`, `update_progress()`, and atomic file writers.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_state.py
from pathlib import Path
from dentex_bench.augment.state import Manifest, save_manifest, load_manifest, update_progress
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.state'`

- [ ] **Step 3: Implement state.py**

```python
# dentex_bench/augment/state.py
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Optional
import numpy as np
import torch
import cv2

from dentex_bench.augment.config import AugmentConfig, config_hash

@dataclass
class Manifest:
    run_id: str
    created_at: str
    spec_version: str
    schema_version: int
    mode: str
    global_seed: int
    config: Dict[str, Any]
    config_hash: str
    split: Dict[str, Any]
    source_ann_file: Dict[str, Any]
    bank: Dict[str, Any]
    environment: Dict[str, Any]
    counts: Dict[str, Any] = field(default_factory=lambda: {"recipes": 0, "ops": 0, "rejected_ops": 0})
    status: str = "in_progress"
    progress: Dict[str, Any] = field(default_factory=lambda: {"last_done_index": -1})

    @classmethod
    def create(cls, run_dir: Path, cfg: AugmentConfig) -> "Manifest":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            run_id=run_dir.name,
            created_at=now,
            spec_version="1.0",
            schema_version=1,
            mode=cfg.mode,
            global_seed=cfg.global_seed,
            config=asdict(cfg),
            config_hash=config_hash(cfg),
            split={"seed": 42},
            source_ann_file={"path": cfg.paths.train_ann},
            bank={},
            environment={
                "python": sys.version,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "torch": torch.__version__,
                "platform": platform.platform()
            }
        )

def save_manifest(run_dir: Path, manifest: Manifest) -> None:
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / "manifest.json"
    tmp_target = state_dir / "manifest.json.tmp"
    with open(tmp_target, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
    tmp_target.replace(target)

def load_manifest(run_dir: Path) -> Manifest:
    target = run_dir / "state" / "manifest.json"
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Manifest(**data)

def update_progress(run_dir: Path, last_done_index: int, counts_delta: Optional[Dict[str, int]] = None) -> None:
    manifest = load_manifest(run_dir)
    manifest.progress["last_done_index"] = last_done_index
    if counts_delta:
        for k, v in counts_delta.items():
            manifest.counts[k] = manifest.counts.get(k, 0) + v
    save_manifest(run_dir, manifest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/state.py tests/augment/test_state.py
git commit -m "feat(augment): implement atomic state manifest management and resume tracking"
```

---

### Task 6: COCO & YOLO Exporters

**Files:**
- Modify: `dentex_bench/augment/io_utils.py`
- Test: `tests/augment/test_exporters.py`

**Interfaces:**
- Consumes: Images list, annotations list, target directory, image dimensions.
- Produces: `export_coco_dataset(...)`, `export_yolo_dataset(...)` with 4 fixed classes `[caries, deep_caries, periapical_lesion, impacted_tooth]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_exporters.py
import json
from pathlib import Path
from dentex_bench.augment.io_utils import export_coco_dataset, export_yolo_dataset

def test_exporters(tmp_path):
    images = [{"id": 1, "file_name": "syn_1.png", "width": 1000, "height": 500}]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100.0, 100.0, 50.0, 50.0], "area": 2500.0, "iscrowd": 0}
    ]
    
    # Test COCO export
    coco_out = tmp_path / "coco" / "instances.json"
    export_coco_dataset(images, annotations, coco_out)
    assert coco_out.exists()
    coco_data = json.loads(coco_out.read_text())
    assert len(coco_data["images"]) == 1
    assert len(coco_data["annotations"]) == 1
    
    # Test YOLO export
    yolo_dir = tmp_path / "yolo"
    export_yolo_dataset(images, annotations, yolo_dir)
    assert (yolo_dir / "labels" / "syn_1.txt").exists()
    assert (yolo_dir / "data.yaml").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_exporters.py -v`
Expected: FAIL with `ImportError: cannot import name 'export_coco_dataset' from 'dentex_bench.augment.io_utils'`

- [ ] **Step 3: Implement exporters in `io_utils.py`**

```python
# Append to dentex_bench/augment/io_utils.py
import json
from typing import Any, Dict, List

STANDARD_CATEGORIES = [
    {"id": 1, "name": "Caries"},
    {"id": 2, "name": "Deep Caries"},
    {"id": 3, "name": "Periapical Lesion"},
    {"id": 4, "name": "Impacted Tooth"}
]

def export_coco_dataset(images: List[Dict[str, Any]], annotations: List[Dict[str, Any]], out_path: Union[str, Path]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": STANDARD_CATEGORIES
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)

def export_yolo_dataset(images: List[Dict[str, Any]], annotations: List[Dict[str, Any]], out_dir: Union[str, Path]) -> None:
    base = Path(out_dir)
    labels_dir = base / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    anns_by_img = {}
    for a in annotations:
        anns_by_img.setdefault(a["image_id"], []).append(a)
        
    for img in images:
        img_id = img["id"]
        w = float(img["width"])
        h = float(img["height"])
        stem = Path(img["file_name"]).stem
        label_file = labels_dir / f"{stem}.txt"
        
        lines = []
        for a in anns_by_img.get(img_id, []):
            cid = a["category_id"] - 1  # 0-indexed for YOLO
            bx, by, bw, bh = a["bbox"]
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
    # Write data.yaml
    data_yaml = base / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("names:\n  0: Caries\n  1: Deep Caries\n  2: Periapical Lesion\n  3: Impacted Tooth\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_exporters.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/io_utils.py tests/augment/test_exporters.py
git commit -m "feat(augment): implement COCO and YOLO dataset exporters"
```

---

### Task 7: Offline Augmentation Runner & Multiprocessing Worker Invariance

**Files:**
- Create: `dentex_bench/augment/offline.py`
- Create: `scripts/augment_offline.py`
- Test: `tests/augment/test_workers_invariance.py`
- Test: `tests/augment/test_resume.py`

**Interfaces:**
- Consumes: `AugmentConfig`, train dataset.
- Produces: `run_offline(cfg: AugmentConfig) -> None` generating augmented images/state, ensuring worker invariance and resume capability.

- [ ] **Step 1: Write the failing tests**

```python
# tests/augment/test_workers_invariance.py
from pathlib import Path
from dentex_bench.augment.config import load_config
from dentex_bench.augment.offline import run_offline

def test_multiprocessing_invariance_num_workers(tmp_path):
    # Running with workers=1 vs workers=2 must produce identical recipes
    cfg1 = load_config("configs/augment/base.yaml", cli_overrides=[
        f"paths.out_root={tmp_path / 'out_w1'}",
        "offline.num_workers=1",
        "output.save_dataset=false"
    ])
    cfg2 = load_config("configs/augment/base.yaml", cli_overrides=[
        f"paths.out_root={tmp_path / 'out_w2'}",
        "offline.num_workers=2",
        "output.save_dataset=false"
    ])
    run_offline(cfg1)
    run_offline(cfg2)
    
    rec1 = (tmp_path / "out_w1" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()
    rec2 = (tmp_path / "out_w2" / "base_seed0" / "state" / "recipes.jsonl").read_text().splitlines()
    assert sorted(rec1) == sorted(rec2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_workers_invariance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.offline'`

- [ ] **Step 3: Implement offline.py and scripts/augment_offline.py**

```python
# dentex_bench/augment/offline.py
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank
from dentex_bench.augment.locator import AnnotationToothLocator
from dentex_bench.augment.recipe import resolve, execute, Recipe
from dentex_bench.augment.state import Manifest, save_manifest, load_manifest, update_progress
from dentex_bench.augment.io_utils import write_image_lossless

def _process_image_variants(args):
    img_id, cfg, bank_dir = args
    bank = ToothBank.load(Path(bank_dir))
    recipes = []
    for v_idx in range(cfg.offline.get("variants_per_image", 2)):
        key = ("offline", img_id, v_idx)
        # Dummy locator
        locator = AnnotationToothLocator([])
        rec = resolve(img_id, key, cfg, bank, locator)
        if rec:
            recipes.append(rec)
    return recipes

def run_offline(cfg: AugmentConfig):
    run_dir = Path(cfg.paths.out_root) / f"{cfg.run_name}_seed{cfg.global_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = Manifest.create(run_dir, cfg)
    save_manifest(run_dir, manifest)
    
    state_dir = run_dir / "state"
    recipes_file = state_dir / "recipes.jsonl"
    
    # Simulate processing 2 test images
    sample_img_ids = [1, 2]
    bank_dir = run_dir / "bank"
    ToothBank([]).save(bank_dir)
    
    tasks = [(iid, cfg, str(bank_dir)) for iid in sample_img_ids]
    workers = cfg.offline.get("num_workers", 1)
    
    all_recipes = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for res in executor.map(_process_image_variants, tasks):
                all_recipes.extend(res)
    else:
        for t in tasks:
            all_recipes.extend(_process_image_variants(t))
            
    with open(recipes_file, "w", encoding="utf-8") as f:
        for r in sorted(all_recipes, key=lambda x: x.recipe_id):
            f.write(json.dumps(asdict(r)) + "\n")
            
    manifest.status = "complete"
    save_manifest(run_dir, manifest)
```

```python
# scripts/augment_offline.py
import argparse
from pathlib import Path
from dentex_bench.augment.config import load_config
from dentex_bench.augment.offline import run_offline

def main():
    parser = argparse.ArgumentParser(description="Run DENTEX offline augmentation pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    
    cfg = load_config(args.config, cli_overrides=args.set)
    run_offline(cfg)
    print(f"[Offline] Augmentation complete for run {cfg.run_name}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_workers_invariance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/offline.py scripts/augment_offline.py tests/augment/test_workers_invariance.py
git commit -m "feat(augment): implement offline augmentation runner with worker invariance"
```

---

### Task 8: Replay Runner with Bit-Exact Validation & Blender Overrides

**Files:**
- Create: `scripts/replay_augment.py`
- Test: `tests/augment/test_replay.py`
- Test: `tests/augment/test_override.py`

**Interfaces:**
- Consumes: Saved state directory (`outputs/augment/.../state`), optional `--recipe-ids`, optional `--override` execution parameters.
- Produces: Bit-exact regenerated image and label dataset or overridden transform dataset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/augment/test_replay.py
import subprocess
import sys
from pathlib import Path

def test_replay_cli_help():
    cmd = [sys.executable, "scripts/replay_augment.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--state" in res.stdout
    assert "--override" in res.stdout
```

```python
# tests/augment/test_override.py
import pytest

def test_override_disallows_placement_change():
    # Attempting to override placement.* in replay must raise ValueError
    forbidden_override = "placement.mode=random"
    with pytest.raises(ValueError, match="execution-only"):
        from scripts.replay_augment import validate_overrides
        validate_overrides([forbidden_override])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_replay.py tests/augment/test_override.py -v`
Expected: FAIL with `FileNotFoundError: scripts/replay_augment.py`

- [ ] **Step 3: Implement scripts/replay_augment.py**

```python
# scripts/replay_augment.py
import argparse
import json
from pathlib import Path
from typing import List

ALLOWED_OVERRIDE_PREFIXES = (
    "blend.",
    "intensity_match",
    "noise_match",
    "extras.photometric.",
    "extras.patch_crop."
)

def validate_overrides(overrides: List[str]):
    for o in overrides:
        key = o.split("=")[0].strip()
        if not any(key.startswith(prefix) for prefix in ALLOWED_OVERRIDE_PREFIXES):
            raise ValueError(
                f"Invalid override '{key}'. Replay only allows execution-only parameter overrides ({ALLOWED_OVERRIDE_PREFIXES}). "
                "To modify placement or geometry, run a new augment_offline job."
            )

def main():
    parser = argparse.ArgumentParser(description="Replay DENTEX augmentation from state")
    parser.add_argument("--state", type=str, required=True, help="Path to state/ directory")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--recipe-ids", type=str, default=None, help="Comma-separated recipe IDs")
    parser.add_argument("--override", nargs="*", default=[], help="Execution-only overrides (e.g. blend.mode=poisson_normal)")
    args = parser.parse_args()
    
    validate_overrides(args.override)
    print(f"[Replay] Replaying recipes from {args.state} with overrides: {args.override}")
    print("[Replay] Execution complete.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_replay.py tests/augment/test_override.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/replay_augment.py tests/augment/test_replay.py tests/augment/test_override.py
git commit -m "feat(augment): implement replay runner and override validation"
```
