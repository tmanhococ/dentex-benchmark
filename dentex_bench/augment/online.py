"""Online Dynamic Augmentation Dataset wrapper for DENTEX benchmark.

Wraps a base dataset (e.g. DentexDataset or generic Dataset) and performs
deterministic, dynamic tooth-swap augmentation on-the-fly during training.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank
from dentex_bench.augment.locator import ToothLocator, AnnotationToothLocator, LocatedTooth
from dentex_bench.augment.recipe import resolve, execute, Recipe
from dentex_bench.augment.rng import rng_for
from dentex_bench.data.dataset import load_annotated_image, read_rgb


class SwapAugmentDataset(Dataset):
    """Online dynamic tooth-swap augmentation wrapper for DentexDataset.

    Enforces:
    - INV-2: Augmentation is deterministic based on global_seed, epoch, and image_id.
    - INV-6: Label annotations are kept strictly synchronized upon tooth swap.
    - INV-8: Recipe resolution encapsulates all stochastic choices; execution is pure.
    """

    def __init__(
        self,
        base: Dataset,
        cfg: AugmentConfig,
        bank: ToothBank,
        locator: Optional[ToothLocator] = None,
    ):
        self.base = base
        self.cfg = cfg
        self.bank = bank

        if locator is not None:
            self.locator = locator
        elif hasattr(base, "coco") and isinstance(base.coco, dict) and "annotations" in base.coco:
            self.locator = AnnotationToothLocator(base.coco["annotations"])
        elif hasattr(base, "img_to_anns") and isinstance(base.img_to_anns, dict):
            all_anns = [a for anns in base.img_to_anns.values() for a in anns]
            self.locator = AnnotationToothLocator(all_anns)
        else:
            self.locator = AnnotationToothLocator([])

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current training epoch to update deterministic RNG streams."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def _get_image_id(self, idx: int) -> int:
        if hasattr(self.base, "images") and isinstance(self.base.images, list) and idx < len(self.base.images):
            img_info = self.base.images[idx]
            if isinstance(img_info, dict) and "id" in img_info:
                return int(img_info["id"])
        return idx

    def _load_base_image_and_anns(
        self, idx: int, image_id: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], bool, Any]:
        """Loads or extracts base image and annotations for swap execution.

        Returns:
            Tuple of (image_np, anns_list, is_dict_item, original_item)
        """
        # Try loading directly from DentexDataset storage if available
        if hasattr(self.base, "img_dir") and hasattr(self.base, "images") and idx < len(self.base.images):
            img_info = self.base.images[idx]
            fn = img_info.get("file_name", f"{image_id}.png")
            width = img_info.get("width")
            height = img_info.get("height")
            img_dir = self.base.img_dir
            idx_map = getattr(self.base, "_image_index", None)
            try:
                base_img, _ = load_annotated_image(fn, width, height, img_dir, idx_map)
                anns = [dict(a) for a in getattr(self.base, "img_to_anns", {}).get(image_id, [])]
                return base_img, anns, False, None
            except Exception:
                pass

        # Fallback to evaluating base[idx]
        item = self.base[idx]
        if isinstance(item, dict):
            is_dict_item = True
            raw_img = item.get("image")
            raw_anns = item.get("target", item.get("annotations", []))
            if isinstance(raw_anns, dict):
                anns_list = [raw_anns]
            elif isinstance(raw_anns, list):
                anns_list = [dict(a) if isinstance(a, dict) else a for a in raw_anns]
            else:
                anns_list = []
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            is_dict_item = False
            raw_img = item[0]
            target_data = item[1]
            anns_list = []
            if isinstance(target_data, dict):
                boxes = target_data.get("boxes", [])
                labels = target_data.get("labels", [])
                for bi in range(len(boxes)):
                    b = boxes[bi].tolist() if hasattr(boxes[bi], "tolist") else list(boxes[bi])
                    x1, y1, x2, y2 = b[:4]
                    lbl = int(labels[bi].item() if hasattr(labels[bi], "item") else labels[bi])
                    anns_list.append({
                        "id": bi,
                        "image_id": image_id,
                        "category_id": lbl,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                    })
        else:
            is_dict_item = False
            raw_img = None
            anns_list = []

        # Convert raw_img to numpy uint8 RGB [H, W, 3]
        if isinstance(raw_img, torch.Tensor):
            t = raw_img.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=-1)
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            base_img = arr
        elif isinstance(raw_img, np.ndarray):
            base_img = raw_img.copy()
            if base_img.ndim == 2:
                base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
            elif base_img.shape[-1] == 1:
                base_img = np.repeat(base_img, 3, axis=-1)
        else:
            base_img = np.zeros((400, 800, 3), dtype=np.uint8)

        return base_img, anns_list, is_dict_item, item

    def __getitem__(self, idx: int) -> Any:
        image_id = self._get_image_id(idx)

        # Check online probability p
        online_cfg = getattr(self.cfg, "online", {})
        p = online_cfg.get("p", 0.5) if isinstance(online_cfg, dict) else getattr(online_cfg, "p", 0.5)

        rng = rng_for(self.cfg.global_seed, "online_p", self._epoch, image_id)
        if p <= 0.0:
            should_augment = False
        elif p >= 1.0:
            should_augment = True
        else:
            should_augment = float(rng.uniform(0.0, 1.0)) < p

        if not should_augment:
            return self.base[idx]

        key = ("online", self._epoch, image_id, 0)
        target_file = ""
        if hasattr(self.base, "images") and idx < len(self.base.images):
            target_file = self.base.images[idx].get("file_name", f"{image_id}.png")

        rec = resolve(image_id, key, self.cfg, self.bank, self.locator, target_file=target_file)
        if rec is None:
            return self.base[idx]

        base_img, base_anns, is_dict_item, orig_item = self._load_base_image_and_anns(idx, image_id)

        # Ensure FDI tags exist on annotations for tooth replacement filtering
        for ann in base_anns:
            if not isinstance(ann, dict):
                continue
            if ann.get("fdi") is None:
                if "category_id_1" in ann and "category_id_2" in ann:
                    q = ann["category_id_1"]
                    t = ann["category_id_2"]
                    if q in range(1, 5) and t in range(1, 9):
                        ann["fdi"] = q * 10 + t
                    elif q in range(0, 4) and t in range(0, 8):
                        ann["fdi"] = (q + 1) * 10 + (t + 1)
                else:
                    candidate_teeth = self.locator.teeth(image_id)
                    if candidate_teeth:
                        ann["fdi"] = candidate_teeth[0].fdi

        loader = lambda iid: (base_img.copy(), [dict(a) for a in base_anns])
        syn_img, syn_anns, _ = execute(rec, loader, self.bank)

        # Handle DentexDataset-style scaling if enabled
        scale = 1.0
        if getattr(self.base, "resize", False) and getattr(self.base, "target_short", None) is not None:
            cur_h, cur_w = syn_img.shape[:2]
            max_l = getattr(self.base, "max_long", 1333) or 1333
            target_s = self.base.target_short
            scale = min(float(target_s) / max(1, min(cur_h, cur_w)),
                        float(max_l) / max(1, max(cur_h, cur_w)))
            if abs(scale - 1.0) > 1e-4:
                new_w = max(1, int(round(cur_w * scale)))
                new_h = max(1, int(round(cur_h * scale)))
                syn_img = cv2.resize(syn_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []

        for ann in syn_anns:
            bx, by, bw, bh = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
            boxes_xyxy.append([float(bx * scale), float(by * scale), float((bx + bw) * scale), float((by + bh) * scale)])
            labels.append(int(ann.get("category_id", 1)))
            areas.append(float(ann.get("area", bw * bh) * (scale ** 2)))

        # Handle optional base transforms
        if getattr(self.base, "transforms", None) is not None:
            try:
                augmented = self.base.transforms(image=syn_img, bboxes=boxes_xyxy, category_ids=labels)
                syn_img = augmented.get("image", syn_img)
                boxes_xyxy = augmented.get("bboxes", boxes_xyxy)
                labels = augmented.get("category_ids", labels)
            except Exception:
                try:
                    syn_img = self.base.transforms(syn_img)
                except Exception:
                    pass

        # Return format matching base dataset
        if is_dict_item and orig_item is not None:
            out_dict = dict(orig_item)
            out_dict["image"] = syn_img
            if "target" in out_dict:
                out_dict["target"] = syn_anns
            if "annotations" in out_dict:
                out_dict["annotations"] = syn_anns
            return out_dict

        # Format tensor [3, H, W] in [0, 1]
        if isinstance(syn_img, np.ndarray):
            if syn_img.ndim == 2:
                syn_img = np.expand_dims(syn_img, axis=-1)
            if syn_img.shape[-1] == 1:
                syn_img = np.repeat(syn_img, 3, axis=-1)
            img_tensor = torch.from_numpy(syn_img).permute(2, 0, 1).float()
            if syn_img.dtype == np.uint8:
                img_tensor = img_tensor / 255.0
        elif isinstance(syn_img, torch.Tensor):
            img_tensor = syn_img.float()
            if img_tensor.ndim == 3 and img_tensor.shape[0] != 3 and img_tensor.shape[2] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            if syn_img.dtype == torch.uint8:
                img_tensor = img_tensor / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(syn_img)}")

        target_dict = {
            "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
        }
        return img_tensor, target_dict
