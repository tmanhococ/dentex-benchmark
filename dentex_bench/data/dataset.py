import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, Union
from torch.utils.data import Dataset

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def build_image_index(root: Union[str, Path]) -> Dict[str, list]:
    """Maps lower-cased file names to every matching image path below root."""
    index: Dict[str, list] = {}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file():
            index.setdefault(p.name.lower(), []).append(p)
    return index


def _candidate_rank(path: Path) -> tuple:
    # DENTEX subsets reuse names (train_0.png ...); the disease subset owns the 4-class labels
    s = path.as_posix().lower()
    return (0 if "disease" in s else 1, s)


def read_rgb(path: Union[str, Path]) -> Optional[np.ndarray]:
    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if loaded is None:
        return None
    if loaded.ndim == 2:
        return cv2.cvtColor(loaded, cv2.COLOR_GRAY2RGB)
    if loaded.shape[2] == 4:
        return cv2.cvtColor(loaded, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)


def resolve_image_path(
    file_name: str,
    width: Optional[int],
    height: Optional[int],
    img_dir: Union[str, Path],
    index: Optional[Dict[str, list]] = None
) -> Path:
    """
    Finds the radiograph an annotation refers to (reading only image headers).

    Raises FileNotFoundError when no file has that name, and ValueError when files with that name
    exist but none has the annotated width x height (i.e. it would be a different radiograph).
    """
    from PIL import Image

    img_dir = Path(img_dir)
    direct = img_dir / file_name
    if direct.is_file():
        candidates = [direct]
    else:
        if index is None:
            index = build_image_index(img_dir)
        candidates = sorted(index.get(Path(file_name).name.lower(), []), key=_candidate_rank)
    if not candidates:
        raise FileNotFoundError(f"Image '{file_name}' not found under {img_dir}")

    rejected = []
    for path in candidates:
        try:
            with Image.open(path) as im:
                w, h = im.size
        except OSError:
            rejected.append(f"{path} (unreadable)")
            continue
        if not width or not height or (w == int(width) and h == int(height)):
            return path
        rejected.append(f"{path} ({w}x{h})")
    raise ValueError(
        f"No file named '{file_name}' matches the annotated size {width}x{height}: {rejected}"
    )


def load_annotated_image(
    file_name: str,
    width: Optional[int],
    height: Optional[int],
    img_dir: Union[str, Path],
    index: Optional[Dict[str, list]] = None
) -> Tuple[np.ndarray, Path]:
    """Loads the radiograph an annotation refers to as RGB uint8 (see resolve_image_path for errors)."""
    path = resolve_image_path(file_name, width, height, img_dir, index)
    image = read_rgb(path)
    if image is None:
        raise ValueError(f"Image '{path}' could not be decoded")
    return image, path


class DentexDataset(Dataset):
    """
    PyTorch Dataset for DENTEX 4-Class Abnormal Tooth Detection.
    Loads COCO formatted annotations and returns images and bounding box targets.
    """
    def __init__(
        self,
        img_dir: Union[str, Path],
        ann_file: Union[str, Path],
        transforms: Optional[Callable] = None,
        target_short: Optional[int] = 800,
        max_long: Optional[int] = 1333,
        resize: bool = True
    ):
        self.img_dir = Path(img_dir)
        ann_file = Path(ann_file)
        with open(ann_file, "r") as f:
            self.coco = json.load(f)
            
        self.images = self.coco.get("images", [])
        self.transforms = transforms
        self.target_short = target_short
        self.max_long = max_long
        self.resize = resize
        
        # Index annotations by image_id
        self.img_to_anns: Dict[int, list] = {}
        for ann in self.coco.get("annotations", []):
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)
        self._image_index: Optional[Dict[str, list]] = None

    def __len__(self) -> int:
        return len(self.images)

    def image_path(self, idx: int) -> Path:
        info = self.images[idx]
        return load_annotated_image(info["file_name"], info.get("width"), info.get("height"), self.img_dir, self._index())[1]

    def _index(self) -> Dict[str, list]:
        if self._image_index is None:
            self._image_index = build_image_index(self.img_dir)
        return self._image_index

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info.get("file_name", f"{img_id}.png")
        height = img_info.get("height")
        width = img_info.get("width")

        if (self.img_dir / file_name).is_file():
            image, _ = load_annotated_image(file_name, width, height, self.img_dir)
        else:
            image, _ = load_annotated_image(file_name, width, height, self.img_dir, self._index())

        scale = 1.0
        if self.resize and self.target_short is not None and image is not None:
            cur_h, cur_w = image.shape[:2]
            max_l = self.max_long if self.max_long is not None else 1333
            scale = min(float(self.target_short) / max(1, min(cur_h, cur_w)),
                        float(max_l) / max(1, max(cur_h, cur_w)))
            if abs(scale - 1.0) > 1e-4:
                new_w = max(1, int(round(cur_w * scale)))
                new_h = max(1, int(round(cur_h * scale)))
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([float(x * scale), float(y * scale), float((x + w) * scale), float((y + h) * scale)])  # xyxy format
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h) * (scale ** 2)))
            
        if self.transforms is not None:
            try:
                # Try Albumentations format
                augmented = self.transforms(image=image, bboxes=boxes, category_ids=labels)
                image = augmented.get("image", image)
                boxes = augmented.get("bboxes", boxes)
                labels = augmented.get("category_ids", labels)
            except Exception:
                # Fallback for standard torchvision/callables
                try:
                    image = self.transforms(image)
                except Exception:
                    pass
            
        # Convert image to float PyTorch tensor [3, H, W]
        if isinstance(image, np.ndarray):
            is_uint8 = (image.dtype == np.uint8)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            if is_uint8:
                img_tensor = img_tensor / 255.0
        elif isinstance(image, torch.Tensor):
            is_uint8 = (image.dtype == torch.uint8)
            img_tensor = image.float()
            if img_tensor.ndim == 3 and img_tensor.shape[0] != 3 and img_tensor.shape[2] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            if is_uint8:
                img_tensor = img_tensor / 255.0
        else:
            raise TypeError(f"Unsupported image type after transforms: {type(image)}")

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32)
        }
        return img_tensor, target
