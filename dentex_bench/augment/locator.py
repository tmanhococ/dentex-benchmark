import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable


__all__ = [
    "LocatedTooth",
    "ToothLocator",
    "AnnotationToothLocator",
    "PseudoLabelToothLocator",
]


@dataclass(frozen=True)
class LocatedTooth:
    """Represents a located tooth candidate for replacement/placement."""

    fdi: int
    bbox: List[float]
    segmentation: Optional[List[List[float]]] = None
    source: str = "annotations"  # "annotations" or "pseudo_label"
    score: float = 1.0


@runtime_checkable
class ToothLocator(Protocol):
    """Protocol for providers locating teeth on target images."""

    def teeth(self, image_id: int) -> List[LocatedTooth]:
        ...


class AnnotationToothLocator:
    """Provides teeth locations extracted directly from ground truth annotations."""

    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        is_zero_indexed: Optional[bool] = None,
    ):
        self._by_image: Dict[int, List[LocatedTooth]] = {}
        if is_zero_indexed is None:
            any_zero = any(
                (ann.get("category_id_1") == 0 or ann.get("category_id_2") == 0)
                for ann in annotations
                if isinstance(ann, dict)
            )
        else:
            any_zero = is_zero_indexed

        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            img_id = ann.get("image_id")
            # Determine FDI from quadrant (cat1) and tooth number (cat2) or direct field
            fdi = ann.get("fdi")
            if fdi is None and "category_id_1" in ann and "category_id_2" in ann:
                q = ann["category_id_1"]
                t = ann["category_id_2"]
                if any_zero:
                    if q in range(0, 4) and t in range(0, 8):
                        fdi = (q + 1) * 10 + (t + 1)
                else:
                    if q in range(1, 5) and t in range(1, 9):
                        fdi = q * 10 + t
                    elif q in range(0, 4) and t in range(0, 8):
                        fdi = (q + 1) * 10 + (t + 1)

            if fdi is not None and img_id is not None:
                tooth = LocatedTooth(
                    fdi=int(fdi),
                    bbox=ann.get("bbox", []),
                    segmentation=ann.get("segmentation"),
                    source="annotations",
                    score=float(ann.get("score", 1.0)),
                )
                self._by_image.setdefault(int(img_id), []).append(tooth)

    def teeth(self, image_id: int) -> List[LocatedTooth]:
        return self._by_image.get(int(image_id), [])


class PseudoLabelToothLocator:
    """Provides tooth positions inferred from tooth enumeration models or external pseudo-labels."""

    def __init__(
        self,
        pseudo_json_path: Union[str, Path],
        min_score: float = 0.5,
    ):
        self.pseudo_json_path = Path(pseudo_json_path)
        self.min_score = float(min_score)
        self._by_image: Dict[int, List[LocatedTooth]] = {}

        if not self.pseudo_json_path.is_file():
            raise FileNotFoundError(f"Pseudo-label file not found: {self.pseudo_json_path}")

        if self.pseudo_json_path.stat().st_size == 0:
            data = {}
        else:
            with open(self.pseudo_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        if isinstance(data, dict):
            annotations = data.get("annotations", [])
        elif isinstance(data, list):
            annotations = data
        else:
            annotations = []

        for ann in annotations:
            if not isinstance(ann, dict):
                continue

            score = float(ann.get("score", 1.0))
            if score < self.min_score:
                continue

            img_id = ann.get("image_id")
            if img_id is None:
                continue

            fdi = ann.get("fdi")
            if fdi is None and "category_id_1" in ann and "category_id_2" in ann:
                q = ann["category_id_1"]
                t = ann["category_id_2"]
                if q in range(1, 5) and t in range(1, 9):
                    fdi = q * 10 + t
                elif q in range(0, 4) and t in range(0, 8):
                    fdi = (q + 1) * 10 + (t + 1)
            elif fdi is None and "category_id" in ann:
                cat = ann["category_id"]
                if isinstance(cat, int) and (11 <= cat <= 48):
                    if cat // 10 in (1, 2, 3, 4) and cat % 10 in range(1, 9):
                        fdi = cat

            if fdi is not None:
                tooth = LocatedTooth(
                    fdi=int(fdi),
                    bbox=ann.get("bbox", []),
                    segmentation=ann.get("segmentation"),
                    source="pseudo_label",
                    score=score,
                )
                self._by_image.setdefault(int(img_id), []).append(tooth)

    def teeth(self, image_id: int) -> List[LocatedTooth]:
        return self._by_image.get(int(image_id), [])

