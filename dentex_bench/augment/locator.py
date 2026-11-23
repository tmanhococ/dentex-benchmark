from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


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
