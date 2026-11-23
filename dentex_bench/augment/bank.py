# dentex_bench/augment/bank.py
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np

from dentex_bench.augment.fdi import is_upper, parse_fdi
from dentex_bench.augment.io_utils import (
    sha256_file,
    write_image_lossless,
    read_image_lossless,
)


@dataclass(frozen=True)
class ToothEntry:
    entry_id: str
    image_id: int
    file_name: str
    fdi: int
    is_upper: bool
    labels: List[str]
    is_healthy: bool
    roi_class: str
    bbox_xywh: List[float]
    roi_xywh: List[int]
    annotations: List[Dict[str, Any]]
    axis_angle_deg: float
    mask_area_px: int
    mask_h: float
    mask_w: float
    ring_stats: Dict[str, float]
    noise_sigma: float
    patch_path: str
    mask_path: str
    roi_mask_path: str
    mask_source: str
    healthy_score: Optional[float] = None


class ToothBank:
    def __init__(self, entries: List[ToothEntry]):
        self.entries = list(entries)
        self._by_id = {e.entry_id: e for e in self.entries}
        self._by_fdi: Dict[int, List[ToothEntry]] = {}
        self._by_class: Dict[str, List[ToothEntry]] = {}
        for e in self.entries:
            self._by_fdi.setdefault(e.fdi, []).append(e)
            self._by_class.setdefault(e.roi_class, []).append(e)

    @staticmethod
    def assert_no_val_leakage(train_ids: Set[int], val_ids: Set[int]) -> None:
        """INV-1: Verify that bank contains zero validation images."""
        leaked = set(train_ids).intersection(set(val_ids))
        if leaked:
            raise AssertionError(
                f"INV-1 VIOLATION: Validation image IDs leaked into bank: {leaked}"
            )

    def get(self, entry_id: str) -> Optional[ToothEntry]:
        return self._by_id.get(entry_id)

    def by_fdi(self, fdi: int) -> List[ToothEntry]:
        return self._by_fdi.get(fdi, [])

    def by_class(self, roi_class: str) -> List[ToothEntry]:
        return self._by_class.get(roi_class, [])

    @classmethod
    def is_cached(cls, bank_dir: Union[str, Path]) -> bool:
        bank_dir = Path(bank_dir)
        return (bank_dir / "bank_index.jsonl").is_file() and (
            bank_dir / "bank_meta.json"
        ).is_file()

    def save(self, out_dir: Union[str, Path]) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        index_file = out_dir / "bank_index.jsonl"
        with open(index_file, "w", encoding="utf-8") as f:
            for e in sorted(self.entries, key=lambda x: x.entry_id):
                f.write(json.dumps(asdict(e)) + "\n")

        bank_hash = sha256_file(index_file)
        meta = {
            "bank_hash": bank_hash,
            "total_entries": len(self.entries),
            "classes": {k: len(v) for k, v in self._by_class.items()},
        }
        with open(out_dir / "bank_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, bank_dir: Union[str, Path]) -> "ToothBank":
        bank_dir = Path(bank_dir)
        index_file = bank_dir / "bank_index.jsonl"
        entries = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(ToothEntry(**data))
        return cls(entries)

    @classmethod
    def build(cls, cfg: Optional[Any] = None) -> "ToothBank":
        """Build ToothBank instance."""
        return cls([])
