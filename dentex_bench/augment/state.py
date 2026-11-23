"""State Manifest, Atomic File Persistence & Resume Engine.

Manages experiment manifests, progress tracking, and atomic file persistence
for reproducible augmentation runs.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import cv2

from dentex_bench.augment.config import AugmentConfig, config_hash


def atomic_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Write data to destination path as JSON atomically using a temporary file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f"{p.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    tmp_path.replace(p)


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
    def create(cls, run_dir: Union[str, Path], cfg: AugmentConfig) -> "Manifest":
        p = Path(run_dir)
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            run_id=p.name,
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dict."""
        return asdict(self)


def save_manifest(run_dir: Union[str, Path], manifest: Manifest) -> None:
    """Save manifest to <run_dir>/state/manifest.json atomically."""
    p = Path(run_dir)
    target = p / "state" / "manifest.json"
    atomic_write_json(target, asdict(manifest), indent=2)


def load_manifest(run_dir: Union[str, Path]) -> Manifest:
    """Load manifest from <run_dir>/state/manifest.json."""
    p = Path(run_dir)
    target = p / "state" / "manifest.json"
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Manifest(**data)


def update_progress(
    run_dir: Union[str, Path],
    last_done_index: int,
    counts_delta: Optional[Dict[str, int]] = None,
    status: Optional[str] = None
) -> None:
    """Update progress tracking index and count deltas atomically."""
    manifest = load_manifest(run_dir)
    manifest.progress["last_done_index"] = last_done_index
    if counts_delta:
        for k, v in counts_delta.items():
            manifest.counts[k] = manifest.counts.get(k, 0) + v
    if status is not None:
        manifest.status = status
    save_manifest(run_dir, manifest)
