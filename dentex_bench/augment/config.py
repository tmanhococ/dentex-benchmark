"""Augmentation configuration dataclass and YAML schema validator.

Defines the hierarchical configuration structure for the Dentex tooth-swap
and augmentation pipeline, enforces invariant constraints (INV-3, INV-7),
and provides deterministic SHA256 configuration hashing.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class PathsConfig:
    data_root: str = "data"
    train_ann: str = "data/processed/instances_train.json"
    val_ann: str = "data/processed/instances_val.json"
    img_dir: str = "data/raw"
    out_root: str = "outputs/augment"


@dataclass
class ROIConfig:
    caries_dilate_px: int = 4
    periapical_bbox_expand: float = 0.15
    impacted_dilate_px: int = 9
    healthy_dilate_px: int = 4


@dataclass
class HealthyFilterConfig:
    predictions_file: Optional[str] = None
    max_disease_score: float = 0.1


@dataclass
class BankConfig:
    mask_source: str = "polygon"
    active_contour: bool = False
    crop_margin_px: int = 24
    roi: ROIConfig = field(default_factory=ROIConfig)
    include_healthy: bool = False
    healthy_filter: HealthyFilterConfig = field(default_factory=HealthyFilterConfig)
    evaluate_mask_methods: bool = False


@dataclass
class LocatorConfig:
    providers: List[str] = field(default_factory=lambda: ["annotations"])
    pseudo_label_file: Optional[str] = None
    min_score: float = 0.5


@dataclass
class PlacementConfig:
    mode: str = "fdi_swap"
    swap_types: List[str] = field(default_factory=lambda: ["disease_to_disease"])
    allow_mirror: bool = True
    min_ops: int = 1
    max_ops: int = 3
    max_tries: int = 20
    class_weight_power: float = 0.5
    class_weights: Optional[Dict[str, float]] = None
    max_neighbor_iou: float = 0.1
    max_arch_dev_ratio: float = 0.5


@dataclass
class GeometryConfig:
    isotropic: bool = False
    scale_min: float = 0.85
    scale_max: float = 1.15
    max_rotate_deg: float = 10.0
    jitter_scale: float = 0.05
    jitter_rotate_deg: float = 3.0


@dataclass
class BlendConfig:
    inpaint_target: bool = True
    inpaint_radius: int = 5
    intensity_match: str = "meanstd"
    mode: str = "core_preserving"
    choices: List[str] = field(
        default_factory=lambda: ["hard", "feather", "poisson_normal", "laplacian", "core_preserving"]
    )
    feather_sigma: float = 5.0
    pyramid_levels: int = 5
    core_erode_px: int = 6
    ring_mode: str = "feather"
    noise_match: bool = True
    log_domain: bool = False


@dataclass
class OutputConfig:
    save_dataset: bool = True
    formats: List[str] = field(default_factory=lambda: ["coco", "yolo"])
    include_originals: bool = True
    image_ext: str = "png"
    overwrite: bool = False


@dataclass
class AugmentConfig:
    version: int = 1
    global_seed: int = 0
    run_name: str = "base"
    paths: PathsConfig = field(default_factory=PathsConfig)
    mode: str = "offline"
    bank: BankConfig = field(default_factory=BankConfig)
    locator: LocatorConfig = field(default_factory=LocatorConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    instance_aug: Dict[str, Any] = field(default_factory=lambda: {"gamma": [0.85, 1.15]})
    extras: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    offline: Dict[str, Any] = field(default_factory=lambda: {"variants_per_image": 2, "num_workers": 1})
    online: Dict[str, Any] = field(default_factory=lambda: {"p": 0.5, "log_recipes": False})
    qc: Dict[str, Any] = field(default_factory=lambda: {"max_seam_score": 2.5, "save_debug_every_n": 50})
    base_transforms: Dict[str, Any] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)


def _dict_to_dataclass(cls: Any, data: dict) -> Any:
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs = {}
    for f_name, f_def in cls.__dataclass_fields__.items():
        if f_name in data:
            val = data[f_name]
            f_type = f_def.type
            if is_dataclass(f_type) and isinstance(val, dict):
                kwargs[f_name] = _dict_to_dataclass(f_type, val)
            else:
                kwargs[f_name] = val
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _apply_override(d: dict, key_path: str, val: Any) -> None:
    keys = key_path.split(".")
    curr = d
    for k in keys[:-1]:
        if k not in curr or not isinstance(curr[k], dict):
            curr[k] = {}
        curr = curr[k]
    leaf = keys[-1]
    if isinstance(val, str):
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.lower() in ("null", "none"):
            val = None
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
    curr[leaf] = val


def validate_config(cfg: AugmentConfig) -> None:
    # INV-3: Vertical flip is strictly prohibited in dental panoramas
    if "vflip" in cfg.base_transforms:
        val = cfg.base_transforms["vflip"]
        if val is not False:
            raise ValueError("vflip is strictly prohibited (INV-3)")
    rotate_cfg = cfg.base_transforms.get("rotate", {})
    if isinstance(rotate_cfg, dict) and rotate_cfg.get("limit_deg", 0) > 5.0:
        raise ValueError(
            f"Full-image rotate limit_deg must be <= 5 degrees (INV-3), got {rotate_cfg.get('limit_deg')}"
        )
    # INV-7: Only lossless PNG output format is allowed
    if cfg.output.image_ext.lower() not in ("png",):
        raise ValueError(f"lossless PNG output format required (INV-7), got '{cfg.output.image_ext}'")


def load_config(path: Union[str, Path], cli_overrides: Optional[List[str]] = None) -> AugmentConfig:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Handle _base_ inheritance
    if "_base_" in data:
        base_path = p.parent / data["_base_"]
        with open(base_path, "r", encoding="utf-8") as bf:
            base_data = yaml.safe_load(bf) or {}
        data.pop("_base_", None)
        data = _deep_merge(base_data, data)

    if cli_overrides:
        for override in cli_overrides:
            k, v = override.split("=", 1)
            _apply_override(data, k.strip(), v.strip())

    cfg = _dict_to_dataclass(AugmentConfig, data)
    validate_config(cfg)
    return cfg


def config_hash(cfg: AugmentConfig) -> str:
    """Compute normalized SHA256 excluding runtime/output dirs."""
    d = asdict(cfg)
    d.pop("output", None)
    d.pop("paths", None)
    d.pop("offline", None)
    d.pop("online", None)
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
