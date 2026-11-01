import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from dentex_bench.env import detect_platform, get_device_info

@dataclass
class BenchmarkConfig:
    platform: str = "local"
    device: str = "cpu"
    vram_gb: float = 0.0
    is_low_vram: bool = False
    
    # Dataset & classes
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: [
        "Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"
    ])
    seed: int = 42
    train_val_split_ratio: float = 0.8
    
    # Training hyperparameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    amp: bool = True
    img_size: int = 800
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    models: List[str] = field(default_factory=lambda: ["hierarchical_net", "diffusion_net"])
    
    # Paths
    root_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    @classmethod
    def from_auto_env(cls, root_dir: str = "."):
        platform = detect_platform()
        dev_info = get_device_info()
        is_low = dev_info["is_low_vram"]
        
        # Adaptive tuning: batch_size=1 with gradient accumulation 16 is universally memory-safe
        batch_size = 1
        grad_accum = 16
        img_size = 800
        num_workers = 0 if os.name == "nt" else 2
            
        base = Path(root_dir)
        output_dir = Path("/kaggle/working/outputs") if platform == "kaggle" else base / "outputs"
        data_dir = base / "data"
        
        return cls(
            platform=platform,
            device=dev_info["device"],
            vram_gb=dev_info["vram_gb"],
            is_low_vram=is_low,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            img_size=img_size,
            num_workers=num_workers,
            root_dir=base,
            data_dir=data_dir,
            output_dir=output_dir
        )
