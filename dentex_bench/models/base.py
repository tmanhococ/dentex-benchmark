from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from dentex_bench.config import BenchmarkConfig


class BaseDetectorRunner(ABC):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        
        # Adaptive AMP GradScaler
        scaler_enabled = bool(self.config.amp and self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_epoch(self, dataloader, epoch: int) -> dict:
        pass

    @abstractmethod
    def predict(self, images: list) -> list:
        pass

    def save_checkpoint(self, path: Path, extra_info: Optional[Dict[str, Any]] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "config": self.config.__dict__,
            "extra_info": extra_info or {}
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"[dentex_bench] Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> dict:
        path = Path(path)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        if self.model is not None and "model_state_dict" in checkpoint and checkpoint["model_state_dict"] is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[dentex_bench] Loaded checkpoint from {path}")
        return checkpoint.get("extra_info", {})
