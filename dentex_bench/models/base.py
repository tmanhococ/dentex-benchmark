from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
from dentex_bench.config import BenchmarkConfig


class BaseDetectorRunner(ABC):
    # Identifies the network layout stored in checkpoints; None disables the check
    ARCH_ID: Optional[str] = None

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

    def _autocast(self):
        if self.device.type == "cuda" and self.scaler.is_enabled():
            return torch.amp.autocast("cuda", enabled=True)
        return nullcontext()

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        finite = len(grads) > 0 and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
        if finite:
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        # GradScaler skips the update itself when it found Inf/NaN gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _run_train_epoch(self, dataloader, compute_loss: Callable[[List[torch.Tensor], List[Dict[str, torch.Tensor]]], torch.Tensor]) -> dict:
        """
        Shared AMP + gradient-accumulation loop. compute_loss receives device-resident images
        (CHW float in [0, 1], original sizes) and targets (xyxy pixel boxes, labels 1..num_classes).
        """
        self.model.train()
        freeze_batchnorm(self.model)
        self.optimizer.zero_grad()
        accum_steps = max(1, self.config.gradient_accumulation_steps)
        total_loss, step = 0.0, 0

        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            targets = [
                {"boxes": t["boxes"].to(self.device).float().reshape(-1, 4), "labels": t["labels"].to(self.device).long()}
                for t in targets
            ]
            with self._autocast():
                loss = compute_loss(images, targets)
            self.scaler.scale(loss / accum_steps).backward()
            total_loss += float(loss.detach())
            step += 1
            if step % accum_steps == 0:
                self._optimizer_step()

        if step % accum_steps != 0:
            self._optimizer_step()
        return {"loss": total_loss / max(1, step)}

    def save_checkpoint(self, path: Path, extra_info: Optional[Dict[str, Any]] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "arch": self.ARCH_ID,
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "config": self.config.__dict__,
            "extra_info": extra_info or {}
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"[dentex_bench] Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: Path) -> dict:
        path = Path(path)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        if self.ARCH_ID is not None and checkpoint.get("arch") != self.ARCH_ID:
            raise ValueError(
                f"{path} was written by architecture '{checkpoint.get('arch')}', but this runner is "
                f"'{self.ARCH_ID}'. Delete the old checkpoint and retrain."
            )
        if self.model is not None and checkpoint.get("model_state_dict") is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[dentex_bench] Loaded checkpoint from {path}")
        return checkpoint.get("extra_info", {})


def checkpoint_matches(path: Path, runner_cls) -> bool:
    """True when path holds a checkpoint written by the current architecture of runner_cls."""
    path = Path(path)
    if not path.is_file():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("arch") == runner_cls.ARCH_ID


def freeze_batchnorm(model: nn.Module):
    """Keeps BatchNorm running statistics fixed: batch_size=1 statistics are too noisy to update them."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
