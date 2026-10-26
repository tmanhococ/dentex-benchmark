import math
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from dentex_bench.models.base import BaseDetectorRunner


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DynamicDiffusionHead(nn.Module):
    """Diffusion-based iterative refinement head for bounding boxes and classes."""

    def __init__(self, num_classes: int = 4, hidden_dim: int = 256, num_proposals: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.box_encoder = nn.Linear(4, hidden_dim)
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)
        self.delta_box_head = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        features: torch.Tensor,
        noisy_boxes: torch.Tensor,
        timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features: (B, C, H, W)
        b, c, h, w = features.shape
        pooled_feat = torch.mean(features, dim=[2, 3])  # (B, C)
        time_embed = self.time_mlp(timesteps)           # (B, C)
        cond = (pooled_feat + time_embed).unsqueeze(1)  # (B, 1, C)

        box_emb = self.box_encoder(noisy_boxes)          # (B, N, C)
        hidden = self.refine(box_emb + cond)            # (B, N, C)

        logits = self.cls_head(hidden)                   # (B, N, num_classes + 1)
        deltas = self.delta_box_head(hidden)             # (B, N, 4)
        refined_boxes = torch.sigmoid(noisy_boxes + deltas)
        return logits, refined_boxes


class HierarchicalDetRunner(BaseDetectorRunner):
    """HierarchicalDet / DiffusionDet runner adapted for 4 pathology classes."""

    def build_model(self, pretrained: bool = True):
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.head = DynamicDiffusionHead(
            num_classes=self.config.num_classes,
            hidden_dim=256,
            num_proposals=100
        )

        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "proj": self.proj,
            "head": self.head
        }).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model

    def _prepare_images(self, images: Union[torch.Tensor, List[torch.Tensor], tuple]) -> torch.Tensor:
        """
        Prepares a batch of images into a single stacked Tensor on self.device,
        handling tensors, lists of equal-sized tensors, or lists of varying-sized tensors.
        """
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                return images.unsqueeze(0).to(self.device)
            return images.to(self.device)
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                return torch.empty((0, 3, 0, 0), device=self.device)
            shapes = [img.shape for img in images]
            if all(s == shapes[0] for s in shapes):
                return torch.stack([img.to(self.device) for img in images])
            else:
                max_h = max(img.shape[-2] for img in images)
                max_w = max(img.shape[-1] for img in images)
                padded = []
                for img in images:
                    pad_h = max_h - img.shape[-2]
                    pad_w = max_w - img.shape[-1]
                    if pad_h > 0 or pad_w > 0:
                        p = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
                    else:
                        p = img
                    padded.append(p.to(self.device))
                return torch.stack(padded)
        raise TypeError(f"Unsupported image type: {type(images)}")

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        accum_steps = max(1, self.config.gradient_accumulation_steps)
        step = 0
        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

        criterion_cls = nn.CrossEntropyLoss()
        criterion_box = nn.L1Loss()

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                images, targets = batch
            elif isinstance(batch, dict):
                images = batch["images"]
                targets = batch.get("targets", [])
            else:
                images, targets = batch, []

            images_tensor = self._prepare_images(images)
            b = images_tensor.size(0)

            # Sample random diffusion timesteps t in [0, 1000]
            timesteps = torch.randint(0, 1000, (b,), device=self.device).float()
            # Random initial noisy boxes in [0, 1]
            noisy_boxes = torch.rand((b, self.head.num_proposals, 4), device=self.device)

            # Loss computation with background penalty: default all to background (num_classes)
            target_cls = torch.full(
                (b, self.head.num_proposals),
                self.config.num_classes,
                dtype=torch.long,
                device=self.device
            )
            target_box = torch.zeros(
                (b, self.head.num_proposals, 4),
                dtype=torch.float32,
                device=self.device
            )

            if isinstance(targets, (list, tuple)):
                for i in range(min(b, len(targets))):
                    t = targets[i]
                    if isinstance(t, dict) and "boxes" in t and len(t["boxes"]) > 0:
                        gt_boxes = t["boxes"].to(self.device)
                        gt_labels = t.get(
                            "labels",
                            torch.ones(len(gt_boxes), dtype=torch.long, device=self.device)
                        ).to(self.device)

                        # Determine image dimensions for normalization
                        if isinstance(images, (list, tuple)) and i < len(images):
                            img_h = images[i].shape[-2]
                            img_w = images[i].shape[-1]
                        else:
                            img_h = images_tensor.shape[2]
                            img_w = images_tensor.shape[3]

                        x1 = gt_boxes[:, 0]
                        y1 = gt_boxes[:, 1]
                        x2 = gt_boxes[:, 2]
                        y2 = gt_boxes[:, 3]
                        cx = ((x1 + x2) / 2.0) / max(1.0, float(img_w))
                        cy = ((y1 + y2) / 2.0) / max(1.0, float(img_h))
                        bw = ((x2 - x1).clamp(min=1e-4)) / max(1.0, float(img_w))
                        bh = ((y2 - y1).clamp(min=1e-4)) / max(1.0, float(img_h))
                        norm_boxes = torch.stack([cx, cy, bw, bh], dim=-1).clamp(0.0, 1.0)

                        n_assign = min(len(gt_boxes), self.head.num_proposals)
                        # DENTEX classes are 1..num_classes; map to 0..num_classes-1
                        target_cls[i, :n_assign] = (gt_labels[:n_assign] - 1).clamp(
                            0, self.config.num_classes - 1
                        )
                        target_box[i, :n_assign] = norm_boxes[:n_assign]

            # Mixed-precision context
            device_type = self.device.type
            if device_type == "cuda" and self.scaler.is_enabled():
                try:
                    autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                except (AttributeError, TypeError):
                    autocast_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                feats = self.proj(self.backbone(images_tensor))
                logits, pred_boxes = self.head(feats, noisy_boxes, timesteps)

                loss_c = criterion_cls(
                    logits.view(-1, self.config.num_classes + 1),
                    target_cls.view(-1)
                )
                loss_b = criterion_box(pred_boxes, target_box)
                loss = loss_c + 2.0 * loss_b
                loss_val = loss.item()
                loss = loss / accum_steps

            self.scaler.scale(loss).backward()
            step += 1

            if step % accum_steps == 0 or (total_batches is not None and step == total_batches):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss_val

        if total_batches is None and accum_steps > 1 and step % accum_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(1, step)
        return {"loss": avg_loss}

    def predict(self, images: Union[torch.Tensor, List[torch.Tensor], tuple]) -> list:
        self.model.eval()
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images_list = [images]
            else:
                images_list = [img for img in images]
        else:
            images_list = list(images)

        if not images_list:
            return []

        results = []
        with torch.no_grad():
            images_tensor = self._prepare_images(images_list)
            b = images_tensor.size(0)
            feats = self.proj(self.backbone(images_tensor))

            # DDIM sampling from random Gaussian boxes / uniform boxes
            noisy_boxes = torch.rand((b, self.head.num_proposals, 4), device=self.device)
            t = torch.zeros((b,), device=self.device).float()
            logits, pred_boxes = self.head(feats, noisy_boxes, t)

            for i in range(b):
                probs = torch.softmax(logits[i], dim=-1)[:, :-1]  # exclude background class
                scores, labels = probs.max(dim=-1)
                keep = scores > 0.2

                boxes = pred_boxes[i][keep]
                orig_img = images_list[i]
                h = orig_img.shape[-2]
                w = orig_img.shape[-1]

                if boxes.numel() > 0:
                    x1 = ((boxes[:, 0] - boxes[:, 2] / 2.0) * float(w)).clamp(0.0, float(w))
                    y1 = ((boxes[:, 1] - boxes[:, 3] / 2.0) * float(h)).clamp(0.0, float(h))
                    x2 = ((boxes[:, 0] + boxes[:, 2] / 2.0) * float(w)).clamp(0.0, float(w))
                    y2 = ((boxes[:, 1] + boxes[:, 3] / 2.0) * float(h)).clamp(0.0, float(h))
                    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                else:
                    boxes_xyxy = torch.zeros((0, 4), device=boxes.device, dtype=torch.float32)

                results.append({
                    "boxes": boxes_xyxy.cpu(),
                    "labels": (labels[keep] + 1).cpu(),
                    "scores": scores[keep].cpu()
                })
        return results
