from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from dentex_bench.models.base import BaseDetectorRunner


class FeaturePyramidNetwork(nn.Module):
    """
    Standard Feature Pyramid Network (FPN) producing multi-scale features P2..P5.
    """
    def __init__(self, in_channels: List[int] = [256, 512, 1024, 2048], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features: [C2, C3, C4, C5]
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=(h, w), mode="nearest")
            
        pyramid = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]
        return pyramid


class HierarchicalDetectionHead(nn.Module):
    """
    Multi-scale hierarchical detection head predicting 4 pathology classes and bounding boxes.
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 4, num_proposals: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.hidden_dim = in_channels
        
        # Learnable queries/proposals for pathology instances
        self.query_embed = nn.Embedding(num_proposals, in_channels)
        
        # Multi-scale pooling layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.level_embed = nn.Embedding(4, in_channels)
        
        # Detection fusion transformer/MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        # Output classification: 4 classes + 1 background class
        self.cls_head = nn.Linear(in_channels, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 4),
            nn.Sigmoid()  # normalized cx, cy, w, h in [0, 1]
        )

    def forward(self, pyramid: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # pyramid: [P2, P3, P4, P5]
        b = pyramid[0].shape[0]
        device = pyramid[0].device
        
        # Global multi-scale summary: pool each level and combine with level embedding
        level_summaries = []
        for lvl, feat in enumerate(pyramid):
            pooled = self.pool(feat).view(b, -1)  # (B, C)
            lvl_emb = self.level_embed.weight[lvl].unsqueeze(0).expand(b, -1)
            level_summaries.append(pooled + lvl_emb)
            
        global_context = torch.stack(level_summaries, dim=1).mean(dim=1)  # (B, C)
        
        # Expand queries across batch and fuse with multi-scale context
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, N, C)
        fused = self.fc(queries + global_context.unsqueeze(1))            # (B, N, C)
        
        logits = self.cls_head(fused)       # (B, N, num_classes + 1)
        pred_boxes = self.box_head(fused)   # (B, N, 4) in [0, 1]
        return logits, pred_boxes


class HierarchicalNetRunner(BaseDetectorRunner):
    """
    HierarchicalNet: ResNet50-FPN multi-scale feature hierarchy for 4-class dental abnormality detection.
    """
    def build_model(self, pretrained: bool = True):
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        
        # Backbone submodules
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # C2: 256
        self.layer2 = resnet.layer2  # C3: 512
        self.layer3 = resnet.layer3  # C4: 1024
        self.layer4 = resnet.layer4  # C5: 2048
        
        self.fpn = FeaturePyramidNetwork(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256
        )
        self.head = HierarchicalDetectionHead(
            in_channels=256,
            num_classes=self.config.num_classes,
            num_proposals=100
        )
        
        self.model = nn.ModuleDict({
            "stem": self.stem,
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
            "fpn": self.fpn,
            "head": self.head
        }).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model

    def _extract_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn([c2, c3, c4, c5])

    def _prepare_images(self, images: Union[torch.Tensor, List[torch.Tensor], tuple]) -> torch.Tensor:
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
                        p = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
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

            # Targets: default background class
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

                        img_h = images[i].shape[-2] if isinstance(images, (list, tuple)) and i < len(images) else images_tensor.shape[2]
                        img_w = images[i].shape[-1] if isinstance(images, (list, tuple)) and i < len(images) else images_tensor.shape[3]

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
                        # Map 1..num_classes to 0..num_classes-1
                        target_cls[i, :n_assign] = (gt_labels[:n_assign] - 1).clamp(
                            0, self.config.num_classes - 1
                        )
                        target_box[i, :n_assign] = norm_boxes[:n_assign]

            # Autocast context
            device_type = self.device.type
            if device_type == "cuda" and self.scaler.is_enabled():
                try:
                    autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                except (AttributeError, TypeError):
                    autocast_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                pyramid = self._extract_pyramid(images_tensor)
                logits, pred_boxes = self.head(pyramid)

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
                self.scaler.unscale_(self.optimizer)
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                finite = len(grads) > 0 and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
                if not finite:
                    # Drop batch if NaN / Inf gradient detected
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            total_loss += loss_val

        if total_batches is None and accum_steps > 1 and step % accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            finite = len(grads) > 0 and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
            if finite:
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.scaler.step(self.optimizer)
            else:
                self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(1, step)
        return {"loss": avg_loss}

    def predict(self, images: Union[torch.Tensor, List[torch.Tensor], tuple]) -> list:
        self.model.eval()
        if not isinstance(images, (list, tuple)):
            images = [images]

        results = []
        if len(images) == 0:
            return results

        images_tensor = self._prepare_images(images)

        device_type = self.device.type
        if device_type == "cuda" and self.scaler.is_enabled():
            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=True)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=True)
        else:
            autocast_ctx = nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                pyramid = self._extract_pyramid(images_tensor)
                logits, pred_boxes = self.head(pyramid)
                probs = F.softmax(logits, dim=-1)

        for i in range(len(images)):
            img = images[i]
            img_h = img.shape[-2]
            img_w = img.shape[-1]

            # Pathology scores (excluding background class index num_classes)
            scores_pathology = probs[i, :, :self.config.num_classes]
            best_scores, best_labels = torch.max(scores_pathology, dim=-1)

            # Map cx, cy, w, h normalized -> pixel xyxy
            cx = pred_boxes[i, :, 0] * img_w
            cy = pred_boxes[i, :, 1] * img_h
            bw = pred_boxes[i, :, 2] * img_w
            bh = pred_boxes[i, :, 3] * img_h

            x1 = (cx - bw / 2.0).clamp(min=0.0, max=float(img_w))
            y1 = (cy - bh / 2.0).clamp(min=0.0, max=float(img_h))
            x2 = (cx + bw / 2.0).clamp(min=0.0, max=float(img_w))
            y2 = (cy + bh / 2.0).clamp(min=0.0, max=float(img_h))

            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            # Labels 1..4 (1-indexed for DENTEX)
            labels_1based = best_labels + 1

            results.append({
                "boxes": boxes_xyxy.cpu(),
                "labels": labels_1based.cpu(),
                "scores": best_scores.cpu()
            })
        return results

    def save_checkpoint(self, path: Union[str, Path], extra_info: Optional[dict] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        if extra_info:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)
        print(f"[dentex_bench] Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> dict:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[dentex_bench] Loaded checkpoint from {path}")
        return checkpoint
