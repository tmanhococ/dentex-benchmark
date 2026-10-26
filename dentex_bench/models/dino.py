from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from dentex_bench.models.base import BaseDetectorRunner


class SimpleDETRHead(nn.Module):
    """Transformer-based detection head with learnable object queries."""

    def __init__(self, num_classes: int = 4, hidden_dim: int = 256, num_queries: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Output classes: num_classes dental pathologies + 1 background class
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = features.shape
        feat_flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        out = self.transformer(feat_flat)

        # Aggregate to queries: align feature length with num_queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        if out.shape[1] < self.num_queries:
            pad_len = self.num_queries - out.shape[1]
            out_features = torch.nn.functional.pad(out, (0, 0, 0, pad_len))
        else:
            out_features = out[:, :self.num_queries, :]

        combined = out_features + queries

        pred_logits = self.class_head(combined)
        pred_boxes = self.bbox_head(combined)
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class DINORunner(BaseDetectorRunner):
    """DINO / DETR Transformer-based architecture runner for DENTEX."""

    def build_model(self, pretrained: bool = True):
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        # Remove avgpool and fc layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.head = SimpleDETRHead(
            num_classes=self.config.num_classes,
            hidden_dim=256,
            num_queries=100
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

        cls_criterion = nn.CrossEntropyLoss()
        l1_criterion = nn.L1Loss()

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

            # Default: all queries assigned to background (index num_classes)
            target_cls = torch.full(
                (b, self.head.num_queries),
                self.config.num_classes,
                dtype=torch.long,
                device=self.device
            )
            target_box = torch.zeros(
                (b, self.head.num_queries, 4),
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

                        n_assign = min(len(gt_boxes), self.head.num_queries)
                        # DENTEX classes are 1..num_classes; map to 0..num_classes-1
                        target_cls[i, :n_assign] = (gt_labels[:n_assign] - 1).clamp(
                            0, self.config.num_classes - 1
                        )
                        target_box[i, :n_assign] = norm_boxes[:n_assign]

            # Mixed-precision context
            device_type = self.device.type
            if device_type == "cuda":
                try:
                    autocast_ctx = torch.amp.autocast("cuda", enabled=self.scaler.is_enabled())
                except (AttributeError, TypeError):
                    autocast_ctx = torch.cuda.amp.autocast(enabled=self.scaler.is_enabled())
            else:
                autocast_ctx = torch.amp.autocast("cpu", enabled=False)

            with autocast_ctx:
                feats = self.proj(self.backbone(images_tensor))
                preds = self.head(feats)

                loss_cls = cls_criterion(
                    preds["pred_logits"].view(-1, self.config.num_classes + 1),
                    target_cls.view(-1)
                )
                loss_box = l1_criterion(preds["pred_boxes"], target_box)
                loss = loss_cls + 5.0 * loss_box
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
            images_list = [images] if images.ndim == 3 else list(images)
        else:
            images_list = list(images)

        if not images_list:
            return []

        results = []
        with torch.no_grad():
            images_tensor = self._prepare_images(images_list)
            feats = self.proj(self.backbone(images_tensor))
            preds = self.head(feats)

            for i in range(len(images_list)):
                logits = preds["pred_logits"][i]
                probs = torch.softmax(logits, dim=-1)[:, :-1]  # exclude background class
                scores, labels = probs.max(dim=-1)

                # Filter by confidence threshold 0.2
                keep = scores > 0.2
                boxes = preds["pred_boxes"][i][keep]

                # Denormalize to individual image coords
                orig_img = images_list[i]
                h = orig_img.shape[-2]
                w = orig_img.shape[-1]
                scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=torch.float32)

                if boxes.numel() > 0:
                    boxes_xyxy = torch.stack([
                        boxes[:, 0] - boxes[:, 2] / 2.0,
                        boxes[:, 1] - boxes[:, 3] / 2.0,
                        boxes[:, 0] + boxes[:, 2] / 2.0,
                        boxes[:, 1] + boxes[:, 3] / 2.0,
                    ], dim=-1) * scale
                else:
                    boxes_xyxy = torch.zeros((0, 4), device=boxes.device, dtype=torch.float32)

                results.append({
                    "boxes": boxes_xyxy.cpu(),
                    "labels": (labels[keep] + 1).cpu(),
                    "scores": scores[keep].cpu()
                })

        return results
