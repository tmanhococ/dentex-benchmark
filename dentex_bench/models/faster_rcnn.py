from typing import Any, Dict, List, Optional
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dentex_bench.models.base import BaseDetectorRunner


class FasterRCNNRunner(BaseDetectorRunner):
    def build_model(self, pretrained: bool = True):
        # 4 dental classes + 1 background = 5 classes
        num_classes = self.config.num_classes + 1
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        accum_steps = max(1, self.config.gradient_accumulation_steps)
        step = 0
        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=self.scaler.is_enabled())
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.scaler.is_enabled())

            with autocast_ctx:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_val = losses.item()
                losses = losses / accum_steps

            self.scaler.scale(losses).backward()
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

    def predict(self, images: list) -> list:
        self.model.eval()
        with torch.no_grad():
            images = [img.to(self.device) for img in images]
            predictions = self.model(images)
        return [{k: v.cpu() for k, v in p.items()} for p in predictions]
