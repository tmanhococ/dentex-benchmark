from typing import List

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dentex_bench.models.base import BaseDetectorRunner


def build_fpn_detector(config, pretrained: bool):
    """
    ResNet-50 -> FPN (P2..P6) -> RPN over every pyramid level -> multi-level RoIAlign -> box head.

    Input normalisation (ImageNet mean/std) and resizing (shorter side config.img_size, longer side
    at most config.max_size) happen inside the model, and predictions are mapped back to the
    resolution of the image that was passed in.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=config.img_size,
        max_size=config.max_size,
        box_score_thresh=0.001,
        box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes + 1)
    freeze_backbone_stem(model.backbone)
    return model


def freeze_backbone_stem(backbone, trainable_layers=("layer2", "layer3", "layer4")):
    """
    Freezes the ResNet stem and layer1, as torchvision does for pretrained detectors. Done explicitly
    so the set of trainable parameters (and the optimizer state in checkpoints) is identical whether
    or not pretrained weights were loaded.
    """
    for name, param in backbone.body.named_parameters():
        if not name.startswith(trainable_layers):
            param.requires_grad_(False)


class HierarchicalNetRunner(BaseDetectorRunner):
    """
    HierarchicalNet: multi-scale feature hierarchy (ResNet-50 + FPN) with a two-stage head.
    Region proposals are generated on every pyramid level and each RoI is pooled from the
    level that matches its size, so small caries and large impacted molars use different scales.
    """
    ARCH_ID = "hierarchical_net/resnet50-fpn-rpn-roialign/v2"

    def build_model(self, pretrained: bool = True):
        self.model = build_fpn_detector(self.config, pretrained).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model

    def train_epoch(self, dataloader, epoch: int) -> dict:
        def compute_loss(images, targets):
            return sum(self.model(images, targets).values())
        return self._run_train_epoch(dataloader, compute_loss)

    def predict(self, images: List[torch.Tensor]) -> list:
        """images: CHW float tensors in [0, 1] of any size. Boxes are xyxy in each input's pixels."""
        self.model.eval()
        if isinstance(images, torch.Tensor):
            images = [images] if images.ndim == 3 else list(images)
        if len(images) == 0:
            return []
        with torch.no_grad(), self._autocast():
            outputs = self.model([img.to(self.device).float() for img in images])
        return [
            {"boxes": o["boxes"].float().cpu(), "labels": o["labels"].cpu(), "scores": o["scores"].float().cpu()}
            for o in outputs
        ]
