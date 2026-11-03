"""
DiffusionNet: DiffusionDet-style detector (Chen et al., ICCV 2023) for 4 dental pathologies.

Training corrupts the ground-truth boxes with the forward diffusion process q(z_t | z_0) and learns
to recover z_0 from RoI features pooled *at the noisy boxes*. Inference starts from Gaussian boxes
and runs DDIM sampling with box renewal. Components:
- ResNet-50 + FPN backbone (same COCO-pretrained initialisation as HierarchicalNet)
- cascade of refinement stages: multi-level RoIAlign -> MLP -> proposal self-attention -> FiLM(t)
- one-to-one Hungarian matching, sigmoid focal + L1 + GIoU losses on every stage
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import MultiScaleRoIAlign, batched_nms, generalized_box_iou, sigmoid_focal_loss
from torchvision.ops import box_convert

from dentex_bench.models.base import BaseDetectorRunner
from dentex_bench.models.hierarchical_net import freeze_backbone_stem


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


def cosine_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(0, 0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


class RefinementStage(nn.Module):
    """One cascade stage: refines every box from the RoI features pooled at its current location."""

    BBOX_WEIGHTS = (2.0, 2.0, 1.0, 1.0)
    SCALE_CLAMP = math.log(1000.0 / 16)

    def __init__(self, num_classes: int, hidden_dim: int = 256, pooler_resolution: int = 7, time_dim: int = 1024):
        super().__init__()
        self.roi_mlp = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(hidden_dim * pooler_resolution ** 2, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, hidden_dim), nn.ReLU(inplace=True),
        )
        self.self_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 1024), nn.ReLU(inplace=True), nn.Linear(1024, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.time_film = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, hidden_dim * 2))
        self.cls_branch = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))
        self.reg_branch = nn.Sequential(*[
            m for _ in range(3) for m in (nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))
        ])
        self.class_logits = nn.Linear(hidden_dim, num_classes)
        self.bbox_deltas = nn.Linear(hidden_dim, 4)

        # Focal-loss prior: start with ~1% foreground probability
        nn.init.constant_(self.class_logits.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.zeros_(self.bbox_deltas.weight)
        nn.init.zeros_(self.bbox_deltas.bias)

    def forward(self, roi_features, prev_features, boxes, time_emb):
        # roi_features: (B*N, C, R, R); boxes: (B, N, 4) xyxy pixels; time_emb: (B, time_dim)
        b, n = boxes.shape[:2]
        feat = self.roi_mlp(roi_features).view(b, n, -1)
        if prev_features is not None:
            feat = feat + prev_features
        feat = self.norm1(feat + self.self_attn(feat, feat, feat, need_weights=False)[0])
        feat = self.norm2(feat + self.ffn(feat))
        scale, shift = self.time_film(time_emb).unsqueeze(1).chunk(2, dim=-1)
        feat = feat * (scale + 1) + shift

        logits = self.class_logits(self.cls_branch(feat))
        deltas = self.bbox_deltas(self.reg_branch(feat))
        return logits, self.apply_deltas(deltas, boxes), feat

    def apply_deltas(self, deltas, boxes):
        wx, wy, ww, wh = self.BBOX_WEIGHTS
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights
        dx, dy = deltas[..., 0] / wx, deltas[..., 1] / wy
        dw = (deltas[..., 2] / ww).clamp(max=self.SCALE_CLAMP)
        dh = (deltas[..., 3] / wh).clamp(max=self.SCALE_CLAMP)
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        return torch.stack([pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                            pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h], dim=-1)


class DiffusionDetector(nn.Module):
    def __init__(self, backbone, transform, num_classes: int = 4, num_proposals: int = 100,
                 num_stages: int = 3, hidden_dim: int = 256, timesteps: int = 1000,
                 sampling_steps: int = 4, signal_scale: float = 2.0, renewal_threshold: float = 0.5,
                 sampling_seed: int = 0):
        super().__init__()
        self.backbone = backbone
        self.transform = transform
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.scale = signal_scale
        self.renewal_threshold = renewal_threshold
        self.sampling_seed = sampling_seed

        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(timesteps), persistent=False)
        self.box_pooler = MultiScaleRoIAlign(["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        time_dim = hidden_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim),
        )
        self.stages = nn.ModuleList([RefinementStage(num_classes, hidden_dim, 7, time_dim) for _ in range(num_stages)])

    # ---- box space helpers: diffusion runs on scaled, normalised cxcywh -----------------------
    def _to_signal(self, boxes_xyxy, size_whwh):
        cxcywh = box_convert(boxes_xyxy / size_whwh, "xyxy", "cxcywh")
        return (cxcywh * 2.0 - 1.0) * self.scale

    def _to_boxes(self, signal, size_whwh):
        cxcywh = ((signal.clamp(-self.scale, self.scale) / self.scale) + 1.0) / 2.0
        # A zero-size proposal could never grow (deltas scale the current width/height)
        cxcywh = torch.cat([cxcywh[..., :2], cxcywh[..., 2:].clamp(min=1e-3)], dim=-1)
        return box_convert(cxcywh, "cxcywh", "xyxy") * size_whwh

    def _run_stages(self, features, boxes, t, image_sizes):
        time_emb = self.time_mlp(t)
        outputs, prev = [], None
        for stage in self.stages:
            rois = self.box_pooler(features, [b for b in boxes], image_sizes)
            logits, boxes, prev = stage(rois, prev, boxes, time_emb)
            outputs.append((logits, boxes))
            boxes = boxes.detach()
        return outputs

    def forward(self, images: List[torch.Tensor], targets=None):
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        image_list, targets = self.transform(images, targets)
        features = self.backbone(image_list.tensors)
        sizes = torch.tensor([[w, h, w, h] for h, w in image_list.image_sizes],
                             dtype=torch.float32, device=image_list.tensors.device).unsqueeze(1)
        if self.training:
            return self._loss(features, image_list.image_sizes, sizes, targets)
        detections = self._sample(features, image_list.image_sizes, sizes)
        return self.transform.postprocess(detections, image_list.image_sizes, original_sizes)

    # ---- training ---------------------------------------------------------------------------
    def _noisy_proposals(self, gt_boxes, size_whwh):
        device = size_whwh.device
        gt_signal = self._to_signal(gt_boxes, size_whwh[0]) if len(gt_boxes) else gt_boxes.new_zeros((0, 4))
        num_gt = gt_signal.shape[0]
        if num_gt > self.num_proposals:
            gt_signal = gt_signal[torch.randperm(num_gt, device=device)[:self.num_proposals]]
        elif num_gt < self.num_proposals:
            pad = torch.randn(self.num_proposals - num_gt, 4, device=device) / 6.0 + 0.5
            pad[:, 2:] = pad[:, 2:].clamp(min=1e-4)
            gt_signal = torch.cat([gt_signal, (pad * 2.0 - 1.0) * self.scale], dim=0)
        t = torch.randint(0, self.timesteps, (1,), device=device)
        ab = self.alphas_cumprod[t]
        noisy = ab.sqrt() * gt_signal + (1 - ab).sqrt() * torch.randn_like(gt_signal)
        return self._to_boxes(noisy, size_whwh[0]), t

    def _loss(self, features, image_sizes, sizes, targets):
        proposals, ts = zip(*[self._noisy_proposals(tg["boxes"], sizes[i]) for i, tg in enumerate(targets)])
        outputs = self._run_stages(features, torch.stack(proposals), torch.cat(ts).float(), image_sizes)
        return sum(self._set_loss(logits, boxes, targets, sizes) for logits, boxes in outputs)

    @torch.no_grad()
    def _match(self, logits, boxes, gt_boxes, gt_labels, size_whwh):
        prob = logits.float().sigmoid()[:, gt_labels]
        neg = (1 - 0.25) * prob ** 2 * -(1 - prob + 1e-8).log()
        pos = 0.25 * (1 - prob) ** 2 * -(prob + 1e-8).log()
        cost = (2.0 * (pos - neg)
                + 5.0 * torch.cdist(boxes.float() / size_whwh, gt_boxes / size_whwh, p=1)
                - 2.0 * generalized_box_iou(boxes.float(), gt_boxes))
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(rows, device=logits.device), torch.as_tensor(cols, device=logits.device)

    def _set_loss(self, logits, boxes, targets, sizes):
        num_gt = max(1, sum(len(t["labels"]) for t in targets))
        cls_target = torch.zeros_like(logits, dtype=torch.float32)
        l1, giou = logits.new_zeros(()).float(), logits.new_zeros(()).float()
        for i, tg in enumerate(targets):
            if len(tg["labels"]) == 0:
                continue
            gt_labels = tg["labels"] - 1
            rows, cols = self._match(logits[i], boxes[i], tg["boxes"], gt_labels, sizes[i])
            cls_target[i, rows, gt_labels[cols]] = 1.0
            matched, gt = boxes[i, rows].float(), tg["boxes"][cols]
            l1 = l1 + F.l1_loss(matched / sizes[i], gt / sizes[i], reduction="sum")
            giou = giou + (1 - torch.diag(generalized_box_iou(matched, gt))).sum()
        cls = sigmoid_focal_loss(logits.float(), cls_target, alpha=0.25, gamma=2.0, reduction="sum")
        return (2.0 * cls + 5.0 * l1 + 2.0 * giou) / num_gt

    # ---- inference: DDIM sampling with box renewal ------------------------------------------
    @torch.no_grad()
    def _sample(self, features, image_sizes, sizes) -> List[Dict[str, torch.Tensor]]:
        b = sizes.shape[0]
        device = sizes.device
        # Private generator: identical inputs give identical detections without touching global RNG state
        gen = torch.Generator(device=device).manual_seed(self.sampling_seed)
        noise = lambda: torch.randn((b, self.num_proposals, 4), generator=gen, device=device)
        times = torch.linspace(-1, self.timesteps - 1, self.sampling_steps + 1).long().flip(0).tolist()
        signal = noise()
        collected = [[] for _ in range(b)]

        for time, time_next in zip(times[:-1], times[1:]):
            t = torch.full((b,), float(time), device=device)
            logits, boxes = self._run_stages(features, self._to_boxes(signal, sizes), t, image_sizes)[-1]
            logits, boxes = logits.float(), boxes.float()
            scores = logits.sigmoid()
            for i in range(b):
                collected[i].append((boxes[i], scores[i]))
            if time_next < 0:
                break

            x_start = self._to_signal(boxes, sizes).clamp(-self.scale, self.scale)
            ab, ab_next = self.alphas_cumprod[time], self.alphas_cumprod[time_next]
            pred_noise = ((1.0 / ab).sqrt() * signal - x_start) / ((1.0 / ab) - 1).sqrt()
            sigma = ((1 - ab / ab_next) * (1 - ab_next) / (1 - ab)).sqrt()
            c = (1 - ab_next - sigma ** 2).sqrt()
            next_signal = x_start * ab_next.sqrt() + c * pred_noise + sigma * noise()
            # Box renewal: replace proposals that converged on background with fresh Gaussian boxes
            keep = scores.max(dim=-1).values > self.renewal_threshold
            signal = torch.where(keep.unsqueeze(-1), next_signal, noise())

        detections = []
        for i in range(b):
            boxes = torch.cat([bx for bx, _ in collected[i]])
            scores = torch.cat([sc for _, sc in collected[i]])
            h, w = image_sizes[i]
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            flat_scores, flat_idx = scores.flatten().topk(min(scores.numel(), self.num_proposals * len(collected[i])))
            box_idx = flat_idx // self.num_classes
            labels = flat_idx % self.num_classes + 1
            cand = boxes[box_idx]
            valid = (cand[:, 2] - cand[:, 0] > 1e-2) & (cand[:, 3] - cand[:, 1] > 1e-2)
            cand, flat_scores, labels = cand[valid], flat_scores[valid], labels[valid]
            keep = batched_nms(cand, flat_scores, labels, 0.5)[:self.num_proposals]
            detections.append({"boxes": cand[keep], "scores": flat_scores[keep], "labels": labels[keep]})
        return detections


class DiffusionNetRunner(BaseDetectorRunner):
    """
    DiffusionNet: progressive box denoising detector (DiffusionDet paradigm) for 4 dental pathologies.
    """
    ARCH_ID = "diffusion_net/diffusiondet-roialign-fpn/v2"

    def build_model(self, pretrained: bool = True):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        # Reuse the COCO-pretrained ResNet-50-FPN and its resize/normalise transform
        reference = fasterrcnn_resnet50_fpn_v2(
            weights=weights, weights_backbone=None,
            min_size=self.config.img_size, max_size=self.config.max_size,
        )
        freeze_backbone_stem(reference.backbone)
        self.model = DiffusionDetector(
            backbone=reference.backbone,
            transform=reference.transform,
            num_classes=self.config.num_classes,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model

    def train_epoch(self, dataloader, epoch: int) -> dict:
        return self._run_train_epoch(dataloader, lambda images, targets: self.model(images, targets))

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
