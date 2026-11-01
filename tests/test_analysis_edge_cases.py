import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from dentex_bench.config import BenchmarkConfig
from dentex_bench.models import HierarchicalNetRunner, DiffusionNetRunner

def test_full_resolution_radiograph_handling():
    """Tests that images with real clinical dimensions (1504x2872) process without broadcasting errors."""
    h, w = 1504, 2872
    grad_1d = np.linspace(40, 200, w, dtype=np.uint8)
    gray = np.tile(grad_1d, (h, 1))
    assert gray.shape == (h, w)
    
    import cv2
    cv2.ellipse(gray, (w // 2, h - 50), (w // 3, h // 2), 0, 180, 360, 230, max(2, int(min(h, w) * 0.015)))
    synth_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    synth_bgr = cv2.cvtColor(synth_rgb, cv2.COLOR_RGB2BGR)
    assert synth_rgb.shape == (h, w, 3)
    assert synth_bgr.shape == (h, w, 3)
    
    cfg = BenchmarkConfig.from_auto_env()
    hnet = HierarchicalNetRunner(cfg)
    hnet.build_model(pretrained=False)
    
    dnet = DiffusionNetRunner(cfg)
    dnet.build_model(pretrained=False)
    
    # Test tensor pass
    tensor = torch.from_numpy(synth_rgb).permute(2, 0, 1).float() / 255.0
    preds_h = hnet.predict([tensor])
    preds_d = dnet.predict([tensor])
    
    assert len(preds_h) == 1
    assert len(preds_d) == 1
    assert "boxes" in preds_h[0]
    assert "boxes" in preds_d[0]
