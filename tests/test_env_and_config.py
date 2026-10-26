# tests/test_env_and_config.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dentex_bench.env import detect_platform, get_device_info, ensure_dependencies
from dentex_bench.config import BenchmarkConfig

def test_detect_platform():
    platform = detect_platform()
    assert platform in ["colab", "kaggle", "local"]

def test_get_device_info():
    info = get_device_info()
    assert "device" in info
    assert "vram_gb" in info
    assert "is_low_vram" in info
    assert isinstance(info["is_low_vram"], bool)

def test_benchmark_config_adaptive():
    cfg = BenchmarkConfig.from_auto_env()
    assert cfg.num_classes == 4
    assert cfg.class_names == ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]
    assert cfg.batch_size >= 1
    assert cfg.gradient_accumulation_steps >= 1
    if cfg.is_low_vram:
        assert cfg.batch_size <= 2
        assert cfg.gradient_accumulation_steps >= 2
