import pytest
from pathlib import Path
from dentex_bench.augment.config import load_config, AugmentConfig, config_hash

def test_load_base_config():
    cfg_path = Path("configs/augment/base.yaml")
    assert cfg_path.exists()
    cfg = load_config(cfg_path)
    assert isinstance(cfg, AugmentConfig)
    assert cfg.global_seed == 0
    assert cfg.placement.mode == "fdi_swap"
    assert cfg.output.image_ext == "png"

def test_config_hash_consistency():
    cfg1 = load_config("configs/augment/base.yaml")
    cfg2 = load_config("configs/augment/base.yaml")
    assert config_hash(cfg1) == config_hash(cfg2)

def test_invalid_config_raises():
    with pytest.raises(ValueError, match="vflip"):
        load_config("configs/augment/base.yaml", cli_overrides=["base_transforms.vflip=True"])
        
    with pytest.raises(ValueError, match="rotate"):
        load_config("configs/augment/base.yaml", cli_overrides=["base_transforms.rotate.limit_deg=15"])
        
    with pytest.raises(ValueError, match="lossless"):
        load_config("configs/augment/base.yaml", cli_overrides=["output.image_ext=jpg"])
