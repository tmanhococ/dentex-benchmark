"""Tests for experiment presets E0-E9 and ablation runner."""

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from dentex_bench.augment.config import load_config


PRESET_NAMES = [
    "E0_baseline.yaml",
    "E1_rfs.yaml",
    "E2_random_paste.yaml",
    "E3_fdi_hard.yaml",
    "E4_fdi_feather.yaml",
    "E5_fdi_poisson.yaml",
    "E6_core.yaml",
    "E7_best_rfs.yaml",
    "E8_hardneg.yaml",
    "E9_patchcrop.yaml",
]


def test_preset_configs_loadable():
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_dir = repo_root / "configs" / "augment"

    for p in PRESET_NAMES:
        cfg_path = config_dir / p
        assert cfg_path.exists(), f"Missing config preset {p}"
        cfg = load_config(cfg_path)
        assert cfg.version == 1
        stem = p.replace(".yaml", "")
        assert cfg.run_name == stem

    # Specific preset checks
    cfg_e0 = load_config(config_dir / "E0_baseline.yaml")
    assert cfg_e0.mode == "none"

    cfg_e1 = load_config(config_dir / "E1_rfs.yaml")
    assert cfg_e1.mode == "none"
    assert cfg_e1.sampling.get("rfs", {}).get("enabled") is True

    cfg_e2 = load_config(config_dir / "E2_random_paste.yaml")
    assert cfg_e2.placement.mode == "random"
    assert cfg_e2.blend.mode == "hard"
    assert cfg_e2.blend.inpaint_target is False

    cfg_e3 = load_config(config_dir / "E3_fdi_hard.yaml")
    assert cfg_e3.placement.mode == "fdi_swap"
    assert cfg_e3.blend.mode == "hard"

    cfg_e4 = load_config(config_dir / "E4_fdi_feather.yaml")
    assert cfg_e4.placement.mode == "fdi_swap"
    assert cfg_e4.blend.mode == "feather"

    cfg_e5 = load_config(config_dir / "E5_fdi_poisson.yaml")
    assert cfg_e5.placement.mode == "fdi_swap"
    assert cfg_e5.blend.mode == "poisson_normal"

    cfg_e6 = load_config(config_dir / "E6_core.yaml")
    assert cfg_e6.placement.mode == "fdi_swap"
    assert cfg_e6.blend.mode == "core_preserving"
    assert cfg_e6.blend.intensity_match == "histogram"

    cfg_e7 = load_config(config_dir / "E7_best_rfs.yaml")
    assert cfg_e7.placement.mode == "fdi_swap"
    assert cfg_e7.blend.mode == "core_preserving"
    assert cfg_e7.sampling.get("rfs", {}).get("enabled") is True

    cfg_e8 = load_config(config_dir / "E8_hardneg.yaml")
    assert cfg_e8.extras.get("hard_negative", {}).get("enabled") is True

    cfg_e9 = load_config(config_dir / "E9_patchcrop.yaml")
    assert cfg_e9.extras.get("patch_crop", {}).get("enabled") is True


def test_ablation_script_help():
    repo_root = Path(__file__).resolve().parent.parent.parent
    script_path = repo_root / "scripts" / "run_ablation.py"
    assert script_path.exists(), "scripts/run_ablation.py does not exist"

    cmd = [sys.executable, str(script_path), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--seeds" in res.stdout
    assert "--out-root" in res.stdout


def test_ablation_script_dry_run():
    from scripts.run_ablation import run_ablation

    commands = run_ablation(
        seeds=[0, 1],
        out_root="outputs/augment_test",
        dry_run=True,
    )
    # 2 seeds * (1 E3 run + 3 replays: E4, E5, E6) = 8 commands
    assert len(commands) == 8

    # Verify seed 0 calls
    e3_seed0 = commands[0]
    assert any("augment_offline.py" in part for part in e3_seed0)
    assert any("global_seed=0" in part for part in e3_seed0)
    assert any("E3_fdi_hard.yaml" in part for part in e3_seed0)

    e4_seed0 = commands[1]
    assert any("replay_augment.py" in part for part in e4_seed0)
    assert any("blend.mode=feather" in part for part in e4_seed0)
    assert any("E3_fdi_hard_seed0" in part for part in e4_seed0)

    e5_seed0 = commands[2]
    assert any("blend.mode=poisson_normal" in part for part in e5_seed0)

    e6_seed0 = commands[3]
    assert any("blend.mode=core_preserving" in part for part in e6_seed0)

    # Verify seed 1 calls
    e3_seed1 = commands[4]
    assert any("global_seed=1" in part for part in e3_seed1)


def test_ablation_runner_mock_execution():
    from scripts.run_ablation import run_ablation

    mock_runner = MagicMock()
    commands = run_ablation(
        seeds=[42],
        out_root="outputs/test_ablation",
        dry_run=False,
        runner=mock_runner,
    )
    assert len(commands) == 4
    assert mock_runner.call_count == 4
