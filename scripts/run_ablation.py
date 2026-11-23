"""Orchestrator for E3-E6 DENTEX Augmentation Ablation.

Executes E3_fdi_hard offline augmentation once per seed to generate the canonical
placement state, then replays E4 (feather), E5 (poisson_normal), and E6 (core_preserving)
from the shared E3 state using execution-only overrides.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, List, Optional, Tuple, Union


DEFAULT_VARIANTS: List[Tuple[str, str]] = [
    ("E4_fdi_feather", "feather"),
    ("E5_fdi_poisson", "poisson_normal"),
    ("E6_core", "core_preserving"),
]


def run_ablation(
    seeds: Optional[List[int]] = None,
    out_root: Union[str, Path] = "outputs/augment",
    config_e3: Union[str, Path] = "configs/augment/E3_fdi_hard.yaml",
    dry_run: bool = False,
    force: bool = False,
    export_dataset: bool = True,
    runner: Optional[Callable[[List[str]], Any]] = None,
) -> List[List[str]]:
    """Orchestrate E3-E6 ablation experiments across one or more seeds.

    Args:
        seeds: List of random seeds to evaluate (default: [0]).
        out_root: Root directory for output artifacts.
        config_e3: Path to E3 baseline configuration YAML.
        dry_run: If True, plan commands without execution.
        force: If True, force rerun and overwrite existing outputs.
        export_dataset: If True, export replayed variants to dataset directory.
        runner: Optional custom execution callback for testing.

    Returns:
        List of command argument lists executed or planned.
    """
    if seeds is None:
        seeds = [0]

    out_root_path = Path(out_root)
    config_e3_path = Path(config_e3)
    all_commands: List[List[str]] = []

    for seed in seeds:
        print(f"\n==========================================")
        print(f"=== Running E3 Base for Seed {seed} ===")
        print(f"==========================================")

        cmd_e3 = [
            sys.executable,
            "scripts/augment_offline.py",
            "--config",
            str(config_e3_path),
            "--set",
            f"global_seed={seed}",
            f"paths.out_root={out_root_path.as_posix()}",
        ]
        if force:
            cmd_e3.append("--force")

        all_commands.append(cmd_e3)
        if dry_run:
            print(f"[DRY-RUN] Would run: {' '.join(cmd_e3)}")
        elif runner is not None:
            runner(cmd_e3)
        else:
            subprocess.run(cmd_e3, check=True)

        # E3 state directory created by augment_offline
        e3_state = out_root_path / f"E3_fdi_hard_seed{seed}" / "state"

        for variant, mode in DEFAULT_VARIANTS:
            print(f"\n=== Replaying {variant} ({mode}) for Seed {seed} ===")
            out_dir = out_root_path / f"{variant}_seed{seed}"
            cmd_replay = [
                sys.executable,
                "scripts/replay_augment.py",
                "--state",
                str(e3_state),
                "--override",
                f"blend.mode={mode}",
                "--out",
                str(out_dir),
            ]
            if export_dataset:
                cmd_replay.append("--export-dataset")

            all_commands.append(cmd_replay)
            if dry_run:
                print(f"[DRY-RUN] Would run: {' '.join(cmd_replay)}")
            elif runner is not None:
                runner(cmd_replay)
            else:
                subprocess.run(cmd_replay, check=True)

    return all_commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate E3-E6 Ablation with shared placement across seeds."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds to evaluate (default: [0])",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="outputs/augment",
        help="Output root directory (default: outputs/augment)",
    )
    parser.add_argument(
        "--config-e3",
        type=str,
        default="configs/augment/E3_fdi_hard.yaml",
        help="Path to E3 fdi_hard configuration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    parser.add_argument(
        "--export-dataset",
        action="store_true",
        default=True,
        help="Export COCO/YOLO datasets in replay steps (default: True)",
    )
    args = parser.parse_args()

    run_ablation(
        seeds=args.seeds,
        out_root=args.out_root,
        config_e3=args.config_e3,
        dry_run=args.dry_run,
        force=args.force,
        export_dataset=args.export_dataset,
    )


if __name__ == "__main__":
    main()
