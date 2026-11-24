"""CLI Runner for DENTEX offline tooth-swap augmentation pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root so it runs standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.augment.config import load_config
from dentex_bench.augment.offline import run_offline_augmentation


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run DENTEX offline augmentation pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output directory")
    parser.add_argument("--overwrite", action="store_true", help="Alias for --force")
    parser.add_argument("--train-ann", type=str, default=None, help="Override train annotations path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--n-samples-per-image", type=int, default=None, help="Variants per image")
    parser.add_argument("--set", nargs="*", default=[], help="CLI overrides, e.g. paths.out_root=... offline.num_workers=2")
    args = parser.parse_args(argv)

    overrides = list(args.set)
    if args.force or args.overwrite:
        overrides.append("output.overwrite=true")
    if args.train_ann:
        overrides.append(f"paths.train_ann={args.train_ann}")
    if args.output_dir:
        overrides.append(f"paths.out_root={args.output_dir}")
    if args.seed is not None:
        overrides.append(f"global_seed={args.seed}")
    if args.n_samples_per_image is not None:
        overrides.append(f"offline.variants_per_image={args.n_samples_per_image}")

    cfg = load_config(args.config, cli_overrides=overrides)
    if args.force or args.overwrite:
        cfg.output.overwrite = True

    run_dir = run_offline_augmentation(cfg)
    manifest_path = Path(run_dir) / "state" / "manifest.json"

    print(f"[Offline] Augmentation complete for run {cfg.run_name}")
    print(f"[Offline] Run directory: {run_dir}")
    print(f"[Offline] Manifest path: {manifest_path}")


if __name__ == "__main__":
    main()
