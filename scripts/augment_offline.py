"""CLI Runner for DENTEX offline tooth-swap augmentation pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root so it runs standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.augment.config import load_config
from dentex_bench.augment.offline import run_offline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DENTEX offline augmentation pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")
    parser.add_argument("--set", nargs="*", default=[], help="CLI overrides, e.g. paths.out_root=... offline.num_workers=2")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.set)
    run_dir = run_offline(cfg)
    manifest_path = Path(run_dir) / "state" / "manifest.json"

    print(f"[Offline] Augmentation complete for run {cfg.run_name}")
    print(f"[Offline] Run directory: {run_dir}")
    print(f"[Offline] Manifest path: {manifest_path}")


if __name__ == "__main__":
    main()
