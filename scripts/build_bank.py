# scripts/build_bank.py
"""Build the DENTEX tooth bank (real polygon outlines, exact FDI) from the training split."""

import argparse
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank_builder import build_tooth_bank


def build_bank_from_config(cfg_path=None, out_bank_dir=None, train_ann=None):
    cfg = load_config(cfg_path or "configs/augment/base.yaml")
    train_ann_path = Path(train_ann) if train_ann else Path(cfg.paths.train_ann)
    bank_dir = Path(out_bank_dir) if out_bank_dir else Path(cfg.paths.out_root) / "bank"
    if not train_ann_path.exists():
        raise FileNotFoundError(f"Training annotations not found at: {train_ann_path}")
    bank, report = build_tooth_bank(train_ann_path, cfg.paths.img_dir, bank_dir, val_ann=cfg.paths.val_ann)
    print(f"[Bank] {report['teeth_in_bank']} teeth saved to {bank_dir}")
    print(f"[Bank] Report: {json.dumps(report)}")
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build DENTEX tooth bank")
    parser.add_argument("--config", type=str, default="configs/augment/base.yaml")
    parser.add_argument("--out", type=str, default=None, help="Output directory for tooth bank")
    parser.add_argument("--output", type=str, default=None, help="Alias for --out")
    parser.add_argument("--train-ann", type=str, default=None, help="Override training annotation path")
    parser.add_argument("--overwrite", action="store_true", help="Kept for compatibility; the bank is always rebuilt")
    args = parser.parse_args(argv)
    out = args.output or args.out
    build_bank_from_config(args.config, out_bank_dir=Path(out) if out else None, train_ann=args.train_ann)


if __name__ == "__main__":
    main()
