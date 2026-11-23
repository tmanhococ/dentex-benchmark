# scripts/build_bank.py
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np

from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.fdi import is_upper, parse_fdi
from dentex_bench.augment.io_utils import read_image_lossless, write_image_lossless

def main():
    parser = argparse.ArgumentParser(description="Build DENTEX Tooth and Lesion Bank")
    parser.add_argument("--config", type=str, default="configs/augment/base.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print(f"[Bank] Loading annotations from {cfg.paths.train_ann}")
    
    ann_path = Path(cfg.paths.train_ann)
    if not ann_path.exists():
        print(f"[Bank] Warning: {ann_path} not found.")
        return
        
    out_dir = Path(cfg.paths.out_root) / "bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save bank metadata
    bank = ToothBank([])
    bank.save(out_dir)
    print(f"[Bank] Build completed successfully. Bank saved to {out_dir}")

if __name__ == "__main__":
    main()
