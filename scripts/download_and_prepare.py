import argparse
from pathlib import Path
from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.downloader import download_dentex_dataset
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

def main():
    parser = argparse.ArgumentParser(description="Download and prepare DENTEX dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()
    
    cfg = BenchmarkConfig.from_auto_env()
    data_dir = Path(args.data_dir)
    raw_dir = download_dentex_dataset(data_dir)
    
    # Locate annotation json in raw directory
    json_files = list(raw_dir.glob("**/*.json"))
    if not json_files:
        print("[dentex_bench] No JSON annotation file found in data/raw yet. Please check dataset extraction.")
        return
        
    # Prioritize annotation json files over other config/metadata json
    candidates = [
        f for f in json_files 
        if "instances" not in f.name and ("train" in f.name.lower() or "dentex" in f.name.lower() or "annotation" in f.name.lower())
    ]
    ann_file = candidates[0] if candidates else json_files[0]
    print(f"[dentex_bench] Using raw annotation file: {ann_file}")
    
    train_out = data_dir / "processed" / "instances_train.json"
    val_out = data_dir / "processed" / "instances_val.json"
    
    stats = convert_dentex_annotations_to_coco(
        raw_ann_file=ann_file,
        train_out=train_out,
        val_out=val_out,
        split_ratio=cfg.train_val_split_ratio,
        seed=args.seed
    )
    print(f"[dentex_bench] COCO format dataset prepared:")
    print(f"  Train: {stats['train_images']} images, {stats['train_annotations']} annotations -> {train_out}")
    print(f"  Val:   {stats['val_images']} images, {stats['val_annotations']} annotations -> {val_out}")

if __name__ == "__main__":
    main()
