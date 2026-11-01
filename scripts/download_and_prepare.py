import sys
import argparse
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.downloader import download_dentex_dataset
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

def main():
    parser = argparse.ArgumentParser(description="Download and prepare DENTEX dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Input data directory containing raw images/annotations")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory where processed annotations are saved")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()
    
    cfg = BenchmarkConfig.from_auto_env()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    
    # Check if raw files already exist in data_dir (e.g. read-only Kaggle input)
    existing_jsons = list(data_dir.glob("**/*.json")) if data_dir.exists() else []
    existing_imgs = list(data_dir.glob("**/*.png")) + list(data_dir.glob("**/*.jpg")) if data_dir.exists() else []
    
    if len(existing_jsons) > 0 and len(existing_imgs) > 0:
        print(f"[dentex_bench] Using existing dataset from {data_dir} ({len(existing_imgs)} images found).")
        raw_dir = data_dir
    else:
        raw_dir = download_dentex_dataset(data_dir)
    
    # Locate annotation json in raw directory
    json_files = list(raw_dir.glob("**/*.json"))
    if not json_files:
        print("[dentex_bench] No JSON annotation file found in data/raw yet. Please check dataset extraction.")
        return
        
    # Prioritize annotation json files for 4-class disease/pathology detection
    def score_ann_file(path: Path) -> int:
        p_str = path.as_posix().lower()
        if "instances" in p_str:
            return -100
        score = 0
        if "disease" in p_str:
            score += 100
        if "level_3" in p_str or "level3" in p_str:
            score += 80
        if "diagnosis" in p_str:
            score += 60
        if "train" in p_str:
            score += 10
        if "dentex" in p_str:
            score += 5
        # Penalize quadrant-only or enumeration-only files when looking for diseases
        if "quadrant" in p_str and "disease" not in p_str:
            score -= 50
        if "enumeration" in p_str and "disease" not in p_str:
            score -= 50
        return score

    json_candidates = [f for f in json_files if "instances" not in f.name]
    json_candidates.sort(key=score_ann_file, reverse=True)
    ann_file = json_candidates[0] if json_candidates else json_files[0]
    print(f"[dentex_bench] Using raw annotation file: {ann_file}")
    
    train_out = out_dir / "processed" / "instances_train.json"
    val_out = out_dir / "processed" / "instances_val.json"
    
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
