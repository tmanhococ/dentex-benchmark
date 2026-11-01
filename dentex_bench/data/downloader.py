import os
import shutil
import zipfile
from pathlib import Path

def download_dentex_dataset(target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    existing_images = list(raw_dir.glob("**/*.png")) + list(raw_dir.glob("**/*.jpg"))
    if len(existing_images) > 100:
        print(f"[dentex_bench] Found {len(existing_images)} existing images in {raw_dir}.")
        return raw_dir

    print(f"[dentex_bench] Downloading DENTEX dataset from Hugging Face Hub (ibrahimhamamci/DENTEX)...")
    try:
        from huggingface_hub import snapshot_download
        # ONLY download training_data (contains the 705 Level 3 images and annotations for training/val benchmark)
        # This prevents downloading 12GB of unnecessary archives and avoids out-of-disk crashes
        download_path = snapshot_download(
            repo_id="ibrahimhamamci/DENTEX",
            repo_type="dataset",
            local_dir=str(raw_dir),
            allow_patterns=["*training_data*", "*.json", "*.md"],
            ignore_patterns=["*.git*", "*test_data*", "*validation_data*"]
        )
        print(f"[dentex_bench] Download complete to {download_path}")
    except Exception as e:
        print(f"[dentex_bench] Warning: Hugging Face snapshot_download failed: {e}")
        print("[dentex_bench] Please place raw DENTEX files in data/raw/ or provide local archive.")
        
    # Unpack zip files and immediately delete the zip to save disk space
    for zip_file in list(raw_dir.glob("**/*.zip")):
        print(f"[dentex_bench] Extracting {zip_file.name} and freeing archive space...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(zip_file.parent)
            # CRITICAL: Delete zip file after extraction to free 5GB disk space
            try:
                zip_file.unlink()
                print(f"[dentex_bench] Cleaned up {zip_file.name} to preserve disk space.")
            except Exception:
                pass
        except Exception as e:
            print(f"[dentex_bench] Warning: Failed to extract {zip_file}: {e}")
            
    return raw_dir
