import os
import json
import shutil
import zipfile
from pathlib import Path

def selective_extract_dentex_zip(zip_file: Path, extract_dir: Path):
    """
    Selectively extract only the 4-class disease/pathology files (JSON annotations and the ~705 disease X-ray images).
    Skipping the unused ~5000 quadrant and enumeration images saves >10 GB of disk space, preventing Kaggle 20 GB disk exhaustion.
    """
    with zipfile.ZipFile(zip_file, 'r') as z:
        all_members = z.namelist()
        
        # Step 1: Extract all .json files first (tiny, < 5 MB)
        json_members = [m for m in all_members if m.lower().endswith(".json") and not m.startswith("__MACOSX")]
        for jm in json_members:
            z.extract(jm, extract_dir)
            
        # Step 2: Score the extracted JSON files to find the disease/Level 3 annotation
        extracted_jsons = list(extract_dir.glob("**/*.json"))
        def score_json(p: Path) -> int:
            s = p.as_posix().lower()
            score = 0
            if "test" in s:
                score -= 100
            if "disease" in s:
                score += 100
            if "level_3" in s or "level3" in s:
                score += 80
            if "quadrant" in s and "disease" not in s:
                score -= 50
            if "enumeration" in s and "disease" not in s:
                score -= 50
            return score
            
        extracted_jsons.sort(key=score_json, reverse=True)
        best_json = extracted_jsons[0] if extracted_jsons else None
        
        # Step 3: Get all target image filenames from the best annotation JSON
        target_img_names = set()
        if best_json and best_json.is_file():
            try:
                with open(best_json, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                img_rows = data if isinstance(data, list) else data.get("images", [data] if "file_name" in data else [])
                for im in img_rows:
                    if isinstance(im, dict) and "file_name" in im:
                        target_img_names.add(Path(str(im["file_name"])).name.lower())
            except Exception as err:
                print(f"[dentex_bench] Note: Error reading {best_json}: {err}")
                
        # Step 4: Extract ONLY the disease images and disease folder files
        print(f"[dentex_bench] Selective extraction: targeting {len(target_img_names)} disease images from {best_json.name if best_json else 'archive'}...")
        extracted_count = 0
        for m in all_members:
            if m.startswith("__MACOSX") or m.endswith("/"):
                continue
            m_lower = m.lower()
            m_name = Path(m).name.lower()
            
            # Match if member is one of the target images, or is inside the disease folder
            is_target_img = m_name in target_img_names
            is_disease_dir = ("quadrant_enumeration_disease" in m_lower or "disease" in m_lower) and ("xray" in m_lower or "image" in m_lower)
            
            if is_target_img or (not target_img_names and is_disease_dir):
                z.extract(m, extract_dir)
                extracted_count += 1
                
        # Fallback if no specific disease files were found: extract all
        if extracted_count == 0:
            print("[dentex_bench] Note: No specific disease pattern matched, extracting full archive...")
            z.extractall(extract_dir)
            extracted_count = len(all_members)
            
        print(f"[dentex_bench] Extracted {extracted_count} disease files. Skipped {len(all_members) - extracted_count} unused quadrant/enumeration files (saved ~10 GB disk space).")


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
        print(f"[dentex_bench] Selectively unpacking {zip_file.name} to avoid disk overflow...")
        try:
            selective_extract_dentex_zip(zip_file, zip_file.parent)
            # CRITICAL: Delete zip file immediately after extraction to free 10.5 GB disk space
            try:
                zip_file.unlink()
                print(f"[dentex_bench] Cleaned up {zip_file.name} (freed ~10.5 GB archive space).")
            except Exception:
                pass
        except Exception as e:
            print(f"[dentex_bench] Warning: Failed to extract {zip_file}: {e}")
            
    return raw_dir
