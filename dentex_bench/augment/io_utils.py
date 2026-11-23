# dentex_bench/augment/io_utils.py
import hashlib
from pathlib import Path
from typing import Union
import cv2
import numpy as np

def read_image_lossless(path: Union[str, Path]) -> np.ndarray:
    """Read image preserving bit depth and channels."""
    p_str = str(path)
    img = cv2.imread(p_str, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    if img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img

def write_image_lossless(path: Union[str, Path], img: np.ndarray) -> None:
    """Write image to disk as lossless PNG using atomic temporary write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(".tmp.png")
    
    if img.ndim == 2:
        to_save = img
    elif img.ndim == 3 and img.shape[2] == 3:
        to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        to_save = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        to_save = img
        
    success = cv2.imwrite(str(tmp_path), to_save, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    if not success:
        raise IOError(f"cv2.imwrite failed for {tmp_path}")
    tmp_path.replace(p)

def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA256 hex digest of file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
