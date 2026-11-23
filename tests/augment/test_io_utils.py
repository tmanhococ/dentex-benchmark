# tests/augment/test_io_utils.py
import numpy as np
from pathlib import Path
from dentex_bench.augment.io_utils import read_image_lossless, write_image_lossless, sha256_file

def test_lossless_io_roundtrip(tmp_path):
    img = (np.random.rand(100, 200, 3) * 255).astype(np.uint8)
    save_path = tmp_path / "test_img.png"
    write_image_lossless(save_path, img)
    
    assert save_path.exists()
    loaded = read_image_lossless(save_path)
    np.testing.assert_array_equal(img, loaded)

def test_sha256_file_consistency(tmp_path):
    p = tmp_path / "test.txt"
    p.write_bytes(b"dentex_test_data")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64
