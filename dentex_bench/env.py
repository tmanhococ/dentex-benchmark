import os
import sys
import subprocess
import importlib

def detect_platform() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working"):
        return "kaggle"
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    return "local"

def get_device_info() -> dict:
    device = "cpu"
    name = "CPU"
    vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    
    is_low_vram = (device == "cpu") or (vram_gb > 0 and vram_gb < 6.0)
    return {
        "device": device,
        "name": name,
        "vram_gb": round(vram_gb, 2),
        "is_low_vram": is_low_vram
    }

def ensure_dependencies(extra_packages: list = None) -> bool:
    core_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("pycocotools", "pycocotools"),
        ("transformers", "transformers"),
        ("albumentations", "albumentations"),
        ("huggingface_hub", "huggingface_hub"),
        ("cv2", "opencv-python"),
        ("pandas", "pandas"),
        ("tabulate", "tabulate"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib")
    ]
    missing = []
    for mod, pkg in core_packages:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
            
    if extra_packages:
        for pkg in extra_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)
                
    if missing:
        print(f"[dentex_bench] Installing missing dependencies: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    return True
