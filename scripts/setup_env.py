import sys
from pathlib import Path

# Add project root to sys.path so scripts can be invoked from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.env import detect_platform, get_device_info, ensure_dependencies
from dentex_bench.config import BenchmarkConfig

def main():
    print("=== DENTEX Benchmark Environment Setup ===")
    platform = detect_platform()
    dev_info = get_device_info()
    print(f"Platform: {platform}")
    print(f"Device: {dev_info['device']} ({dev_info['name']}), VRAM: {dev_info['vram_gb']} GB")
    print(f"Low VRAM Mode: {dev_info['is_low_vram']}")
    
    print("\nChecking dependencies...")
    ensure_dependencies()
    print("All core dependencies verified.")
    
    cfg = BenchmarkConfig.from_auto_env()
    print(f"\nAuto-tuned defaults: batch_size={cfg.batch_size}, grad_accum={cfg.gradient_accumulation_steps}, img_size={cfg.img_size}")

if __name__ == "__main__":
    main()
