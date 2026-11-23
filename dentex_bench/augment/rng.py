# dentex_bench/augment/rng.py
import hashlib
import numpy as np

def stable_int(*parts) -> int:
    """Generate a deterministic 64-bit unsigned integer from arbitrary string/int keys."""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    h = hashlib.blake2b(raw, digest_size=8)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)

def rng_for(global_seed: int, *key) -> np.random.Generator:
    """Instantiate an isolated, reproducible PCG64 Generator keyed by global_seed and key parts."""
    seed_int = stable_int(*key)
    # SeedSequence accepts integer or sequence of integers
    ss = np.random.SeedSequence([global_seed & 0xFFFFFFFF, seed_int & 0xFFFFFFFF])
    return np.random.Generator(np.random.PCG64(ss))
