# tests/augment/test_rng.py
import numpy as np
from dentex_bench.augment.rng import stable_int, rng_for

def test_stable_int_deterministic():
    val1 = stable_int("offline", 123, 1)
    val2 = stable_int("offline", 123, 1)
    assert val1 == val2
    assert isinstance(val1, int)

def test_rng_for_reproducibility():
    rng1 = rng_for(42, "offline", 100, 0)
    rng2 = rng_for(42, "offline", 100, 0)
    
    seq1 = rng1.uniform(0, 1, size=10)
    seq2 = rng2.uniform(0, 1, size=10)
    np.testing.assert_array_equal(seq1, seq2)

def test_rng_for_key_independence():
    rng1 = rng_for(42, "offline", 100, 0)
    rng2 = rng_for(42, "offline", 100, 1)
    
    seq1 = rng1.uniform(0, 1, size=10)
    seq2 = rng2.uniform(0, 1, size=10)
    assert not np.array_equal(seq1, seq2)
