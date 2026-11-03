import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def release_cuda_memory():
    """Return cached GPU memory after every test.

    Several tests launch scripts/notebooks in a subprocess that needs the GPU; on a 4 GB card
    they hit CUDA OOM if this pytest process still holds memory cached by earlier model tests.
    """
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
