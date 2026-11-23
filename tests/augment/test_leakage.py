import pytest
from dentex_bench.augment.bank import ToothBank


def test_leakage_assertion_raises_when_val_id_in_bank():
    val_image_ids = {10, 11, 12}
    train_image_ids = {1, 2, 3, 10}  # 10 is leaked!

    with pytest.raises(AssertionError, match="INV-1"):
        ToothBank.assert_no_val_leakage(train_image_ids, val_image_ids)


def test_leakage_assertion_passes_when_disjoint():
    val_image_ids = {10, 11, 12}
    train_image_ids = {1, 2, 3, 4}
    # Should pass without raising
    ToothBank.assert_no_val_leakage(train_image_ids, val_image_ids)
