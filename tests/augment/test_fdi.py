import pytest
from dentex_bench.augment.fdi import (
    parse_fdi,
    is_upper,
    tooth_number,
    quadrant,
    mirror_fdi,
    is_valid_swap_pair,
)


def test_parse_and_properties():
    assert parse_fdi(16) == (1, 6)
    assert parse_fdi(48) == (4, 8)
    assert is_upper(16) is True
    assert is_upper(26) is True
    assert is_upper(36) is False
    assert is_upper(48) is False
    assert tooth_number(37) == 7
    assert quadrant(21) == 2


def test_mirror_fdi():
    # Upper jaw: Q1 <-> Q2
    assert mirror_fdi(16) == 26
    assert mirror_fdi(26) == 16
    assert mirror_fdi(11) == 21
    # Lower jaw: Q3 <-> Q4
    assert mirror_fdi(36) == 46
    assert mirror_fdi(46) == 36
    assert mirror_fdi(48) == 38


def test_invalid_fdi_raises():
    with pytest.raises(ValueError):
        parse_fdi(55)
    with pytest.raises(ValueError):
        parse_fdi(19)


def test_is_valid_swap_pair():
    # Direct match (no mirror)
    valid, mirror = is_valid_swap_pair(16, 16)
    assert valid is True and mirror is False

    # Symmetric mirror match
    valid, mirror = is_valid_swap_pair(16, 26, allow_mirror=True)
    assert valid is True and mirror is True

    # Lower jaw symmetric match
    valid, mirror = is_valid_swap_pair(36, 46, allow_mirror=True)
    assert valid is True and mirror is True

    # Upper to Lower jaw (forbidden INV-4)
    valid, _ = is_valid_swap_pair(16, 46, allow_mirror=True)
    assert valid is False

    # Different tooth numbers (forbidden INV-4)
    valid, _ = is_valid_swap_pair(16, 17, allow_mirror=True)
    assert valid is False
