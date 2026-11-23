from typing import Tuple

MIRROR_QUADRANT = {1: 2, 2: 1, 3: 4, 4: 3}


def parse_fdi(fdi: int) -> Tuple[int, int]:
    """Parse two-digit FDI integer into (quadrant, tooth_number)."""
    fdi_val = int(fdi)
    q = fdi_val // 10
    t = fdi_val % 10
    if q not in (1, 2, 3, 4) or t not in range(1, 9):
        raise ValueError(
            f"Invalid FDI tooth number: {fdi}. Expected quadrant 1..4 and tooth 1..8."
        )
    return q, t


def quadrant(fdi: int) -> int:
    return parse_fdi(fdi)[0]


def tooth_number(fdi: int) -> int:
    return parse_fdi(fdi)[1]


def is_upper(fdi: int) -> bool:
    """Returns True if upper jaw (quadrant 1 or 2), False for lower jaw (3 or 4)."""
    return quadrant(fdi) in (1, 2)


def mirror_fdi(fdi: int) -> int:
    """Mirror FDI number horizontally across dental arch (Q1<->Q2, Q3<->Q4)."""
    q, t = parse_fdi(fdi)
    mirrored_q = MIRROR_QUADRANT[q]
    return mirrored_q * 10 + t


def is_valid_swap_pair(
    src_fdi: int, dst_fdi: int, allow_mirror: bool = True
) -> Tuple[bool, bool]:
    """Validate if src_fdi can be placed into dst_fdi according to INV-4:

    Must share the same tooth number (1..8) and jaw (upper/lower).
    Returns: (is_valid, requires_horizontal_mirror)
    """
    src_q, src_t = parse_fdi(src_fdi)
    dst_q, dst_t = parse_fdi(dst_fdi)

    if src_t != dst_t:
        return False, False
    if is_upper(src_fdi) != is_upper(dst_fdi):
        return False, False

    if src_q == dst_q:
        return True, False
    elif allow_mirror and mirror_fdi(src_fdi) == dst_fdi:
        return True, True

    return False, False
