"""NumPy/Python bit-exact references for the RTL microbench."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def saturate(value: int, bits: int) -> tuple[int, bool]:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    return min(max(value, minimum), maximum), value < minimum or value > maximum


def rne_shift(value: int, fraction_bits: int) -> int:
    if fraction_bits == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << fraction_bits)
    half = 1 << (fraction_bits - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return -quotient if value < 0 else quotient


def binary_g128_dot(
    activation: Iterable[int],
    signs: Iterable[int],
    scales: Iterable[int],
    *,
    group_size: int = 128,
    accumulator_bits: int = 24,
    scale_fraction_bits: int = 8,
    output_bits: int = 32,
) -> tuple[int, bool]:
    x = np.asarray(list(activation), dtype=np.int64)
    w = np.asarray(list(signs), dtype=np.int64)
    s = list(scales)
    if x.shape != w.shape or x.size % group_size:
        raise ValueError("shape/group mismatch")
    result = 0
    saturated = False
    for group in range(x.size // group_size):
        dot = int(np.dot(x[group * group_size : (group + 1) * group_size], w[group * group_size : (group + 1) * group_size]))
        dot, hit = saturate(dot, accumulator_bits)
        saturated |= hit
        result += rne_shift(dot * int(s[group]), scale_fraction_bits)
    result, hit = saturate(result, output_bits)
    return result, saturated or hit


def pack_trits(trits: Iterable[int]) -> list[int]:
    values = list(trits)
    if len(values) % 5 or any(value not in (-1, 0, 1) for value in values):
        raise ValueError("trits must be -1/0/+1 and a multiple of five")
    packed = []
    for start in range(0, len(values), 5):
        byte = 0
        multiplier = 1
        for trit in values[start : start + 5]:
            byte += (int(trit) + 1) * multiplier
            multiplier *= 3
        packed.append(byte)
    return packed


def unpack_trits_threshold(code: int) -> list[int]:
    """Mirror the divider-free 81/27/9/3 RTL decoder."""
    if code < 0 or code > 242:
        raise ValueError("packed 5-trit code must be in [0, 242]")
    digit4 = 2 if code >= 162 else (1 if code >= 81 else 0)
    remainder = code - digit4 * 81
    digit3 = 2 if remainder >= 54 else (1 if remainder >= 27 else 0)
    remainder -= digit3 * 27
    digit2 = 2 if remainder >= 18 else (1 if remainder >= 9 else 0)
    remainder -= digit2 * 9
    digit1 = 2 if remainder >= 6 else (1 if remainder >= 3 else 0)
    digit0 = remainder - digit1 * 3
    return [digit0 - 1, digit1 - 1, digit2 - 1, digit3 - 1, digit4 - 1]


def ternary_dot(activation: Iterable[int], trits: Iterable[int], *, output_bits: int = 24) -> tuple[int, bool]:
    value = int(np.dot(np.asarray(list(activation), dtype=np.int64), np.asarray(list(trits), dtype=np.int64)))
    return saturate(value, output_bits)
