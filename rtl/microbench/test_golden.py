from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from golden import (  # noqa: E402
    binary_g128_dot,
    pack_trits,
    rne_shift,
    ternary_dot,
    unpack_trits_threshold,
)


def test_threshold_decoder_exhaustively_round_trips_all_243_codes():
    for code in range(243):
        trits = unpack_trits_threshold(code)
        assert pack_trits(trits) == [code]


def test_direct_and_tl5_share_the_same_random_dot_reference():
    rng = np.random.default_rng(20260831)
    for lanes in (10, 640, 672):
        activation = rng.integers(-128, 128, size=lanes)
        trits = rng.integers(-1, 2, size=lanes)
        padded = np.pad(trits, (0, (-lanes) % 5), constant_values=0)
        reconstructed = np.asarray(
            [trit for code in pack_trits(padded) for trit in unpack_trits_threshold(code)]
        )[:lanes]
        assert ternary_dot(activation, trits, output_bits=32) == ternary_dot(
            activation, reconstructed, output_bits=32
        )


def test_binary_min_max_and_rne_boundaries():
    activation = np.asarray([127] * 64 + [-128] * 64)
    signs = np.asarray([1] * 128)
    assert binary_g128_dot(
        activation, signs, [3], accumulator_bits=14, scale_fraction_bits=1, output_bits=16
    ) == (-96, False)
    assert rne_shift(5, 1) == 2
    assert rne_shift(7, 1) == 4
    assert rne_shift(-5, 1) == -2
    assert rne_shift(-7, 1) == -4


def test_binary_accumulator_saturation_boundary():
    activation = [127] * 128
    signs = [1] * 128
    value, saturated = binary_g128_dot(
        activation, signs, [511], accumulator_bits=8, scale_fraction_bits=0, output_bits=12
    )
    assert value == 2047
    assert saturated is True
