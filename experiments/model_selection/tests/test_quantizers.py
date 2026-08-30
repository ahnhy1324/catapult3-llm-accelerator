from __future__ import annotations

import torch

from quantizers import (
    QuantizedDynamicCache,
    apply_bankai_row_xor,
    binary_group_linear_reference,
    binary_integer_accumulator,
    quantize_fixed_scale,
    saturate_signed_accumulator,
    signed_symmetric_range,
    symmetric_fake_quantize,
)


def test_reserved_min_ranges():
    assert signed_symmetric_range(3) == (-3, 3)
    assert signed_symmetric_range(4) == (-7, 7)
    assert signed_symmetric_range(8) == (-127, 127)


def test_per_token_rne_and_zero_contract():
    x = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    dequant, codes, scales, stats = symmetric_fake_quantize(x, 3)
    assert codes.tolist()[0] == [-3, -2, 0, 2, 3]
    assert codes.tolist()[1] == [0, 0, 0, 0, 0]
    assert torch.equal(dequant[1], x[1])
    assert scales[1].item() == 1.0
    assert stats.zero_scale_count == 1
    assert stats.clipped_count == 0


def test_grouped_quantization_is_independent():
    x = torch.tensor([[1.0, -1.0, 10.0, -10.0]])
    dequant, codes, scales, stats = symmetric_fake_quantize(x, 4, group_size=2)
    assert codes.tolist() == [[7, -7, 7, -7]]
    assert scales.shape == (1, 2)
    assert torch.allclose(dequant, x)
    assert stats.granularity == "per_2_group"


def test_quantized_cache_never_requantizes_stored_prefix():
    cache = QuantizedDynamicCache(4)
    first_k = torch.tensor([[[[0.10, -0.25, 0.75, -1.00]]]])
    first_v = torch.tensor([[[[-0.20, 0.30, -0.80, 1.10]]]])
    stored_k, stored_v = cache.update(first_k, first_v, 0)
    frozen_k = stored_k.clone()
    frozen_v = stored_v.clone()

    next_k = torch.tensor([[[[3.0, -2.0, 1.0, -0.5]]]])
    next_v = torch.tensor([[[[-4.0, 2.5, 0.5, 1.5]]]])
    extended_k, extended_v = cache.update(next_k, next_v, 0)

    assert torch.equal(extended_k[..., :1, :], frozen_k)
    assert torch.equal(extended_v[..., :1, :], frozen_v)
    assert cache.update_count == 2


def test_accumulator_saturation():
    value = torch.tensor([-600, -512, -511, 511, 512, 900], dtype=torch.int64)
    saturated, count = saturate_signed_accumulator(value, 10)
    assert saturated.tolist() == [-512, -512, -511, 511, 511, 511]
    assert count == 3


def test_fixed_scale_ties_to_even():
    scale = torch.tensor([0.5 / 16, 1.5 / 16, 2.5 / 16])
    quantized = quantize_fixed_scale(scale, fraction_bits=4, total_bits=8)
    assert quantized.tolist() == [0.0, 0.125, 0.125]


def test_binary_group_linear_reference_exact_q1():
    activation = torch.tensor([[1.0, -0.5, 0.25, -0.25]])
    signs = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]])
    weight = signs * torch.tensor([[0.5], [0.25]])
    output, stats = binary_group_linear_reference(
        activation,
        weight,
        activation_bits=8,
        activation_group_size=None,
        weight_group_size=4,
        accumulator_bits=20,
    )
    _, codes, scales, _ = symmetric_fake_quantize(activation, 8)
    expected_acc = codes.to(torch.int64) @ signs.to(torch.int64).T
    expected = expected_acc.float() * scales.reshape(-1, 1) * torch.tensor([[0.5, 0.25]])
    assert torch.allclose(output, expected, atol=1e-6, rtol=0)
    assert stats.accumulator_saturation_count == 0
    assert stats.weight_group_max_relative_spread == 0.0


def _bankai_case(rows: int, cols: int, tokens: int) -> None:
    generator = torch.Generator().manual_seed(rows * 1000 + cols)
    activation = torch.randint(-127, 128, (tokens, cols), generator=generator, dtype=torch.int16)
    sign_weight = torch.where(
        torch.randint(0, 2, (rows, cols), generator=generator, dtype=torch.int8) == 0,
        -torch.ones((), dtype=torch.int8),
        torch.ones((), dtype=torch.int8),
    )
    row_flip = torch.arange(rows) % 3 == 0
    baseline = binary_integer_accumulator(activation, sign_weight)
    patched = binary_integer_accumulator(activation, apply_bankai_row_xor(sign_weight, row_flip))
    expected = baseline.clone()
    expected[:, row_flip] = -expected[:, row_flip]
    assert torch.equal(patched, expected)


def test_bankai_attention_projection_bit_exact():
    _bankai_case(rows=32, cols=128, tokens=7)


def test_bankai_mlp_projection_bit_exact():
    _bankai_case(rows=96, cols=256, tokens=5)
