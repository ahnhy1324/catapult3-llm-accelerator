import math

import torch

from nvfp4_fakequant import estimate_packed_bytes, fake_quantize_nvfp4


def test_storage_costs():
    rows, cols = 128256, 2560
    _, row_bits = estimate_packed_bytes(rows, cols, "row16")
    _, block_bits = estimate_packed_bytes(rows, cols, "block16x16")
    assert abs(row_bits - 4.5) < 1e-5
    assert abs(block_bits - 4.03125) < 1e-5


def test_zero_matrix():
    x = torch.zeros(17, 19)
    for layout in ("row16", "block16x16"):
        y, stats = fake_quantize_nvfp4(x, layout=layout, chunk_rows=16)
        assert torch.equal(x, y)
        assert stats.mse == 0
        assert math.isfinite(stats.cosine_similarity)


def test_shape_and_finite():
    torch.manual_seed(1)
    x = torch.randn(35, 47) * 0.03
    for layout in ("row16", "block16x16"):
        y, stats = fake_quantize_nvfp4(x, layout=layout, chunk_rows=17)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert stats.effective_bits_per_weight > 4.0
        assert stats.normalized_mse < 0.2
        assert stats.cosine_similarity > 0.9
