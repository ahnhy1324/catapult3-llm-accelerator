"""Software NVFP4 fake-quantization helpers.

This module emulates NVIDIA's NVFP4 numerical format for quality studies on
hardware that does not have native NVFP4 support. It intentionally returns a
normal floating-point tensor after quantize/dequantize; it is not a fast kernel
and it does not provide inference speedups.

Supported weight layouts:

* ``row16``: one FP8 E4M3 scale per 16 consecutive values in every row.
  This is the FPGA-friendly layout considered for a tied embedding/LM head.
* ``block16x16``: one FP8 E4M3 scale per 16x16 weight block, matching the
  default 2-D weight-scaling concept used by NVIDIA Transformer Engine.

Both layouts use one tensor-wide FP32 global scale and FP4 E2M1 values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Literal

import torch

Layout = Literal["row16", "block16x16"]

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E4M3FN_MAX = 448.0
_E4M3FN_MIN_POSITIVE = 2.0**-9
_E2M1_MAX = 6.0


@dataclass(frozen=True)
class NVFP4Stats:
    layout: str
    rows: int
    cols: int
    padded_rows: int
    padded_cols: int
    numel: int
    global_amax: float
    global_scale: float
    scale_count: int
    packed_bytes: int
    effective_bits_per_weight: float
    mse: float
    normalized_mse: float
    max_abs_error: float
    cosine_similarity: float
    saturation_fraction: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _as_device(device: str | torch.device | None, fallback: torch.device) -> torch.device:
    return fallback if device is None else torch.device(device)


def _quantize_e4m3fn(values: torch.Tensor) -> torch.Tensor:
    """Round non-negative values to PyTorch's finite E4M3 FP8 grid."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "This demo needs a PyTorch build exposing torch.float8_e4m3fn "
            "(PyTorch 2.1 or newer is recommended)."
        )
    clipped = values.clamp_(min=0.0, max=_E4M3FN_MAX)
    return clipped.to(torch.float8_e4m3fn).to(torch.float32)


def _quantize_e2m1(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round normalized values to FP4 E2M1 and return (q, saturated_mask)."""
    abs_values = values.abs()
    thresholds = torch.tensor(_E2M1_THRESHOLDS, device=values.device, dtype=torch.float32)
    levels = torch.tensor(_E2M1_LEVELS, device=values.device, dtype=torch.float32)
    indices = torch.bucketize(abs_values, thresholds, right=False)
    magnitudes = levels[indices]
    signs = torch.where(values < 0, -1.0, 1.0)
    quantized = magnitudes * signs
    saturated = abs_values > _E2M1_MAX
    return quantized, saturated


def _streaming_amax(weight: torch.Tensor, chunk_rows: int, work_device: torch.device) -> float:
    maximum = 0.0
    with torch.inference_mode():
        for start in range(0, weight.shape[0], chunk_rows):
            chunk = weight[start : start + chunk_rows].detach().to(
                device=work_device, dtype=torch.float32, non_blocking=True
            )
            if chunk.numel():
                maximum = max(maximum, float(chunk.abs().amax().item()))
    return maximum


def _storage_shape(rows: int, cols: int, layout: Layout) -> tuple[int, int, int]:
    padded_cols = ceil(cols / 16) * 16
    if layout == "row16":
        padded_rows = rows
        scale_count = rows * (padded_cols // 16)
    elif layout == "block16x16":
        padded_rows = ceil(rows / 16) * 16
        scale_count = (padded_rows // 16) * (padded_cols // 16)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return padded_rows, padded_cols, scale_count


def estimate_packed_bytes(rows: int, cols: int, layout: Layout) -> tuple[int, float]:
    """Return packed bytes and effective bits/original-weight for a layout."""
    padded_rows, padded_cols, scale_count = _storage_shape(rows, cols, layout)
    total_bits = padded_rows * padded_cols * 4 + scale_count * 8 + 32
    packed_bytes = ceil(total_bits / 8)
    bits_per_weight = total_bits / (rows * cols)
    return packed_bytes, bits_per_weight


def fake_quantize_nvfp4(
    weight: torch.Tensor,
    *,
    layout: Layout = "row16",
    chunk_rows: int = 256,
    work_device: str | torch.device | None = None,
    output_device: str | torch.device = "cpu",
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, NVFP4Stats]:
    """Fake-quantize a 2-D weight tensor to NVFP4 and dequantize it.

    The implementation is row-chunked so the 128256 x 2560 BitNet tied
    embedding does not need a second full FP32 copy in memory.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight matrix, got shape {tuple(weight.shape)}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    rows, cols = map(int, weight.shape)
    if layout == "block16x16" and chunk_rows % 16:
        chunk_rows = ceil(chunk_rows / 16) * 16

    source_device = weight.device
    work = _as_device(work_device, source_device)
    out_device = torch.device(output_device)
    out_dtype = output_dtype or weight.dtype

    global_amax = _streaming_amax(weight, chunk_rows, work)
    global_scale = 1.0 if global_amax == 0.0 else global_amax / (_E4M3FN_MAX * _E2M1_MAX)

    padded_rows, padded_cols, scale_count = _storage_shape(rows, cols, layout)
    output = torch.empty((rows, cols), dtype=out_dtype, device=out_device)

    sum_sq_error = 0.0
    sum_sq_source = 0.0
    dot_product = 0.0
    sum_sq_output = 0.0
    max_abs_error = 0.0
    saturated_values = 0
    value_count = rows * cols

    with torch.inference_mode():
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            x = weight[start:end].detach().to(device=work, dtype=torch.float32, non_blocking=True)
            original_rows = end - start

            if cols != padded_cols:
                x = torch.nn.functional.pad(x, (0, padded_cols - cols))

            if layout == "row16":
                blocks = x.reshape(original_rows, padded_cols // 16, 16)
                block_amax = blocks.abs().amax(dim=-1, keepdim=True)
                raw_block_scale = block_amax / (_E2M1_MAX * global_scale)
                block_scale = _quantize_e4m3fn(raw_block_scale)
                block_scale = torch.where(
                    (block_amax > 0) & (block_scale == 0),
                    torch.full_like(block_scale, _E4M3FN_MIN_POSITIVE),
                    block_scale,
                )
                effective_scale = block_scale * global_scale
                safe_scale = torch.where(block_amax == 0, torch.ones_like(effective_scale), effective_scale)
                normalized = blocks / safe_scale
                q, saturated = _quantize_e2m1(normalized)
                dequant = (q * effective_scale).reshape(original_rows, padded_cols)

            elif layout == "block16x16":
                padded_chunk_rows = ceil(original_rows / 16) * 16
                if original_rows != padded_chunk_rows:
                    x = torch.nn.functional.pad(x, (0, 0, 0, padded_chunk_rows - original_rows))
                blocks = x.reshape(padded_chunk_rows // 16, 16, padded_cols // 16, 16)
                block_amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
                raw_block_scale = block_amax / (_E2M1_MAX * global_scale)
                block_scale = _quantize_e4m3fn(raw_block_scale)
                block_scale = torch.where(
                    (block_amax > 0) & (block_scale == 0),
                    torch.full_like(block_scale, _E4M3FN_MIN_POSITIVE),
                    block_scale,
                )
                effective_scale = block_scale * global_scale
                safe_scale = torch.where(block_amax == 0, torch.ones_like(effective_scale), effective_scale)
                normalized = blocks / safe_scale
                q, saturated = _quantize_e2m1(normalized)
                dequant = (q * effective_scale).reshape(padded_chunk_rows, padded_cols)[:original_rows]
            else:
                raise ValueError(f"Unsupported layout: {layout}")

            dequant = dequant[:, :cols]
            source = x[:original_rows, :cols]
            error = dequant - source

            sum_sq_error += float(error.square().sum().item())
            sum_sq_source += float(source.square().sum().item())
            dot_product += float((source * dequant).sum().item())
            sum_sq_output += float(dequant.square().sum().item())
            max_abs_error = max(max_abs_error, float(error.abs().amax().item()))
            if layout == "row16":
                saturated_values += int(saturated.reshape(original_rows, padded_cols)[:, :cols].sum().item())
            else:
                saturated_values += int(
                    saturated.reshape(-1, padded_cols)[:original_rows, :cols].sum().item()
                )

            output[start:end].copy_(dequant.to(device=out_device, dtype=out_dtype), non_blocking=False)

    mse = sum_sq_error / max(value_count, 1)
    normalized_mse = sum_sq_error / max(sum_sq_source, 1e-30)
    cosine_similarity = dot_product / max((sum_sq_source * sum_sq_output) ** 0.5, 1e-30)
    packed_bytes, effective_bits = estimate_packed_bytes(rows, cols, layout)

    stats = NVFP4Stats(
        layout=layout,
        rows=rows,
        cols=cols,
        padded_rows=padded_rows,
        padded_cols=padded_cols,
        numel=value_count,
        global_amax=global_amax,
        global_scale=global_scale,
        scale_count=scale_count,
        packed_bytes=packed_bytes,
        effective_bits_per_weight=effective_bits,
        mse=mse,
        normalized_mse=normalized_mse,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
        saturation_fraction=saturated_values / max(value_count, 1),
    )
    return output, stats
