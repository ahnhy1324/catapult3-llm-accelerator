"""Reference fake quantizers for the Catapult3 model-selection experiment.

The routines in this file are deliberately explicit.  They emulate numerical
contracts and collect evidence; they are not optimized inference kernels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

import torch


@dataclass
class QuantizationStats:
    bits: int
    qmin: int
    qmax: int
    granularity: str
    group_size: int | None
    value_count: int
    zero_scale_count: int
    clipped_count: int
    endpoint_count: int
    saturation_rate: float
    endpoint_rate: float
    input_abs_max: float
    input_abs_p50: float
    input_abs_p90: float
    input_abs_p99: float
    input_abs_p999: float
    scale_min: float
    scale_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BinaryLinearStats:
    activation_bits: int
    activation_group_size: int | None
    weight_group_size: int
    accumulator_bits: int
    accumulator_saturation_count: int
    group_count: int
    fixed_scale_fraction_bits: int | None
    group_output_fraction_bits: int | None
    weight_group_max_relative_spread: float
    activation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    flat = values.detach().float().abs().reshape(-1)
    # Quantiles are diagnostic only.  A deterministic strided sample keeps the
    # hook bounded for long PPL sequences and does not alter quantized values.
    if flat.numel() > 65_536:
        stride = ceil(flat.numel() / 65_536)
        flat = flat[::stride][:65_536]
    return float(torch.quantile(flat, q).item())


def signed_symmetric_range(bits: int) -> tuple[int, int]:
    """Return the reserved-min symmetric signed code range.

    Three bits therefore map to ``[-3, 3]``, four to ``[-7, 7]`` and eight to
    ``[-127, 127]``.  The most-negative two's-complement code is unused.
    """
    if bits < 2 or bits > 16:
        raise ValueError(f"bits must be in [2, 16], got {bits}")
    qmax = (1 << (bits - 1)) - 1
    return -qmax, qmax


def _reshape_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    cols = int(x.shape[-1])
    padded = ceil(cols / group_size) * group_size
    if padded != cols:
        x = torch.nn.functional.pad(x, (0, padded - cols))
    return x.reshape(*x.shape[:-1], padded // group_size, group_size), cols


def symmetric_fake_quantize(
    x: torch.Tensor,
    bits: int,
    *,
    group_size: int | None = None,
    collect_percentiles: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, QuantizationStats]:
    """Per-token or last-dimension grouped symmetric quantize/dequantize.

    Scaling is ``amax / qmax``.  Codes use round-to-nearest, ties-to-even
    (``torch.round``), then explicit clipping to the reserved-min range.
    Zero groups use scale one and code zero.
    """
    if not torch.is_floating_point(x):
        raise TypeError("symmetric_fake_quantize expects a floating tensor")
    if not torch.isfinite(x).all():
        raise ValueError("input contains NaN or Inf")
    qmin, qmax = signed_symmetric_range(bits)
    original_shape = x.shape
    work = x.detach().float()

    if group_size is None:
        grouped = work.unsqueeze(-2)
        original_cols = int(work.shape[-1])
        granularity = "per_token"
    else:
        grouped, original_cols = _reshape_groups(work, group_size)
        granularity = f"per_{group_size}_group"

    amax = grouped.abs().amax(dim=-1, keepdim=True)
    zero = amax == 0
    scale = torch.where(zero, torch.ones_like(amax), amax / qmax)
    unrounded = grouped / scale
    rounded = torch.round(unrounded)
    # Saturation is a representability event after the frozen RNE step. Tiny
    # floating division noise above an exact endpoint is not counted as a
    # clipped value when it still rounds to that endpoint.
    clipped_mask = (rounded < qmin) | (rounded > qmax)
    codes_grouped = rounded.clamp(qmin, qmax).to(torch.int16)
    dequant_grouped = codes_grouped.float() * scale

    if group_size is None:
        codes = codes_grouped.squeeze(-2).reshape(original_shape)
        dequant = dequant_grouped.squeeze(-2).reshape(original_shape)
        scales = scale.squeeze(-1).squeeze(-1)
    else:
        flat_codes = codes_grouped.reshape(*work.shape[:-1], -1)[..., :original_cols]
        flat_dequant = dequant_grouped.reshape(*work.shape[:-1], -1)[..., :original_cols]
        codes = flat_codes.reshape(original_shape)
        dequant = flat_dequant.reshape(original_shape)
        scales = scale.squeeze(-1)

    endpoint_count = int((codes.abs() == qmax).sum().item())
    value_count = int(x.numel())
    scale_values = scale[~zero]
    stats = QuantizationStats(
        bits=bits,
        qmin=qmin,
        qmax=qmax,
        granularity=granularity,
        group_size=group_size,
        value_count=value_count,
        zero_scale_count=int(zero.sum().item()),
        clipped_count=int(clipped_mask.sum().item()),
        endpoint_count=endpoint_count,
        saturation_rate=float(clipped_mask.sum().item()) / max(value_count, 1),
        endpoint_rate=endpoint_count / max(value_count, 1),
        input_abs_max=float(work.abs().amax().item()) if value_count else 0.0,
        input_abs_p50=_percentile(work, 0.5) if collect_percentiles else 0.0,
        input_abs_p90=_percentile(work, 0.9) if collect_percentiles else 0.0,
        input_abs_p99=_percentile(work, 0.99) if collect_percentiles else 0.0,
        input_abs_p999=_percentile(work, 0.999) if collect_percentiles else 0.0,
        scale_min=float(scale_values.amin().item()) if scale_values.numel() else 0.0,
        scale_max=float(scale_values.amax().item()) if scale_values.numel() else 0.0,
    )
    return dequant.to(x.dtype), codes, scales, stats


def quantize_fixed_scale(
    scale: torch.Tensor,
    *,
    fraction_bits: int,
    total_bits: int = 24,
) -> torch.Tensor:
    """Unsigned fixed-point scale approximation with ties-to-even rounding."""
    if fraction_bits < 0 or total_bits <= fraction_bits:
        raise ValueError("invalid fixed-point width")
    if not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("scale must be finite and non-negative")
    maximum = (1 << total_bits) - 1
    code = torch.round(scale.float() * (1 << fraction_bits)).clamp(0, maximum)
    return code / float(1 << fraction_bits)


def saturate_signed_accumulator(accumulator: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    if bits < 2 or bits > 63:
        raise ValueError("accumulator bits must be in [2, 63]")
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    saturated = (accumulator < lower) | (accumulator > upper)
    return accumulator.clamp(lower, upper), int(saturated.sum().item())


def binary_group_linear_reference(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation_bits: int,
    activation_group_size: int | None,
    weight_group_size: int = 128,
    accumulator_bits: int = 32,
    fixed_scale_fraction_bits: int | None = None,
    group_output_fraction_bits: int | None = None,
) -> tuple[torch.Tensor, BinaryLinearStats]:
    """Reference a biasless Q1_0-g128 linear boundary.

    ``weight`` is the unpacked FP tensor.  Each group must contain a common
    magnitude and signs.  The integer dot is accumulated per weight group,
    optionally saturated, then multiplied by activation and weight scales.
    """
    if activation.shape[-1] != weight.shape[-1] or weight.ndim != 2:
        raise ValueError("activation/weight shape mismatch")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains NaN or Inf")

    in_features = int(weight.shape[-1])
    padded = ceil(in_features / weight_group_size) * weight_group_size
    x = activation.detach().float()
    w = weight.detach().float()
    if padded != in_features:
        x = torch.nn.functional.pad(x, (0, padded - in_features))
        w = torch.nn.functional.pad(w, (0, padded - in_features))

    x_dequant, x_codes, _, x_stats = symmetric_fake_quantize(
        x, activation_bits, group_size=activation_group_size
    )
    del x_dequant
    groups = padded // weight_group_size
    w_grouped = w.reshape(w.shape[0], groups, weight_group_size)
    valid = torch.ones_like(w_grouped, dtype=torch.bool)
    if padded != in_features:
        valid[:, -1, in_features % weight_group_size :] = False
    abs_w = w_grouped.abs()
    valid_count = valid.sum(dim=-1).clamp_min(1)
    weight_scale = (abs_w * valid).sum(dim=-1) / valid_count
    spread = torch.where(valid, (abs_w - weight_scale.unsqueeze(-1)).abs(), torch.zeros_like(abs_w))
    relative_spread = spread / weight_scale.unsqueeze(-1).clamp_min(1e-30)
    max_relative_spread = float(relative_spread.amax().item())
    signs = torch.where(w_grouped < 0, -1, 1).to(torch.int64)
    signs = torch.where(valid, signs, torch.zeros_like(signs))

    prefix_shape = activation.shape[:-1]
    flat_codes = x_codes.reshape(-1, groups, weight_group_size).to(torch.int64)
    if activation_group_size is None:
        _, _, token_scale, _ = symmetric_fake_quantize(x, activation_bits, group_size=None)
        x_scale = token_scale.reshape(-1, 1).expand(-1, groups)
    elif activation_group_size == weight_group_size:
        _, _, group_scale, _ = symmetric_fake_quantize(
            x, activation_bits, group_size=weight_group_size
        )
        x_scale = group_scale.reshape(-1, groups)
    else:
        raise ValueError("reference supports per-token or weight-group-aligned activation scaling")

    output = torch.zeros((flat_codes.shape[0], weight.shape[0]), dtype=torch.float64)
    saturation_count = 0
    for group in range(groups):
        acc = flat_codes[:, group] @ signs[:, group].transpose(0, 1)
        acc, count = saturate_signed_accumulator(acc, accumulator_bits)
        saturation_count += count
        multiplier = x_scale[:, group].unsqueeze(-1) * weight_scale[:, group].unsqueeze(0)
        if fixed_scale_fraction_bits is not None:
            multiplier = quantize_fixed_scale(
                multiplier, fraction_bits=fixed_scale_fraction_bits
            )
        contribution = acc.double() * multiplier.double()
        if group_output_fraction_bits is not None:
            step = float(1 << group_output_fraction_bits)
            contribution = torch.round(contribution * step) / step
        output += contribution

    stats = BinaryLinearStats(
        activation_bits=activation_bits,
        activation_group_size=activation_group_size,
        weight_group_size=weight_group_size,
        accumulator_bits=accumulator_bits,
        accumulator_saturation_count=saturation_count,
        group_count=groups,
        fixed_scale_fraction_bits=fixed_scale_fraction_bits,
        group_output_fraction_bits=group_output_fraction_bits,
        weight_group_max_relative_spread=max_relative_spread,
        activation=x_stats.to_dict(),
    )
    return output.reshape(*prefix_shape, weight.shape[0]).to(activation.dtype), stats


def apply_bankai_row_xor(sign_weight: torch.Tensor, row_flip: torch.Tensor) -> torch.Tensor:
    """Flip every binary sign in selected output rows."""
    if sign_weight.ndim != 2 or row_flip.ndim != 1 or row_flip.numel() != sign_weight.shape[0]:
        raise ValueError("row mask shape mismatch")
    if not torch.all((sign_weight == -1) | (sign_weight == 1)):
        raise ValueError("sign_weight must contain only -1/+1")
    factors = torch.where(row_flip.bool(), -1, 1).to(sign_weight.dtype).unsqueeze(-1)
    return sign_weight * factors


def binary_integer_accumulator(activation_code: torch.Tensor, sign_weight: torch.Tensor) -> torch.Tensor:
    """Exact wide integer accumulator used by the Bankai equivalence test."""
    if activation_code.shape[-1] != sign_weight.shape[-1]:
        raise ValueError("activation/weight shape mismatch")
    return activation_code.to(torch.int64) @ sign_weight.to(torch.int64).transpose(0, 1)


try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - import error is reported by the runners
    DynamicCache = object  # type: ignore[misc,assignment]


class QuantizedDynamicCache(DynamicCache):  # type: ignore[misc]
    """Dynamic cache that stores each new post-RoPE K/V entry quantized once."""

    def __init__(self, bits: int, *, group_size: int | None = None) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.update_count = 0
        self.value_count = 0
        self.clipped_count = 0
        self.endpoint_count = 0
        self.max_abs = 0.0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # type: ignore[override]
        qk, _, _, sk = symmetric_fake_quantize(
            key_states, self.bits, group_size=self.group_size, collect_percentiles=False
        )
        qv, _, _, sv = symmetric_fake_quantize(
            value_states, self.bits, group_size=self.group_size, collect_percentiles=False
        )
        self.update_count += 1
        self.value_count += sk.value_count + sv.value_count
        self.clipped_count += sk.clipped_count + sv.clipped_count
        self.endpoint_count += sk.endpoint_count + sv.endpoint_count
        self.max_abs = max(self.max_abs, sk.input_abs_max, sv.input_abs_max)
        return super().update(qk, qv, layer_idx, cache_kwargs)

    def stats_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "code_mapping": f"reserved_min_symmetric_{signed_symmetric_range(self.bits)}",
            "scale_granularity": "per_token_per_head"
            if self.group_size is None
            else f"per_token_per_head_group{self.group_size}",
            "rounding": "RNE_TIES_EVEN",
            "clipping": "EXPLICIT_TO_SYMMETRIC_RANGE",
            "update_count": self.update_count,
            "value_count": self.value_count,
            "clipped_count": self.clipped_count,
            "saturation_rate": self.clipped_count / max(self.value_count, 1),
            "endpoint_rate": self.endpoint_count / max(self.value_count, 1),
            "input_abs_max": self.max_abs,
        }
