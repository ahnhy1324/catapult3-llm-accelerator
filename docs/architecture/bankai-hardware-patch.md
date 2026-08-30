# Bankai row-patch hardware contract

Status: fixed-point equivalence tested; public behavioral benefit not reproduced.

Evidence tags used here:

- `MEASURED_CPU`: integer unit tests executed on the host.
- `CALCULATED_FROM_CONFIG`: table sizes derived from Bonsai 1.7B geometry.
- `PROJECTED_FPGA`: implementation behavior that still requires RTL integration and fit.

## Exact patch point

The safe patch operation is:

```text
binary group dots
  -> signed group scale multiply
  -> symmetric RNE group rounding
  -> wide projection sum
  -> conditional two's-complement negate   <-- patch here
  -> residual add / nonlinear operation
  -> final quantized cast or saturation
```

For a selected output row, XORing every binary weight sign negates every exact
integer group dot. Signed scale multiplication and round-to-nearest-ties-even
are odd-symmetric, so the wide projection contribution remains exactly
negated. The CPU tests cover attention and MLP shapes, signed scales, group
rounding, two's-complement minimum values, residual placement, and output
casts (`MEASURED_CPU`).

The equivalence is not unconditional:

- Group saturation can break it because an N-bit signed range contains one
  extra negative value.
- Final saturation has the same asymmetric-minimum problem.
- Negating after a residual add also negates the residual and is wrong.
- Negating after a narrow output cast cannot represent the positive image of
  the two's-complement minimum value.
- A negative group scale does not itself break the identity, but deployed Q1
  magnitude scales are expected to be non-negative; an unexpected signed
  scale is a model-health failure.

Therefore the enable bit must control a wide conditional negate after all
group contributions have been summed, before residual addition and before any
narrow saturating cast. Any group accumulator configuration that can saturate
is incompatible with the exact patch claim.

## Patch table

The logical record is:

| Field | Width | Meaning |
|---|---:|---|
| `layer` | 5 bits | Bonsai 1.7B layer 0..27 |
| `projection` | 3 bits | q, k, v, o, gate, up, or down |
| `output_row` | 13 bits | row within the selected projection |
| `enable` | 1 bit | apply the wide negate |

A sparse record rounds to 32 bits. A dense bitmap is simpler on the inference
path: each emitted output row reads one enable bit from a bank selected by
layer and projection.

For Bonsai 1.7B, each layer has 20,480 patchable projection rows:

```text
q 2048 + k 1024 + v 1024 + o 2048
+ gate 6144 + up 6144 + down 2048 = 20,480 rows/layer
20,480 * 28 layers = 573,440 bits = 71,680 bytes
```

This is exactly 28 M20K payload blocks at 20,480 bits/block, ignoring banking
and port-shape packing (`CALCULATED_FROM_CONFIG`). A double-buffered request
profile needs 143,360 bytes or 56 raw M20K payload blocks. A sparse list needs
`4 * enabled_rows` bytes plus an expansion step; it becomes smaller than the
dense bitmap below 17,920 enabled rows.

## Request switching and II

A request-specific bitmap is loaded into the inactive bank, then a bank-select
bit changes at a token boundary. The pointer swap is one cycle. Loading costs
71,680 bytes per dense profile; sparse expansion costs at least one write per
record. Neither operation belongs on the active token pipeline
(`PROJECTED_FPGA`).

Once resident, the bitmap lookup and conditional wide negate are fully
pipelineable and do not inherently increase inference II. That statement must
still be checked in the integrated post-fit design. No claim is made here that
any public Bankai patch improves model behavior; only the arithmetic identity
and its safe hardware boundary are established.
