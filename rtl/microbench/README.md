# Model-selection RTL microbench

These kernels answer only the datapath questions needed for the first
Catapult3 model choice. They are not a complete accelerator.

- `bonsai_binary_g128_dot.sv`: packed binary signs, 128-value A8 groups,
  fixed-point group scale multiply, RNE, and wide accumulation.
- `bitnet_direct_ternary.sv`: threshold-based 5-trits/byte decode followed by
  `-a`, zero, or `+a` and a staged reduction.
- `bitnet_tl5.sv`: 243-entry table per five activation lanes, a 244-cycle
  single-bank table build, synchronous lookup, and staged reduction.

All three interfaces accept a bubble-free valid stream at II=1 after their
setup/latency. TL5's steady-state II does not make its table build free; the
e2e roof model charges the build once per activation tile.

Run the NumPy golden tests:

```bash
python -m pytest rtl/microbench/test_golden.py rtl/microbench/test_quartus_parser.py -q
```

Run the bit-exact testbench with Icarus:

```bash
python rtl/microbench/run_iverilog.py
```

Run the exact-part Quartus representative sweep on a Windows host with
Quartus Pro 25.1:

```powershell
python rtl\microbench\run_quartus_sweep.py `
  --quartus E:\Altera\quartus\bin64\quartus_sh.exe `
  --build-root C:\quartus-v2 `
  --output results\model_selection_v2\rtl_sweep.json `
  --only representative
```

The flow targets the board marking `10AXF40GAE`, which Quartus resolves to
Arria 10 GX `10AX115_JZ`, speed grade 2. Quartus databases and reports stay
outside Git; the compact parsed JSON is checked in.
