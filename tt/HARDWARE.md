# Hardware runbook - pull this repo onto a Tenstorrent machine and run

This document is the pull-and-run path. Device execution requires `tt-metal`.
Wormhole bring-up ran at a pinned `tt-metal` SHA, with numerical failures still
open. The one-shot script checks that SHA before building. Treat the device
records as diagnostics, not a completed paper experiment.
The tile API and vendor derivative were inspected at the Wormhole pin
`f9524a5f1b75180f00ce7413c2fc1c5cca1f29ee`; the new kernel still needs an
on-board compile and run.
The first on-board results, open issues and the porting notes for pin `f9524a5f`
are in [`research/device_bringup_wormhole.md`](../research/device_bringup_wormhole.md).
Those measurements are diagnostic and excluded from paper performance claims.

## Next Wormhole exploration

After pulling branch `wh4`, use the same `TT_METAL_HOME` and
`BW_SYN_ARCH=wormhole` environment as below, with `torch` and `ttnn` optional
for this mode:

```bash
git fetch origin
git switch wh4
git pull --ff-only
./tt/scripts/run_on_device.sh
```

The default now runs the fused polynomial/exponent candidate, not the old ten-case
critical harness. It checks 6,774 constant tiles: every positive BF16 `x` encoding
from 4 through 88.5, six fixed gradients, six signed gradients around the rounded
normal-output boundary, and selected negative `x` values. Only normal reference
outputs are in scope. The host reference classifies 4,636 cases as in scope.

Five kernels share the same inputs and dataflow: fused candidate, previous split4,
materialized helper, vendor tanh derivative times gradient, and a fused vendor
tail times gradient. The last comparator uses the same SFPU traversal and packing
as the candidate. Every comparator's correctness is recorded; a faster comparator
that fails the contract is not an equally accurate baseline.

After a passing candidate sweep, the harness times 128 recorded inputs (five
warmups, 20 measurements), checks candidate outputs on every timed launch, and
runs BF16 transport/compute probes. Each tile repeats one pair across 1,024 lanes;
mixed-lane tensors and end-to-end workloads remain future validation. Zones
include buffer waits and packing; these are single-core development diagnostics.

All evidence, including build failures, is saved in an unignored
`results/runs/wormhole_<source>_<UTC>_d3/` directory: provenance, build log,
correctness CSV/log, raw profiler, cycle summary, and format probes. Commit the
whole packet after the run:

```bash
git add results/runs
git commit -m "Wormhole fused tail run"
git push origin wh4
```

The script requires committed tracked source changes. A failing numerical sweep
still writes correctness and format results but withholds timing. A JIT/build
failure preserves the log and exit status. The new SFPI path has been reviewed
against the pinned sources but has not been compiled on the board.

Optional ablations (each writes a separate evidence directory):

```bash
BW_SYN_TAIL_DEGREE=4 ./tt/scripts/run_on_device.sh  # quartic accuracy variant
BW_SYN_CB_TILES=1 ./tt/scripts/run_on_device.sh    # buffer-depth ablation
./tt/scripts/run_on_device.sh --explore           # correctness only
./tt/scripts/run_on_device.sh --critical          # old ten-case path + TTNN
```

The tail path requires finite normal `g` and `4 <= |x| <= 88.5`; its contract covers
normal BF16 reference outputs only. It has no full-domain dispatch or special-input
guard. The 50% reduction in cycles is an unmeasured target; compare against both
split4 and the fused vendor baseline before attributing a gain to the algorithm.

## Roles

See [`AGENTS.md`](../AGENTS.md) for Reviewer / Systems Engineer / Researcher.

## 0. Labels (mandatory)

| Label | Meaning |
|-------|---------|
| **host** | CPU BF16 model + scaled f64 oracle |
| **host_sim** | C++ algorithmic model; separate from the SFPI device source |
| **device** | Measured on Wormhole/Blackhole via tt-metal |

Never publish device FTZ/cycle claims from host-only runs.

---

## 1. Host path (any machine) — already done offline

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/export_results --out results
python experiments/scripts/check_oracle_mpmath.py  # optional; mpmath 1.3.0
./build/run_tt_harness --csv results/hw/host_sim.csv
```

Paper host tables land under `results/paper/` (`ablation_host.csv`, `synth_cost.csv`,
`exhaustive_host.csv`, `oracle_probes.csv`). The device run writes
`results/paper/device_status.csv`. The optional
high-precision check covers selected finite probes and is not a proof over all
BF16 input pairs.

The one-shot device script uses `0` for its completed smoke packet, `1` for a
numerical failure, `3` for a device wiring error, `4` for a missing fresh
profiler log, and `5` for a TTNN baseline failure.

---

## 2. What **you** do on the TT machine

Copy this checklist. Nothing below can be finished on a laptop without the board.

### Step A — identify the tt-metal source revision

```bash
cd /path/to/bw_kernel_syn
export TT_METAL_HOME=/path/to/tt-metal
export BW_SYN_ARCH=wormhole
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"   # newer tt-metal reads this, not TT_METAL_HOME
git -C "$TT_METAL_HOME" rev-parse HEAD
```

`run_on_device.sh` writes the **exact** 40-char SHA to `tt/pin/tt-metal.COMMIT`
when that file contains `UNSET`. It refuses a later run against a different SHA.
It also refuses tracked changes in `TT_METAL_HOME`. Keep the resulting pin,
`results/paper/build_wormhole_*.log`, device provenance, and raw CSVs with the evidence.

### Step B — build with tt-metal

```bash
export BW_SYN_ROOT="$PWD"
cmake -S . -B build-tt -DCMAKE_BUILD_TYPE=Release -DBW_SYN_WITH_TTMETAL=ON
cmake --build build-tt -j 2
./build-tt/bw_syn_cli doctor
# This reports the configured environment; it does not probe the physical board.
```

### Step C — run device harness

```bash
# one-shot tail experiment (recommended; no torch/ttnn dependency):
./tt/scripts/run_on_device.sh

# legacy ten-case critical experiment:
python3 -c 'import torch, ttnn'
./tt/scripts/run_on_device.sh --critical

# or manually reproduce that legacy experiment:
./build-tt/export_results --out results
TT_METAL_DEVICE_PROFILER=1 ./build-tt/run_tt_harness --device --csv results/hw/${BW_SYN_ARCH}.csv
python3 tt/scripts/ttnn_tanh_baseline.py --device-csv results/hw/${BW_SYN_ARCH}.csv
```

The legacy `--critical` path builds the project, enables profiling for two Metal kernels, copies a fresh
raw profiler log to `results/paper/`, then disables profiling and measures
TTNN `tanh_bw` on the same ten BF16 tiles. The TTNN CSVs record output bits,
individual enqueue-to-synchronize host times, median and p95, and the module
path. Confirm that the imported TTNN build came from the pinned revision; a
module path outside the source tree is marked unverified.

### Step D — legacy full-domain acceptance (still failing subnormal cases)

Open `results/hw/wormhole.csv` (or `blackhole.csv`) and `results/paper/device_status.csv`.

- [ ] Row `label=device` for `(45,4)` has `got_bits=0x008f` (or document the device round)
- [ ] All ten device rows have `pass=1`, including `(45,4)`
- [ ] The `(max finite x,+inf g)` row is checked. An unguarded host factorization
      produces NaN there; the device guard must be checked on Wormhole.
- [ ] `false_zero_baseline=1` on `(45,4)` if reporting a device baseline output mismatch; this does not identify FTZ as the cause
- [ ] `tt_metal_commit` column matches `tt/pin/tt-metal.COMMIT`
- [ ] Commit this CSV + pin file (do **not** paste `host_sim` rows)
- [ ] Inspect `results/paper/ttnn_tanh_accuracy.csv` and the raw profiler CSV;
      record the installed TTNN build provenance

### Step E — paper metrics (cycles / insn / regs)

The custom-kernel correctness CSV leaves `cycles_per_tile,insn,regs` **empty**.
The profiler zones `BW_SYN_TANH_FACTORED` and `BW_SYN_TANH_MATERIALIZED` cover
ten-tile compute loops, including CB waits and packing. The harness runs one
correctness pass, five warmups, and 20 measured repetitions of each variant.
`summarize_tt_profiler` writes median and p95 cycle counts, and cycles per tile,
to `results/paper/device_cycles_wormhole.csv` from the final 20 zones. Keep the
raw profiler log alongside the summary and inspect zone pairing and scope. The
TTNN host enqueue-to-synchronize times include software overhead and must not
be compared directly with kernel cycles. Instruction and register counts require
compiler artifacts from the pinned build. The [Metalium device profiler guide](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)
documents the profiler flag and raw log location; check them against the pin.

---

## 3. Kernel locations

| Path | Role |
|------|------|
| `tt/kernels/compute/tanh_bw_scale_separated.cpp` | Device: factored `g*h*h`, with `h=2 exp(-|x|)/(1+exp(-2|x|))`; pack once |
| `tt/kernels/compute/tanh_bw_baseline.cpp` | Device: the same `h`, pack/unpack `h*h`, then multiply by `g` |
| `tt/kernels/compute/tanh_bw_tail_fused.cpp` | Scoped fused polynomial/exponent product; degree 0 selects fused vendor comparator |
| `include/bw_syn/detail/tail_exp_product.hpp` | Shared arithmetic body for host and device adapters |
| `tt/kernels/common/tanh_factor.h` | Shared tile-wide exponential and reciprocal sequence |
| `tt/kernels/dataflow/` | Reader / writer |
| `tt/kernels/common/cb_indices.h` | Shared CB indices |
| `tt/host/tt_metal_backend.cpp` | Host launch (`TtDeviceSession`) |
| `kernels/generated/` | **SFPI sketch** for supported nodes only (not device-linked) |

The legacy device kernel is a hand-written factorization, not a lowering of the scale-aware
IR. The emitter rejects unsupported exponent and rounding nodes rather than
changing their meaning. The compute kernels use documented tile-wide exp and
reciprocal operations with FP32 destination accumulation, but their numerical
behavior on the pinned board is unmeasured. First device
claim is **tanh** only.
The corresponding unguarded host IR fails eight special boundary pairs and
two paired-bit probes, despite passing the finite boundary sample. The device
kernel now uses tile-wide `isfinite`, `isinf`, and `where` operations to preserve
an infinite incoming gradient for finite x. The `(max finite x,+inf g)` tile
checks that guard on the board. The kernel still is not a lowering of the
selected synthesized IR.
Sigmoid is covered on **host** tables
(`results/paper/ablation_host.csv`).

---

## 4. If build/API fails on your pin

1. Confirm `find_package(TT-Metalium)` finds `$TT_METAL_HOME/build/lib/cmake/tt-metalium`
   (the build links the exported `TT::Metalium` target for headers and deps).
2. Classic vs metalium headers: `tt/host/tt_metal_backend.cpp` already dual-includes.
3. If your SHA only exposes `EnqueueProgram`, replace `detail::LaunchProgram` in that
   file and note the SHA in a comment + the pin file.
