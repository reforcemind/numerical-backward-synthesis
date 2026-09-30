# Hardware runbook - pull this repo onto a Tenstorrent machine and run

This document is the **authoritative** pull-and-run path. Host simulation always
works without a board. Device execution requires `tt-metal`.

## Roles

See [`AGENTS.md`](../AGENTS.md) for Reviewer / Systems Engineer / Researcher.

## 0. Labels (mandatory)

| Label | Meaning |
|-------|---------|
| **host** | CPU BF16 model + scaled f64 oracle |
| **host_sim** | Same kernels as device intent, executed on host |
| **device** | Measured on Wormhole/Blackhole via tt-metal |

Never publish device FTZ/cycle claims from host-only runs.

---

## 1. Host path (any machine) — already done offline

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/export_results --out results
./build/run_tt_harness --csv results/hw/host_sim.csv
```

Paper host tables land under `results/paper/` (`ablation_host.csv`, `synth_cost.csv`,
`device_status.csv`).

**Exit codes:** `0` pass, `1` numerical fail, `2` usage, `3` device wiring error.

---

## 2. What **you** do on the TT machine

Copy this checklist. Nothing below can be finished on a laptop without the board.

### Step A — pin firmware

```bash
cd /path/to/bw_kernel_syn
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_COMMIT=$(git -C "$TT_METAL_HOME" rev-parse HEAD)
echo "$TT_METAL_COMMIT" > tt/pin/tt-metal.COMMIT   # replace UNSET
echo "$TT_METAL_COMMIT" > results/device_tt_metal_commit.txt
```

`tt/pin/tt-metal.COMMIT` must be the **exact** 40-char SHA of `TT_METAL_HOME`. The
harness refuses `--device` if they differ.

### Step B — build with tt-metal

```bash
export BW_SYN_ROOT="$PWD"
export BW_SYN_ARCH=wormhole    # or: blackhole
cmake -S . -B build-tt -DCMAKE_BUILD_TYPE=Release -DBW_SYN_WITH_TTMETAL=ON
cmake --build build-tt -j
./build-tt/bw_syn_cli doctor
# expect: available=1 home=... commit=<sha> arch=wormhole
```

### Step C — run device harness

```bash
# one-shot (recommended):
./tt/scripts/run_on_device.sh

# or manually:
./build-tt/run_tt_harness --device --csv results/hw/${BW_SYN_ARCH}.csv
./build-tt/export_results --out results
```

### Step D — acceptance (must all be true before citing device)

Open `results/hw/wormhole.csv` (or `blackhole.csv`) and `results/paper/device_status.csv`.

- [ ] Row `label=device` for `(45,4)` has `got_bits=0x008f` (or document the device round)
- [ ] That row `pass=1`; nearby critical cases `pass=1`
- [ ] `false_zero_baseline=1` on `(45,4)` if claiming device intermediate flush
- [ ] `tt_metal_commit` column matches `tt/pin/tt-metal.COMMIT`
- [ ] Commit this CSV + pin file (do **not** paste `host_sim` rows)

### Step E — optional metrics (cycles / insn / regs)

The CSV leaves `cycles_per_tile,insn,regs` **empty** on device until you attach a
profiler / kernel dump from your pinned `tt-metal`. Do not invent numbers. When you
have them, fill those columns by hand or extend `tt/host/tt_metal_backend.cpp` under
the same pin SHA and re-run.

---

## 3. Kernel locations

| Path | Role |
|------|------|
| `tt/kernels/compute/tanh_bw_scale_separated.cpp` | Device: product in DST, pack once |
| `tt/kernels/compute/tanh_bw_baseline.cpp` | Device: pack/unpack sech² then mul |
| `tt/kernels/dataflow/` | Reader / writer |
| `tt/kernels/common/cb_indices.h` | Shared CB indices |
| `tt/host/tt_metal_backend.cpp` | Host launch (`TtDeviceSession`) |
| `kernels/generated/` | **SFPI sketch** (not device-linked) |

First device claim is **tanh** only. Sigmoid is covered on **host** tables
(`results/paper/ablation_host.csv`).

---

## 4. If build/API fails on your pin

1. Confirm `find_library` sees `tt_metal` / `tt_metalium` under `$TT_METAL_HOME/build/lib`.
2. Classic vs metalium headers: `tt/host/tt_metal_backend.cpp` already dual-includes.
3. If your SHA only exposes `EnqueueProgram`, replace `detail::LaunchProgram` in that
   file and note the SHA in a comment + the pin file.
