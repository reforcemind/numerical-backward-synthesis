# Wormhole device bring-up: tanh backward (first measurements)

**Diagnostic record only.** This run is excluded from paper performance tables.
The `wh2` exploration replaces its ten-tile workload with a finite tail sweep
and adds a format-path probe; those results have not been run on hardware.

First on-board run of the two tanh-backward compute kernels and the TTNN
`tanh_bw` baseline. Everything here is **device**-labelled. It comes from one
session on one chip and is a bring-up record, not a paper result. See
[Caveats](#caveats) before quoting any number.

| | |
|---|---|
| Date (UTC) | 2026-10-01, run started 21:01:12 |
| Machine | Wormhole Galaxy (`wh-glx6u-02`, 32 chips), firmware bundle 19.8.1, KMD 2.9.0 |
| Chip / core | device 0, a single Tensix core at logical (0,0) |
| tt-metal | `f9524a5f1b75180f00ce7413c2fc1c5cca1f29ee` (`main`), Release, Tracy profiler on |
| Dispatch | slow dispatch (`TT_METAL_SLOW_DISPATCH_MODE=1`, set by `TtDeviceSession`) |
| bw_syn | `ae20db3` **plus uncommitted porting changes** (listed under [Porting changes](#porting-changes-needed-for-this-tt-metal-pin)) |
| Provenance | `results/paper/device_provenance_wormhole.txt` (kernel sha256s are for the working tree) |

## 1. Summary

1. **Motivating case works on the board.** At `(x=45, g=4)` the factored kernel
   returns `0x008f`, matching the oracle. The materialized kernel and TTNN `tanh_bw`
   both return `0x0000`.
2. **8 of 10 critical tiles pass.** The two failures are exactly the cases whose
   oracle result is a BF16 **subnormal**. The device returns `+0` for both. Under the
   default contract (`flush_output_subnormals = false`) these are real failures, and
   `device_status.csv` reports `all_pass=0`.
3. **As shipped, the factored kernel is about 37% slower** than the materialized
   kernel (5,748 vs 4,207 cycles/tile).
4. **The slowdown comes from the infinite-gradient guard, not from scale separation.**
   With the guard removed, the factored kernel takes **3,965 cycles/tile, about 6%
   faster** than materialized. The guard costs about 1,780 cycles/tile.
5. **The guard's target case passed without the guard.** `(max finite x, +inf g)`
   returned `0x7f80` both with and without it. One probe is not enough to drop the
   guard (see §5.2).
6. **TTNN `tanh_bw` passes 3 of 10** of the same tiles. It returns `0x0000` wherever
   the result is below the normal range, and also for the infinite-gradient tile.

## 2. Correctness: critical tiles

Each tile is filled with one constant `(x, g)` pair. The harness checks that all
1,024 lanes agree. The oracle is the host scaled-f64 reference rounded to BF16 RNE.

| x | g | oracle | factored (device) | materialized (device) | TTNN `tanh_bw` | factored pass |
|---|---|---|---|---|---|---|
| 45 | 4 | `0x008f` | `0x008f` | `0x0000` | `0x0000` | 1 |
| 44 | 4 | `0x0204` | `0x0204` | `0x0224` | `0x0204` | 1 |
| 46 | 4 | `0x0013` (subnormal) | `0x0000` | `0x0000` | `0x0000` | **0** |
| 45 | 2 | `0x0047` (subnormal) | `0x0000` | `0x0000` | `0x0000` | **0** |
| 45 | 8 | `0x010f` | `0x010f` | `0x0000` | `0x0000` | 1 |
| −45 | 4 | `0x008f` | `0x008f` | `0x0000` | `0x0000` | 1 |
| 45 | −4 | `0x808f` | `0x808f` | `0x0000` | `0x0000` | 1 |
| 0 | 1 | `0x3f80` | `0x3f80` | `0x3f80` | `0x3f80` | 1 |
| 20 | 1 | `0x239d` | `0x239d` | `0x239d` | `0x239d` | 1 |
| 3.3895e38 (max finite) | +inf | `0x7f80` | `0x7f80` | `0x7f80` | `0x0000` | 1 |

Sources: `results/hw/wormhole.csv`, `results/paper/ttnn_tanh_accuracy.csv`.

Notes:

- At `(45, −4)` the materialized kernel returns `+0` (`0x0000`), not `−0`. The
  oracle sign is negative, so the baseline also loses the sign there.
- At `(44, 4)` the materialized kernel returns `0x0224` against an oracle of `0x0204`.
  That is a large error in a *normal* result, not only a false zero. The reported
  `false_zero_baseline` flag does not cover this row.
- The subnormal rows fail for every path tested, TTNN included. **This record
  does not establish where the flush happens** (SFPU arithmetic, DST, or the BF16
  packer). Per `AGENTS.md`, do not call it device FTZ until a probe isolates it
  (§5.1).

## 3. Cycles

Profiler zones `BW_SYN_TANH_FACTORED` and `BW_SYN_TANH_MATERIALIZED` wrap the
10-tile compute loop, including CB waits and packing. There is 1 correctness launch,
then 5 warm-ups, then 20 measured launches. Each TRISC records its own copy of the
zone. A launch is measured from the earliest TRISC start to the latest TRISC end
(`tools/summarize_tt_profiler.cpp`).

| Variant | median cycles / tile | p95 cycles / tile | Source |
|---|---|---|---|
| Factored, as shipped (with inf guard) | 5,748 | 5,757 | `results/paper/device_cycles_wormhole.csv` |
| Factored, inf guard removed (ablation) | **3,965** | 3,978 | `results/paper/ablation_no_inf_guard_wormhole/device_cycles.csv` |
| Materialized | 4,207 | 4,212 | `results/paper/device_cycles_wormhole.csv` |
| Materialized (rerun alongside ablation) | 4,208 | 4,222 | ablation directory |

Median per-TRISC zone length (cycles/tile):

| Variant | TRISC0 (unpack) | TRISC1 (math) | TRISC2 (pack) |
|---|---|---|---|
| Factored, with guard | 5,182 | **5,748** | 5,220 |
| Factored, no guard | 3,315 | **3,965** | 3,617 |
| Materialized | 4,162 | **4,207** | 4,119 |

The chip clock is reported as 1000 MHz in the profiler header, so 1 cycle ≈ 1 ns.

### Interpretation (hedged)

- TRISC1 is the longest zone in these runs. The zone includes circular-buffer
  waits, so this observation alone does not establish math utilization or a
  math-bound kernel.
- **Materialization is close to free here.** In the materialized kernel the extra
  pack to `cb_tmp` and unpack back overlap with SFPU work on the other TRISCs. So
  "pack once" saves only about 6% (4,207 → 3,965). The
  scale-separated kernel is therefore a **correctness** improvement with a small
  performance gain on this kernel shape. It should not be presented as a speed
  result.
- **The guard is expensive** because it adds four SFPU operations (`isfinite`,
  `isinf`, a binary `mul`, `where`), each with its own `*_init` reconfiguration,
  plus an extra `copy_tile`. We have not separated the init cost from the
  operation cost.
- The two materialized measurements (4,207 and 4,208) suggest good repeatability
  within one session. Repeatability across sessions and chips has not been measured.

### TTNN timing is not comparable

`results/paper/ttnn_tanh_summary.csv` gives a host enqueue-to-synchronize median of
34,166 ns (p95 36,876 ns) over 30 repeats of 10 tiles. That is wall time including
host software overhead and fast dispatch. It is **not** comparable with the kernel
cycle counts above.

## 4. Infinite-gradient guard ablation

`results/paper/ablation_no_inf_guard_wormhole/` contains:

| File | Content |
|---|---|
| `no_inf_guard.patch` | Exact change applied to `tt/kernels/compute/tanh_bw_scale_separated.cpp` for the run (guard removed; the now-unused `copy_tile(cb_x, 0, 3)` was kept) |
| `device_correctness.csv` | Harness output for the 10 tiles |
| `device_cycles.csv` | Summary from `summarize_tt_profiler` |
| `raw_device_profiler.csv` | Raw profiler log |

The ablation was run on the same chip, pin and session as §2–3, right after the main
run. The kernel source was restored afterwards. The correctness result is identical
to the guarded kernel: same 8/10, same two subnormal failures, and
`(max finite x, +inf g)` still returns `0x7f80`.

Why the guard was there: on the host, the unguarded factorization computes
`h(x) = 2e^{-|x|}/(1+e^{-2|x|})`. For large finite `|x|` this underflows to 0, and
`(+inf)·0·0` gives NaN where the oracle gives `+inf`. On the board, both the
unguarded factored kernel and the materialized kernel return `+inf` for that tile.
One of three explanations must hold: the device `exp`/`recip` sequence does not
produce `h = 0` at that `x`, or the SFPU multiply does not produce NaN for `inf·0`,
or some later step rewrites the NaN. Which one is not known yet.

## 5. Open issues

### 5.1 Subnormal outputs return +0 (contract failure)

- **Observation:** every path (factored, materialized, TTNN) returns `0x0000` where
  the oracle result is subnormal.
- **Not yet known:** whether SFPU arithmetic, DST storage in FP32 mode, or the
  FP32→BF16 pack flushes. A minimal probe: a kernel that copies a tile of BF16
  subnormal values straight through (unpack → DST → pack), plus one that multiplies
  a normal by a scale giving a subnormal product. Compare the outputs.
- **Decision needed (researcher):** keep the default contract and report 8/10, or
  define a device contract with `flush_output_subnormals = true` and report
  against both. Changing the contract only to make the table pass would be an
  overclaim. If it changes, update `research/numerical_semantics.md` and add tests.

### 5.2 Is the inf guard needed on device?

Recommended next step: a device sweep with `g ∈ {+inf, −inf}` and `x` over the
range where the host `h(x)` underflows (|x| roughly 45 to max finite, both signs),
plus `x = ±inf` and `x = NaN`. Also record the raw `h(x)` tile for large `x` to see
what the device produces.

- If no sweep point yields an incorrect NaN, the guard can be dropped. The factored
  kernel is then about 6% faster than materialized.
- Otherwise, a cheaper guard is needed. The only case to catch is "g infinite and x
  finite", so `where(isinf(g) && isfinite(x), g, g·h·h)` might be done with fewer SFPU
  operations and fewer `*_init` switches.

Either way the change is local to `tt/kernels/compute/tanh_bw_scale_separated.cpp`
(about 10 lines) plus new sweep cases in the harness. No host or IR refactor is
needed.

### 5.3 Host result drift on this machine

Re-running `export_results` here changed one committed host row in
`results/paper/exhaustive_host.csv`: sigmoid `three_bf16_bit_permutations`
`max_ulp` went from 0 to 1. All 196,608 cases still pass, and no host numerics code
was changed. This is probably a libm difference from the machine that produced the
committed CSV. Decide which value to commit, and record the libm/compiler in the
host provenance. `synth_cost.csv` also changed, but only in its wall-clock timing
column.

## 6. Caveats

- One session, one chip (device 0 of a Galaxy), one core, slow dispatch. No
  cross-chip or cross-session variance yet.
- Each tile holds a single constant `(x, g)`. SFPU timing is assumed to be
  data-independent for these ops, but we have not checked that.
- Cycle zones include CB waits and packing. Instruction counts and register usage
  are still `unmeasured`.
- The run used an uncommitted working tree on top of `ae20db3`. Commit the porting
  changes, then **re-run** `tt/scripts/run_on_device.sh` so the provenance commit
  matches the code.
- The first device attempt in this session aborted with the device open. The next
  run logged "dispatch kernels still running" on the other Galaxy chips and cleared
  them during init. Later runs were clean. If such warnings show up, re-run before
  trusting numbers.

## 7. Porting changes needed for this tt-metal pin

| Change | File |
|---|---|
| Link tt-metal through its exported CMake package (`find_package(TT-Metalium)`, `TT::Metalium`) instead of a hand-written include list; current headers need `tt_stl`, fmt, spdlog, etc. | `CMakeLists.txt` |
| Export `TT_METAL_RUNTIME_ROOT` (tt-metal no longer derives its root from `TT_METAL_HOME`) | `tt/scripts/run_on_device.sh`, `tt/HARDWARE.md` |
| Dataflow include moved to `api/dataflow/dataflow_api.h` | `tt/kernels/dataflow/*.cpp` |
| `where_tile<DataFormat::Float32>` (the enum is no longer under `tt::`) | `tt/kernels/compute/tanh_bw_scale_separated.cpp` |
| Open the device in slow dispatch to match `detail::LaunchProgram` / `WriteToBuffer` / `ReadFromBuffer`; mixing modes aborts in `CloseDevice` | `tt/host/tt_metal_backend.cpp` |
| Profiler CSV format: phase column is now `type` (`ZONE_START`/`ZONE_END`), there is no usable run ID under slow dispatch, and each TRISC logs its own zone. The old parser would have mixed launches across TRISCs. | `tools/summarize_tt_profiler.cpp` |

Remaining warnings (not errors): `noc_async_read_tile` and `noc_async_write_tile` are
deprecated in favour of `noc_async_read_page` and `noc_async_write_page`.

## 8. Reproducing on this machine

The home directory is a 9.4 GB NFS quota, so builds, caches and the Python env must
live on `/localdev`.

```bash
export L=/localdev/ijankowski
export CPM_SOURCE_CACHE=$L/cpmcache CCACHE_DIR=$L/ccache
export UV_CACHE_DIR=$L/uv-cache UV_PYTHON_INSTALL_DIR=$L/uv-python
export TT_METAL_CACHE=$L/tt-metal-cache PYTHON_ENV_DIR=$L/tt-metal-python-env-main

# tt-metal (once per pin; profiler is on by default)
cd ~/tt-metal-work/tt-metal
./build_metal.sh --release --build-dir $L/tt-metal-build-prof --cpm-source-cache $L/cpmcache --enable-ccache
./create_venv.sh

# experiment
source $PYTHON_ENV_DIR/bin/activate
export TT_METAL_HOME=~/tt-metal-work/tt-metal
export BW_SYN_ARCH=wormhole BW_SYN_TT_DEVICE_ID=0 BW_SYN_BUILD_JOBS=16 BUILD_DIR=$L/bw-syn-build-tt
cd ~/tmp/numerical-backward-synthesis
./tt/scripts/run_on_device.sh; echo "exit=$?"   # currently exits 1 (subnormal rows)
```

Do not reuse a tt-metal build directory that was configured with
`--disable-profiler`. `ENABLE_TRACY=OFF` sticks in the CMake cache, and the run then
fails with exit 4 (no profiler log).
