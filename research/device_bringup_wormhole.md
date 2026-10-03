# Wormhole device bring-up: tanh backward (first measurements)

**Diagnostic record only.** This run is excluded from paper performance tables.
The later `wh2` profiler trace and `wh3` rerun are described below. The first-run
tables are historical; `wh3` overwrote the default `results/paper` filenames.
Use the referenced Git revisions when reproducing an older table.

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
./tt/scripts/run_on_device.sh --critical; echo "exit=$?"   # legacy path: subnormal rows fail
```

Do not reuse a tt-metal build directory that was configured with
`--disable-profiler`. `ENABLE_TRACY=OFF` sticks in the CMake cache, and the run then
fails with exit 4 (no profiler log).

## Run 2: tail profiler trace only

Commit `acefed0` added a raw Wormhole profiler trace at tt-metal pin
`f9524a5f1b75180f00ce7413c2fc1c5cca1f29ee`. It contains 32 paired
single-core launches per kernel: seven correctness batches, five warmups, and
20 timed batches of 128 tiles. Replaying it with the launch-count and ordering
checks in `summarize_tt_profiler` gives median 400,378.5 cycles for the tail
candidate and 526,849 cycles for the materialized baseline (3,128 and 4,116
cycles per tile, respectively, after rounding). The candidate's zone is about
24% shorter in this diagnostic batch. These are kernel-zone cycles, not
end-to-end latency or a paper speedup.

The commit has no `tail_timed_wormhole.csv` or `format_wormhole.csv`. The trace
cannot establish whether either kernel met the numerical contract, which inputs
were timed, or where BF16 subnormals disappeared. The `wh3` harness gates timing
on the full normal-output sweep, records the timed input selection, checks the
profiler launch order, and compares the pinned tt-metal tanh-derivative primitive
multiplied by the same gradient. A fresh device run is required for those claims.

## Run 3: the default command repeated the legacy path

Results commit `e35b636` records source
`acaa87bbe49f79d8c539a6971f14b6131364a659`, UTC start
`2026-10-02T14:06:37Z`, and the same tt-metal pin
`f9524a5f1b75180f00ce7413c2fc1c5cca1f29ee`.

| Legacy variant | Median cycles/tile | p95 cycles/tile |
|----------------|-------------------:|----------------:|
| Guarded factored | 5,749.750 | 5,762.000 |
| Materialized | 4,206.650 | 4,210.200 |

The factored kernel again passes 8/10 critical cases. `(46,4)` and `(45,2)`
return zero for subnormal oracle outputs `0x0013` and `0x0047`. TTNN reports
34,121.5 ns median host enqueue-to-sync time; that timing scope cannot be
compared to the compute-zone cycle counts above.

This packet contains no tail correctness, tail cycle summary, or format probe.
It therefore provides no evidence for or against the `wh3` tail candidate.
The old no-argument script selected the critical harness, and exploratory
filenames under `results/hw` were ignored by Git. Both contributed to an
ambiguous handoff.

On `wh4`, the default command selects the new fused tail experiment and saves
every run under an unignored timestamped `results/runs/` directory, including
failure logs. The cubic/quartic candidate and its fair fused vendor comparator
are described in `research/numerical_semantics.md`; their device results are in
Run 4 below. The previous ten-case path remains available with `--critical`.

## Run 4: fused exponential-product tail on Wormhole (`wh4`)

First device results for the fused candidate described in
`research/numerical_semantics.md`. All numbers are **device**-labelled
single-core diagnostics, not paper measurements.

| | |
|---|---|
| Source | `d8d09c08de4818f65c83254c9ac3f596bbdaa5ea` (`wh4`), clean tree |
| tt-metal | `f9524a5f1b75180f00ce7413c2fc1c5cca1f29ee`, separate worktree build, slow dispatch |
| Machine | `wh-glx6u-02`, reservation `241170` (earlier runs: `238565`), device 0 |
| Main packet | `results/runs/wormhole_d8d09c0_20261003T095124Z_d3/` (degree 3, `cb_tiles=2`) |
| Ablations | `…T095157Z_d4/` (degree 4), `…T095212Z_d3/` (degree 3, `cb_tiles=1`) |
| Workload | 6,774 constant tiles; 4,636 have a normal BF16 reference output (in scope) |
| Timing | 128 recorded inputs (`4 <= |x| <= 88.5`, oracle exponent fields 1 to 241), 5 warmups + 20 measured launches, 128 tiles per launch |

### 4.1 Comparison

Correctness is over the 4,636 in-scope cases. Cycles are medians over the 20
measured launches, divided by 128 tiles. "Exact" counts outputs bit-identical
to the oracle among that kernel's passing outputs.

| Kernel | Pass | Failures | Exact | Median cyc/tile | p95 cyc/tile | vs split4 |
|---|---:|---|---:|---:|---:|---:|
| **Fused candidate, degree 3** | **4,636** | none | 88.6% | **2,027** | 2,093 | 0.65× |
| Fused candidate, degree 4 | 4,636 | none | 98.8% | 2,122 | 2,192 | 0.68× |
| Fused vendor tail × g | 3,278 | 1,358 return ±0 | 99.2% | 1,735 | 1,803 | 0.56× |
| Split4 tail (`wh2`/`wh3`) | 3,798 | 838 return ±0 | 99.5% | 3,125 | 3,126 | 1.00× |
| Vendor derivative × g | 3,278 | 1,358 return ±0 | 99.2% | 3,405 | 3,412 | 1.09× |
| Materialized | 3,203 | 1,363 ±0, 70 wrong nonzero | 87.9% | 4,116 | 4,117 | 1.32× |

All five medians were recomputed independently from the raw profiler CSV and
agree with `cycles.csv`. The degree-3 candidate's signed error distribution
is −1 ULP: 122, 0: 4,108, +1: 406. The upward skew matches the one-sided constant
offsets in the polynomial.

### 4.2 Analysis

**What the approach is.** For finite tail inputs, the kernel never forms
`sech²(x)` or `exp(-2|x|)` as a value. It range-reduces `t = -2|x|` to
`k·ln2 + r`, evaluates a short polynomial in `r`, multiplies by the
*significand* of `g` (exponent forced to 127), rounds that bounded product to
BF16, and only then composes the final exponent from the product, `g`, `k` and
the scale. The tiny factor exists only as an integer exponent. This is scale
separation done with exponent arithmetic rather than with a second floating-point
factor.

**Why the comparators fail where they do.**

- *Vendor derivative × g* and *fused vendor tail × g* fail on the same 1,358
  cases, and every failure is a false zero. Their failing outputs reach BF16
  exponent field 128 (values near 2). Once the derivative alone underflows,
  no gradient can scale it back, even when `g` is near the BF16 maximum. Fusing
  the vendor polynomial with `g` does not change this, because the product is
  still formed after the derivative has gone to zero.
- *Split4* fails only at oracle exponent field 1, the smallest normal binade
  (838 cases, all false zeros). The host model of split4 passes all 4,636 cases.
  The difference is consistent with the format probe in this packet: the device
  compute path flushes FP32 subnormals and corrupts some field-1 values
  (`0x0080` → `0x00c0`). Split4 passes through subnormal intermediates near the
  boundary. The candidate rounds the bounded significand first and promotes to
  signed minimum normal by setting the exponent directly, so it never relies on a
  subnormal in DST.
- *Materialized* fails the most (1,433) and is also the only path with wrong
  nonzero outputs (70). Its host model passes 3,924, so most of its device
  failures also come from the device path rather than the formula.

**Device versus host model.** Of the 4,636 in-scope outputs, 4,630 are
bit-identical between the device and the host `std::fma` adapter. The other 6
differ by +1 ULP and still pass. Out of scope, 6 of 2,138 differ (min normal
versus zero). `std::fma` is not a bit-exact model of SFPMAD. These 12
differences are the measured size of that gap for this sweep only.

**Where the speed comes from.** Against split4 the candidate saves 35% of
cycles; against materialized it saves 51%; against the vendor derivative it saves
40%. But the fused vendor comparator uses the same single SFPU traversal, reader
and packing, and it is **14% faster** than the candidate (1,735 vs 2,027). The
speed-up over split4 and materialized therefore comes mainly from **fusion**: one
SFPU pass per tile instead of a chain of `*_init`/op pairs. It does not come
from the exponent-composition method. The method's measured contribution is
**correctness range**. It recovers the 1,358 vendor false zeros at a cost of
about 292 cycles/tile (+17%) relative to an equally fused but inaccurate kernel.
The 50% reduction target in the semantics note is met against materialized,
not against split4 or the fused vendor baseline.

**Degree 3 versus 4.** Degree 4 adds one multiply-add per element. It costs
94 cycles/tile (+4.7%) and raises exact outputs from 88.6% to 98.8%. Both
degrees stay within 1 ULP on every in-scope case. Degree 3 is the better choice
under the current 1-ULP contract. Degree 4 is preferable if exactness matters
(for example, bit-reproducibility against a reference).

**Buffer depth.** With `cb_tiles=1`, only materialized slows (4,116 → 4,560,
+11%). Its extra pack/unpack round trip through `cb_tmp` needs a second buffer
slot to overlap. The fused candidate (2,024), split4 and the vendor kernels are
unchanged, so their inner loops are not limited by buffer capacity.

**Bottleneck and run-to-run spread.** TRISC1 (math) is still the longest zone,
but for the candidate TRISC0 (1,995) and TRISC2 (2,009) are within 2% of it
(2,027). For materialized the three are within 0.2% of each other, but at twice
the length. The legacy and split4 kernels vary by 0.1–0.2% across measured
launches. Both fused kernels show discrete steps: the candidate's 20 launches
cluster at 2,024, ~2,031, ~2,092 and 2,163; vendor fused clusters at 1,734,
~1,742 and ~1,803. The inputs are identical every launch, so this is not
data-dependent arithmetic. A plausible but untested explanation: once math work
is short, reader/NoC/DRAM timing starts to show in the zone. Medians are robust
to it; p95 is 3–4% above the median.

**Repeatability.** The degree-3 candidate measured 2,027 and 2,024 in two runs 48 s
apart. Split4 measured 3,125 and materialized 4,116 here, against 3,128 and 4,116
on the previous day under a different reservation. Cross-chip variance is still
unmeasured.

### 4.3 Limits of this result

- Scope is finite `4 <= |x| <= 88.5`, finite normal `g`, and normal reference
  outputs. Out of scope the candidate returns ±0 (1,574 cases) or signed minimum
  normal (564 cases, the documented boundary promotion). There is no dispatch,
  inf/NaN guard, or subnormal support, so this is not a replacement for the
  full-domain kernel.
- Each tile repeats one `(x, g)` pair. Mixed-lane tiles, multi-core runs, fast
  dispatch and end-to-end TTNN-level latency are untested.
- The vendor comparators are tt-metal primitives inside this harness. They are
  not TTNN's `tanh_bw` op. TTNN `tanh_bw` passes 3 of 10 critical tiles in the
  `--critical` run on the same day, and no device cycle count exists for it.
- Range reduction, polynomial evaluation, exponent reconstruction and fusion are
  established techniques (see the attribution note in
  `research/numerical_semantics.md`). Scale separation is not claimed as novel,
  and `claimed_sound` remains false.

### 4.4 Suggested next measurements

1. A copy-only compute kernel timed with the same reader and packer gives the
   fixed per-tile floor. Cycles above it can then be attributed to SFPU work.
2. Try cutting the 17% gap to fused vendor. Profile the candidate's SFPU
   sequence (range reduction plus reconstruction branches) and test whether the
   boundary-repair `v_if` blocks dominate.
3. Repeat the default run on two or three more chips and reservations for
   variance.
4. Profile TTNN `tanh_bw` on device with the same tiles, for a direct comparison
   with what tt-metal ships.

## Run 5: generated quadratic (`wh5`) and tt-metal `main` `tanh_bw`

All numbers are **device**-labelled single-session diagnostics.

| | |
|---|---|
| Source | `cb1cc6c6189e41ca8dd26bc27006b469c09aedff` (`wh5`), clean tree |
| tt-metal build | `f9524a5f1b75180f00ce7413c2fc1c5cca1f29ee`, slow dispatch for bw_syn kernels |
| Machine | `wh-glx6u-02`, reservation `241170`, device 0 |
| Main packet | `results/runs/wormhole_cb1cc6c_20261003T145747Z_d2/` (degree 2, default) |
| Same-source cubic | `results/runs/wormhole_cb1cc6c_20261003T145822Z_d3/` |
| TTNN packet | `results/runs/wormhole_ttnn_tanh_bw_main_20261003T150140Z/` |
| Workload | Same 6,774 tiles and 4,636 in-scope cases as Run 4; 128 timed inputs |

### 5.1 tt-metal `main` is the pinned implementation for `tanh_bw`

On 2026-10-03, `origin/main` was `70222e907f979825d94d74ef43e50fa9e7f8aff8`, 110
commits after the pin. The diff from the pin to that commit leaves the whole
`tanh_bw` path unchanged: `ttnn/.../unary_backward/` (op, program factory,
`eltwise_bw_tanh.cpp`), `api/compute/eltwise_unary/tanh_derivative.h`, and the
Wormhole `ckernel_sfpu_tanh_derivative.h`. The only diff touching files on that
search path is unrelated new headers added to `tt_metal/hw/sources.cmake`. The
pinned build's `ttnn.tanh_bw` is therefore the current `main` implementation. It was
loaded from the pin worktree, which is recorded in the packet provenance.

`main`'s kernel copies `grad` and `input` into DEST, applies
`TanhDerivative<Approx::Exact>` and `MulBinary`, then packs. Unlike the
bw_syn harness, it does not force FP32 DEST accumulation.

### 5.2 How TTNN was measured

The script `ttnn_tanh_bw_sweep.py` is saved in the TTNN packet. It is a Python
cross-check, not repository code.

- **Correctness:** one `ttnn.tanh_bw` call over all 6,774 constant tiles from
  the Run 5 `tail.csv`. The script checks lane uniformity and grades in-scope cases
  with the same `<= 1` BF16 ULP rule against the recorded oracle bits.
- **Cycles:** the 128 timed inputs are replicated once per core, so each of the
  72 cores processes exactly those 128 tiles. One correctness launch, 5 warmups
  and 20 measured launches were run under the device profiler. Per core and
  launch, the span runs from the earliest to the latest `TRISC-KERNEL` zone across
  the three TRISCs, divided by 128. The log contains exactly 26 launches, all of
  them `tanh_bw`.
- **Not like-for-like** with bw_syn zones. TTNN runs in fast dispatch on 72 cores
  with its own reader and writer and shared DRAM. `TRISC-KERNEL` also includes
  kernel initialisation, whereas the bw_syn zones start after
  `compute_kernel_hw_startup`. Treat the TTNN cycle figure as the shipped op's
  per-core cost, not as a controlled kernel comparison. The controlled
  counterpart is the "vendor derivative × g" row.

### 5.3 Results

| Kernel | Pass / 4,636 | Failures | Exact | Median cyc/tile | Max of 20 |
|---|---:|---|---:|---:|---:|
| **Generated quadratic (wh5 default)** | **4,636** | none | 81.9% | **1,531** | 1,603 |
| wh4 cubic, same packet | 4,636 | none | 88.6% | 1,986 | 2,056 |
| Fused vendor tail × g | 3,278 | 1,358 ±0 | 99.2% | 1,691 | 1,755 |
| Split4 | 3,798 | 838 ±0 | 99.5% | 3,125 | 3,127 |
| Vendor derivative × g | 3,278 | 1,358 ±0 | 99.2% | 3,405 | 3,828 |
| Materialized | 3,203 | 1,363 ±0, 70 wrong | 87.9% | 4,117 | 4,119 |
| **TTNN `tanh_bw` (`main`)** | 3,258 | 1,378 ±0 | 88.2% | 4,154 † | 4,172 (p95) † |

† Per core, 72 cores, fast dispatch, `TRISC-KERNEL` zone (see §5.2).

The same-source degree-3 packet repeats the cubic at 1,985 and fused vendor at
1,691. The two `wh5` packets agree to within 1 cycle/tile on every shared kernel.

### 5.4 Analysis

**The generated quadratic is the first candidate faster than the equally fused
vendor kernel.** At 1,531 cycles/tile it is 9.5% faster than fused vendor (1,691),
23% faster than the cubic in the same packet (1,986), and 51% faster than split4.
It still passes all 4,636 in-scope cases. In Run 4 the cubic was 17% *slower*
than fused vendor. That gap is now reversed by an algorithmic change, measured
within one packet and one pipeline.

**Pipeline versus algorithm.** `wh5` also removed a redundant second
`copy_init` from the shared fused path. That change alone accounts for the cubic
moving from 2,027 (Run 4) to 1,986, and fused vendor from 1,735 to 1,691, about
40 cycles/tile each. Every comparison above is within the Run 5 packet, so all
three fused kernels include it. The remaining 455 cycles/tile between the cubic
and the quadratic come from the shorter arithmetic: base-2 range reduction and two
polynomial FMAs instead of a two-part `ln 2` reduction and three FMAs, plus
minimum-normal saturation in place of the `v_if` boundary repair.

**Cost of the shorter polynomial.** Exact outputs drop from 88.6% (cubic) to
81.9%. Every output stays within the 1-ULP contract. The error is more
balanced: −1 ULP on 543 cases and +1 on 294, against 122 and 406 for the
cubic. If bit-exactness matters, degree 4 (98.8% exact, Run 4) remains the option.

**tt-metal `main`.** The shipped op fails 1,378 of 4,636 in-scope cases, all
false zeros, for the same reason as the vendor comparator. The derivative
underflows before the gradient is applied. It is also less exact on the cases
it passes (88.2% versus 99.2% for the bw_syn vendor-derivative kernel). Its
outputs match that kernel bit-for-bit on 4,019 of 4,636 tiles, and the two differ on
230 pass/fail outcomes. This is consistent with `main` keeping DEST in BF16
rather than FP32; that attribution is not isolated. Per core, the shipped op
takes 4,154 cycles/tile in its multi-core setting. The generated quadratic takes
1,531 single-core, about 2.7× less, while also passing the 1,378 cases `main`
returns as zero. Because the settings differ (§5.2), the 2.7× is indicative
only. The controlled figure is 2.2× against vendor derivative × g (3,405) in the
same harness.

**Repeatability.** The materialized kernel measured 4,116–4,117 in every tail
packet from Run 2 to Run 5, across two days and two reservations. Split4 measured
3,125–3,128. The fused kernels again show small launch-to-launch steps (max up
to 4.7% above median); medians are unaffected.

### 5.5 Limits

The scope from Run 4 §4.3 still applies: finite `4 <= |x| <= 88.5`, finite normal
`g`, normal reference outputs only, constant tiles, single core, no dispatch to
a full-domain kernel, and `claimed_sound=false`. The quadratic's coefficients were
accepted by an exhaustive **host** sweep (`results/host/tail_synthesis/`). The
device run covers the 6,774-tile sweep only.

## Summary across all runs

Cycles are median cycles per tile. "Pass" is over each run's own in-scope set,
and the sets grew between runs (Run 2: 753, Run 3: 2,938, Runs 4–5: 4,636), so
pass counts are comparable only within a column. Critical rows are the
ten-case legacy harness.

| Kernel | Run 1 `wh1` (Oct 1) | Run 2 `wh2` | Run 3 `wh3` | Run 4 `wh4` (Oct 3) | Run 5 `wh5` (Oct 3) |
|---|---|---|---|---|---|
| Factored + inf guard (critical, 10 tiles) | 5,748; 8/10 | – | 5,750; 8/10 | 5,707; 8/10 | – |
| Factored, guard removed (ablation) | 3,965; 8/10 | – | – | – | – |
| Materialized (critical) | 4,207 | – | 4,207 | 4,218 | – |
| Split4 tail | – | 3,128; 753/753 | 3,128; 2,938/2,938 | 3,125; 3,798/4,636 | 3,125; 3,798/4,636 |
| Fused cubic tail | – | – | – | 2,027; 4,636/4,636 | 1,986; 4,636/4,636 |
| Fused quartic tail | – | – | – | 2,122; 4,636/4,636 | – |
| **Generated quadratic tail** | – | – | – | – | **1,531; 4,636/4,636** |
| Vendor derivative × g | – | – | 3,407; 2,610/2,938 | 3,405; 3,278/4,636 | 3,405; 3,278/4,636 |
| Fused vendor tail × g | – | – | – | 1,735; 3,278/4,636 | 1,691; 3,278/4,636 |
| Materialized (tail sweep) | – | 4,116; 648/753 | 4,116; 2,562/2,938 | 4,116; 3,203/4,636 | 4,117; 3,203/4,636 |
| TTNN `tanh_bw`, critical | – | – | 3/10 | 3/10 | – |
| **TTNN `tanh_bw`, tt-metal `main`** | – | – | – | – | **4,154 †; 3,258/4,636** |

† Per core, 72 cores, fast dispatch, `TRISC-KERNEL` zone; not a controlled
kernel comparison (see §5.2).

Evidence: Run 1 and the critical rows are under `results/paper/`. Runs 4–5 are under
`results/runs/`. The Run 2 and Run 3 tail sweeps were written to git-ignored
`results/hw/` files and were not committed (see "Run 2" and "Run 3" above). Their
figures are from this bring-up session only.
