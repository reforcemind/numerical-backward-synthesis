# Completing the setup and focusing new approaches

Status as of 2026-10-03, after Run 5 (`research/device_bringup_wormhole.md`).
This note covers three things: what the setup still lacks, what to build next,
and what to watch for when trying new kernel ideas. It is a working plan, not a
results document. All results cited are **device** diagnostics from single
sessions.

## 1. Where things stand

| Area | Status |
|---|---|
| Best candidate | Generated base-2 quadratic (`wh5`). 4,636/4,636 in-scope tail cases correct, 1,531 cycles/tile single-core |
| Best equally fused comparator | Fused vendor tail × g: 1,691 cycles/tile, 1,358 false zeros |
| tt-metal `main` `tanh_bw` | 3,258/4,636; 1,378 false zeros; ~4,154 cycles/tile per core (72 cores, not controlled) |
| Scope covered | Finite `4 <= |x| <= 88.5`, finite normal `g`, normal reference outputs, constant tiles, one core, slow dispatch |
| Not covered | `|x| < 4` (most real inputs), inf/NaN, subnormal outputs, mixed-lane tiles, multi-core, fast dispatch, TTNN integration, sigmoid on device, cross-chip variance |

The honest claim today is narrow. The candidate is correct where `main` returns
false zeros, and about 2.2× faster than the same derivative × g computation in
the same harness, **on the tail inputs it covers**. Neither a model-level accuracy
gain nor an end-to-end speedup has been shown.

## 2. What the setup still needs

### 2.1 Environment and reproducibility (small, do first)

| Gap | What to add | Why |
|---|---|---|
| Exports retyped in every shell | Commit `tt/scripts/env_wormhole.sh` that sets the `/localdev` paths, `TT_METAL_HOME` (pin worktree), `TT_METAL_RUNTIME_ROOT`, `BUILD_DIR`, venv activation and `PYTHONPATH` to the pin's `ttnn` | Two of this session's failures were a fresh shell and a home-quota build |
| Pin worktree created by hand | A script that creates or refreshes `/localdev/.../tt-metal-pin` at `tt/pin/tt-metal.COMMIT` with submodules and builds it | Your daily tt-metal checkout moves between feature branches; the pin must not |
| TTNN measured from a scratch script | Move `ttnn_tanh_bw_sweep.py` (now in the Run 5 TTNN packet) into `tt/scripts/` and call it from `run_on_device.sh` | Every packet should carry the shipped-op comparison, generated the same way |
| Pin drift versus `main` | A check that diffs the pin against `origin/main` on the op path (`unary_backward/`, `tanh_derivative*`, `kernel_lib/eltwise`) and records the result in provenance | Run 5's "pin = main" statement was checked by hand |
| Deprecated host APIs | Port `tt/host/tt_metal_backend.cpp` from `CreateDevice`, `detail::LaunchProgram`, `CreateBuffer(InterleavedBufferConfig)` and `detail::Write/ReadFromBuffer` to `distributed::MeshDevice` / `EnqueueMeshWorkload` / `MeshBuffer` | Still present on `main` today, but their stated removal dates (2026-09-21 to 2026-10-30) have passed or are near. Any re-pin could break the harness |
| Single-chip evidence | Run every packet on at least 3 chips (other device IDs on the Galaxy) and 2 reservations | Cross-chip variance is unmeasured |

### 2.2 Harness capabilities (needed for any real-world claim)

1. **Mixed-lane tiles.** The backend throws on non-uniform output in a tile
   (`tt_metal_backend.cpp`, "nonuniform output in constant-input device tile").
   Add a mode where every one of the 1,024 lanes holds a different `(x, g)`, and
   compare lane-wise. This tests lane divergence in `v_if` blocks and SFPU
   condition codes, which constant tiles cannot reach.
2. **Full-domain kernel.** One kernel that handles every BF16 `x`:
   - the existing near-zero path (or the vendor derivative) for `|x| < 4`
   - the tail path for `4 <= |x| <= 88.5`
   - a defined result for `|x| > 88.5`, ±inf, NaN and inf `g`
   This is the only form that can replace `tanh_bw`. Measure its cost on
   all-small, all-tail and mixed tiles. Per-lane selection costs cycles, and that
   cost decides the real speedup.
3. **Copy-only floor kernel.** Time a kernel with the same reader, copies and
   pack but no SFPU work. Cycles above this floor are attributable to arithmetic;
   below it, a new approach cannot help.
4. **Main's kernel in the controlled harness.** Port `eltwise_bw_tanh.cpp` as-is
   (BF16 DEST, `kernel_lib` chain) as a harness comparator. Then the shipped
   kernel is compared single-core under identical dataflow, and the FP32-vs-BF16
   DEST effect is isolated (Run 5 could only infer it).
5. **Multi-core and fast dispatch.** Run the candidates across the full compute
   grid with fast dispatch, so throughput (tiles/s per chip) can be compared with
   the TTNN measurement directly.
6. **TTNN-level integration.** Wrap the candidate as a TTNN op (or a `generic_op`
   with the candidate compute kernel) so it runs through the same program factory,
   memory configs and op profiler as `ttnn.tanh_bw`.

### 2.3 Inputs and metrics

| Add | Detail |
|---|---|
| Exhaustive device x-sweep | All 65,536 BF16 `x` for a fixed set of `g`, in mixed-lane tiles (64 tiles cover every `x` once per `g`) |
| Realistic distributions | Capture `(x, g)` pairs from a real model's tanh/sigmoid backward (any small LLM or MLP with tanh/GELU-tanh) and report the fraction of lanes in the tail and in each failure class |
| Special values | ±0 signs, ±inf, NaN, max finite, smallest normal `g`; currently out of scope and unspecified |
| ULP histogram, not just pass | Signed ULP counts and exact %, per kernel (already computed ad hoc in Runs 4–5; make it part of `tail.log`) |
| Throughput | Tiles/s per chip under multi-core, next to single-core cycles/tile |
| Instructions and registers | Still `unmeasured`; dump the SFPU disassembly from the JIT cache to count instructions per element |

### 2.4 Numerics that are still open

1. **Subnormal outputs.** The compute path flushes FP32 subnormals (format probe).
   Decide between (a) a device contract with `flush_output_subnormals = true`,
   reported alongside the default, or (b) a reconstruction that writes BF16
   subnormal bit patterns directly at pack time. Do not change the contract just
   to pass tests.
2. **Smallest-normal corruption.** `compute_copy(0x0080)` returns `0x00c0`. Probe
   every exponent-field-0/1/2 pattern through copy, multiply and pack to find the
   rule. Any kernel whose outputs land in that binade depends on it.
3. **Host model versus SFPMAD.** The host `std::fma` adapter matches the device on
   4,630/4,636 in-scope outputs. A bit-exact SFPMAD model would make host
   exhaustive sweeps trustworthy for the device as well.
4. **Soundness.** `claimed_sound` stays false. The exponent-equivariance argument
   in `numerical_semantics.md` is the path to a proof that would cover all `g`
   without enumeration.

## 3. Two tracks and their order

The PLDI 2027 deadline is 12 November 2026 (`research_plan_pldi2027.md`). That is
about six weeks from now. Paper evidence and upstream readiness need different
things.

### Track A: paper (by early November)

The plan requires ≥2 functions on hardware, a synthesis method rather than one
kernel, and comparison against equally accurate baselines.

| Priority | Item | Exit criterion |
|---:|---|---|
| A1 | Environment script and pin-worktree script (§2.1) | A fresh shell reproduces Run 5 with one command |
| A2 | Sigmoid on device with the same generated recipe | Sigmoid tail packet with correctness and cycles against the same comparator set |
| A3 | Main's kernel and a copy-only floor in the harness (§2.2 items 3–4) | The tables show shipped kernel, floor and candidate under identical dataflow |
| A4 | Mixed-lane exhaustive x-sweep (§2.3) | Correctness on all 65,536 `x` for the fixed `g` set, mixed lanes |
| A5 | Variance on 3 chips and 2 reservations | Medians with ranges in every cycle table |
| A6 | Subnormal decision and the smallest-normal probe (§2.4 items 1–2) | A written device contract the tables are graded against |
| A7 | Second function family beyond the exp tail (erf or ELU) through the same synthesis | Evidence that the method generalizes, which the positioning requires |

### Track B: upstream replacement for `tanh_bw` (after the paper, or in parallel if staffed)

| Priority | Item | Exit criterion |
|---:|---|---|
| B1 | Full-domain kernel with per-lane selection (§2.2 item 2) | Correct on every BF16 `x` under the chosen contract; cycles on small, tail and mixed tiles |
| B2 | Port the host backend to `MeshDevice` APIs (§2.1) | Harness builds against current `main` without deprecated calls |
| B3 | TTNN op integration, multi-core, fast dispatch (§2.2 items 5–6) | Op-profiler comparison with `ttnn.tanh_bw` on the same tensors |
| B4 | Realistic-distribution benchmark (§2.3) | End-to-end op time on captured activations, plus the fraction of lanes where `main` is wrong |
| B5 | tt-metal tests | Accuracy test (pytest, BF16 exhaustive x) and perf test in tt-metal's own suites |

Only B1–B4 can support a statement like "faster than tt-metal's `tanh_bw`".
Until then, speedups apply to tail inputs only.

## 4. What to focus on when developing new approaches

Lessons from Runs 1–5, written as rules.

1. **Compare against the strongest equally accurate *and* equally fused
   baseline, inside one packet.** The wh4 cubic looked 35% faster than split4.
   The fused vendor kernel, with the same single SFPU pass, was 14% faster still. The
   gain was fusion, not the method. Keep `vendor_fused` (or its successor) in every
   packet, and attribute gains only against it.
2. **Separate pipeline changes from algorithm changes.** wh5's `copy_init`
   removal gave every fused kernel about 40 cycles/tile. Land pipeline changes in
   all comparators at once, and report algorithm deltas within a packet.
3. **Gate timing on correctness.** A fast kernel that returns false zeros is not
   a baseline. Keep the wh3+ rule: no timing summary unless the candidate sweep
   passes.
4. **Do not trust the host model for device numerics.** Split4 passed 4,636 on
   the host and 3,798 on the device. Run the format probe in every packet, and
   treat any intermediate that can enter the FP32 subnormal or smallest-normal
   range on the device as suspect.
5. **Never materialize the tiny quantity.** Every kernel that formed `sech²(x)`
   (vendor, TTNN, materialized) lost outputs once it underflowed. Carry tiny scales
   as integer exponents and apply them last; this is the core of the method.
6. **Count what the SFPU actually does.** The cheapest wins so far came from fewer
   FMAs (quadratic versus cubic), fewer `*_init` switches (fusion) and fewer `v_if`
   regions (saturation in place of branchy repair). Before writing a variant,
   estimate FMAs, init switches and conditional regions per element, then confirm
   with cycles and, once available, instruction counts.
7. **Make exactness a reported cost.** Shorter polynomials trade exact outputs for
   cycles (98.8% → 88.6% → 81.9% from degree 4 to 2). State the contract the
   kernel is optimized for, and show the ULP histogram next to cycles.
8. **Check where the inputs are.** Tail-only wins matter only in proportion to
   tail lanes in real workloads. Measure the input distribution before optimizing
   a region further.
9. **Measure the dispatch cost early.** A path that is fast in isolation can
   lose its advantage once lanes must choose between paths. Prototype per-lane
   selection with the cheapest available inner paths before refining either one.
10. **Keep each claim within its evidence.** Label host and device results.
    Do not call device flush "FTZ" until it is isolated. Keep `claimed_sound`
    false until proved. Do not present scale separation as novel by itself
    (`AGENTS.md`).
11. **Commit every packet, including failures.** Runs 2 and 3 left their tail
    evidence in ignored files. Use `results/runs/` packets with provenance for
    everything that a table cites.

## 5. Per-run checklist

Before a device run:

- [ ] `source tt/scripts/env_wormhole.sh` (once added); `TT_METAL_HOME` is the pin worktree
- [ ] Pin worktree at `tt/pin/tt-metal.COMMIT`, clean, built with the profiler on
- [ ] Repository tree clean (the script enforces this)
- [ ] Candidate host sweep and `ctest` pass

After the run:

- [ ] Packet under `results/runs/` has provenance, build log, correctness CSV, raw profiler, cycles and format probe
- [ ] Medians recomputed from the raw profiler (independent script) and launch count checked
- [ ] Comparator table includes fused vendor, main's kernel (once added) and the floor kernel (once added)
- [ ] ULP histogram and failure classes recorded per kernel
- [ ] `research/device_bringup_wormhole.md` updated, and the packet committed and pushed
