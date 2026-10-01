# Prior-art matrix

Status: **related** / **orthogonal** / **baseline to cite** / **gap (this work)**.

Scale separation itself is **not** novel. The proposed contribution is a
contract-guided synthesis method for complete low-precision backward programs.
The differentiation below is a hypothesis pending same-task baseline experiments
and device evidence. Host candidates now pass sampled checks; the current cost
model remains a **host** proxy.

| Work | Venue / artifact | What it does | Overlap | Differentiation (this work) |
|------|------------------|--------------|---------|-----------------------------|
| [Herbie](https://herbie.uwplse.org/) (Panchekha et al.) | PLDI 2015+ | Accuracy-driven rewrite of FP *expressions* via sampling + rule search; later platform/ImplIR lowering | Expression rewrite, Pareto accuracy/cost | Targets general expressions vs reals, not **backward** \(g\cdot f'\) under a typed **numerical contract**, not BF16 intermediate false-zeros of \(f'\), not accelerator kernel selection |
| Poseidon (Qian et al.) | CGO 2026 / EGRAPHS 2026 | Profile-guided numerical rewriting *inside a production compiler*; composes Herbie-style algebraic rewrites + precision tuning; DP selects global tradeoffs | Compiler-scale rewrite, accuracy/perf frontier | Operates on profiled general FP subgraphs. Does **not** synthesize scale-separated **backward kernels**, expose frexp/ScaleMul IR, or verify against a backward-specific false-zero contract |
| RLibm / RLibm-prog / RLibm-all | POPL 2022, PLDI 2022 | Correctly rounded *elementary* libm via rounding-interval LP polynomials (often progressive / multi-format) | Low-prec math correctness | Synthesizes \(f(x)\) polynomials, not \(g\cdot f'(x)\) products; no unsafe-intermediate-derivative problem; not SFPU/backward IR |
| MegaLibm | math-lib synthesis | Range reduction + approx synthesis for libm-style functions | Approx / range reduction knobs | Same structural gap as RLibm: forward elementary functions, not backward products |
| Metalibm | codegen | Generator for elementary functions on CPU ISAs | Code generation | Hand/spec-driven elementary kernels; no contract synthesis over backward IR |
| GSL scaled specials | numerical recipes | Scaled exp/products for extended range | Scale-aware evaluation | Library primitives, not PL synthesis or contract verification |
| Daisy / Rosa / FPTuner | verified FP | Error bounds, precision allocation | Verification techniques to cite | Bound/tune existing expressions; do not emit scale-separated backward programs |
| Salsa / Precimonious | precision tuning | Mixed precision search | Orthogonal/related | Precision allocation ≠ representation-aware scale IR for underflows of \(f'\) |
| SLEEF / odd-even libm | optimized kernels | Hand-tuned math | Orthogonal | Deployment kernels, not synthesis |
| Autograd / JAX-XLA remat | AD systems | Graph rematerialization | Orthogonal | Graph-level; do **not** fix BF16 flush of \(f'\) before \(\times g\) |
| Scaled BLAS / extended exponents | numerics | Separate mantissa/exponent products | Representation idea | Not synthesis of backward programs under ULP/false-zero contracts |
| [TT-Metalium Wormhole tanh derivative](https://github.com/tenstorrent/tt-metal/blob/9ac55ec9be762d2ce46eec193299eb34cff2f8d7/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh_derivative.h) | source inspected at `9ac55ec9` | Piecewise polynomial and exponential tail for `sech²`; source comments report one-BF16-ULP derivative accuracy | Strong competing derivative kernel | Compare the complete `g·sech²(x)` result and cost on the pinned revision; the source comment is not our device evidence |
| TTNN `tanh_bw` | production | Public backward operation; implementation may change by revision | **Baseline to measure** | Run the same inputs through TTNN on the pinned build before claiming any device mismatch or advantage |

## Concrete comparison checklist (camera-ready)

When writing Related Work, answer in one sentence each:

1. **Herbie:** Run the same complete backward expressions and BF16 input sample;
   document supported operations, search settings, numerical outputs, and cost.
2. **Poseidon:** Compare the same expressions if its compiler pipeline supports
   the target; document any unsupported representation or target constraint.
3. **RLibm:** Test whether a better elementary-function approximation changes
   final-gradient accuracy when the intermediate derivative is materialized.

## Tentative novelty (use only if table rows stay accurate)

> We explore contract-gated backward IR templates (scale vs materialize,
> rounding/normalization variants), compare final outputs with a host reference,
> and rank sampled candidates by a host cost proxy. Selected tanh and sigmoid
> host IRs pass boundary and fixed-axis BF16 checks. Whole-pair verification,
> device-aware selection, and hardware FTZ claims require further work.

## Search log

| Query | Used for |
|-------|----------|
| Poseidon CGO 2026 numerical rewriting | Row above; PDF [ece.is](https://ece.is/assets/pdf/poseidon-cgo26.pdf) |
| Herbie floating point expression rewriting | Row above; [herbie.uwplse.org](https://herbie.uwplse.org/) |
| RLibm correctly rounded elementary polynomials | Row above; POPL/PLDI RLibm line |
| low-precision backward-expression synthesis | No direct hit matching this combination (gap support) |
| scale-aware floating-point computation synthesis | Hits scaled BLAS / GSL; not backward IR synthesis |

Update this file if a reviewer points to a closer system; do not weaken the differentiation bullets without evidence.
