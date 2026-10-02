# Prior-art matrix

Status: **related** / **orthogonal** / **baseline to cite** / **gap (this work)**.

Scale separation itself is **not** novel. The proposed contribution is a
contract-guided synthesis method for complete low-precision backward programs.
The differentiation below is a hypothesis pending same-task baseline experiments
and device evidence. Host candidates now pass sampled checks; the current cost
model remains a **host** proxy.

| Work | Venue / artifact | What it does | Overlap | Differentiation (this work) |
|------|------------------|--------------|---------|-----------------------------|
| [Herbie](https://herbie.uwplse.org/doc/latest/platforms.html) (Panchekha et al.) | PLDI 2015+ | Accuracy-driven FP expression rewriting; platform API permits custom representations, operations and costs | Can take the complete \(g\cdot f'\) expression and is a strong same-task baseline | Whether a configured Herbie platform can model Wormhole unpack/compute/pack stages and our output contract is an experiment, not an assumed gap |
| [Poseidon](https://2026.cgo.org/details/cgo-2026-papers/38/Thinking-Fast-and-Correct-Automated-Rewriting-of-Numerical-Code-through-Compiler-Aug) (Qian et al.) | CGO 2026 | Profile-guided numerical rewriting in a compiler with accuracy/performance tradeoffs | Compiler-scale candidate search and selection | Run the same backward expressions and target constraints before asserting a method difference; a backward-only input name does not establish novelty |
| RLibm / RLibm-prog / RLibm-all | POPL 2022, PLDI 2022 | Correctly rounded *elementary* libm via rounding-interval LP polynomials (often progressive / multi-format) | Low-prec math correctness | Synthesizes \(f(x)\) polynomials, not \(g\cdot f'(x)\) products; no unsafe-intermediate-derivative problem; not SFPU/backward IR |
| MegaLibm | math-lib synthesis | Range reduction + approx synthesis for libm-style functions | Approx / range reduction knobs | Same structural gap as RLibm: forward elementary functions, not backward products |
| Metalibm | codegen | Generator for elementary functions on CPU ISAs | Code generation | Hand/spec-driven elementary kernels; no contract synthesis over backward IR |
| GSL scaled specials | numerical recipes | Scaled exp/products for extended range | Scale-aware evaluation | Library primitives, not PL synthesis or contract verification |
| Daisy / Rosa / FPTuner | verified FP | Error bounds, precision allocation | Verification techniques to cite | Bound/tune existing expressions; do not emit scale-separated backward programs |
| Salsa / Precimonious | precision tuning | Mixed precision search | Orthogonal/related | Precision allocation ≠ representation-aware scale IR for underflows of \(f'\) |
| SLEEF / odd-even libm | optimized kernels | Hand-tuned math | Orthogonal | Deployment kernels, not synthesis |
| Autograd / JAX-XLA remat | AD systems | Graph rematerialization | Orthogonal | Graph-level; do **not** fix BF16 flush of \(f'\) before \(\times g\) |
| Scaled BLAS / extended exponents | numerics | Separate mantissa/exponent products | Representation idea | Not synthesis of backward programs under ULP/false-zero contracts |
| [TT-Metalium Wormhole tanh derivative](https://github.com/tenstorrent/tt-metal/blob/f9524a5f1b75180f00ce7413c2fc1c5cca1f29ee/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh_derivative.h) | source inspected at board pin `f9524a5f` | Piecewise polynomial and exponential tail for `sech²`; source saturates the derivative to zero for `|x|>=45` and reports one-BF16-ULP derivative accuracy | Strong existing derivative implementation | The target question is whether the complete product with `g` can be scheduled to keep a representable final gradient; source comments are not our device evidence |
| TTNN `tanh_bw` | production | Public backward operation; implementation may change by revision | **Baseline to measure** | Run the same inputs through TTNN on the pinned build before claiming any device mismatch or advantage |

## Concrete comparison checklist (camera-ready)

When writing Related Work, answer in one sentence each:

1. **Herbie:** Run the same complete backward expressions and BF16 input sample;
   include a custom platform when needed, and document supported operations,
   search settings, numerical outputs, and cost.
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
