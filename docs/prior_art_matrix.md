# Prior-art matrix

Status: **related** / **orthogonal** / **baseline to cite** / **gap (this work)**.

Scale separation itself is **not** novel. Claims rest on the *combination*: contract-gated
enumeration of low-precision **backward** IR templates (scale vs materialize, round/norm
variants), semantics-checked transforms, non-tautological verification, and selection by a
**host** cost proxy (device selection deferred until measured rows exist).

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
| TTNN / tt-metal `unary_backward` | production | Device kernels; common pattern: materialize derivative then mul | **Baseline to measure** | Motivates the false-zero; this work synthesizes/verifies alternatives. Device false-zero/FTZ numbers require a pinned `tt-metal` run (`docs/HARDWARE.md`) |

## Concrete comparison checklist (camera-ready)

When writing Related Work, answer in one sentence each:

1. **Herbie:** Would Herbie, given `g * sech2(x)` in BF16, invent frexp/ldexp scale separation and a final-only round policy under an explicit false-zero ban? (Expected: no; different search object and contract.)
2. **Poseidon:** Does Poseidon’s profile+DP pipeline emit a scale-aware backward IR or only rewrite/precision-tune existing graphs? (Expected: latter.)
3. **RLibm:** Does a correctly rounded `tanh`/`exp` help if the bug is flushing `sech2` before multiplying by `g`? (Expected: orthogonal; product structure is the issue.)

## Tentative novelty (use only if table rows stay accurate)

> We enumerate contract-gated backward IR templates (scale vs materialize, round/norm
> variants), verify against a non-tautological oracle (ULP + false-zero ban), and select by a
> **host** cost proxy. Device-aware selection and hardware FTZ claims are deferred until
> `label=device` CSV rows exist under a pinned `tt-metal` commit.

## Search log

| Query | Used for |
|-------|----------|
| Poseidon CGO 2026 numerical rewriting | Row above; PDF [ece.is](https://ece.is/assets/pdf/poseidon-cgo26.pdf) |
| Herbie floating point expression rewriting | Row above; [herbie.uwplse.org](https://herbie.uwplse.org/) |
| RLibm correctly rounded elementary polynomials | Row above; POPL/PLDI RLibm line |
| low-precision backward-expression synthesis | No direct hit matching this combination (gap support) |
| scale-aware floating-point computation synthesis | Hits scaled BLAS / GSL; not backward IR synthesis |

Update this file if a reviewer points to a closer system; do not weaken the differentiation bullets without evidence.
