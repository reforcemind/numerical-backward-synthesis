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
| [RLibm32](https://people.cs.rutgers.edu/~sn349/papers/rlibm32-pldi-2021-preprint.pdf) and RLibm family | PLDI 2021 and related work | Correctly rounded elementary functions synthesized against final rounding intervals; accounts for numerical error in polynomial evaluation, range reduction, and output compensation; uses counterexample-guided polynomial generation and piecewise tables | Strong overlap in output-contract-guided polynomial synthesis, compensation-aware constraints, counterexample refinement, and piecewise approximation | The cited RLibm32 work targets elementary-function outputs. Whether its interval construction or related RLibm systems can be adapted to a complete backward product and BF16 intermediate representation contract needs direct comparison; this is an untested distinction, not an established gap |
| [MegaLibm](https://arxiv.org/abs/2311.01515) | 2023 preprint | Modular DSL for implementing, testing, and tuning math-library functions, including composable range reduction, approximation synthesis, and orthogonal tuning of precision and evaluation schemes | Substantial overlap in modular math implementation, approximation synthesis, and tuning for accuracy and speed | Evaluate whether MegaLibm can express and search the same complete backward products and target representation constraints; its existence alone does not establish a structural gap |
| [Chassis](https://arxiv.org/abs/2410.14025) | ASPLOS 2025 | Compiles mathematical expressions to target operators described by their real-expression approximation, estimated cost, and accuracy; iterative target-aware instruction selection optimizes speed and accuracy | Strong overlap in target-aware numerical compilation and cost/accuracy selection | Compare complete backward expressions under the BF16 contract and target unpack/compute/pack behavior; any difference in representation constraints or search space remains to be demonstrated |
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
3. **RLibm:** Test whether rounding-interval and compensation-aware synthesis
   can be adapted to the complete backward product, and whether changing only
   the elementary derivative approximation changes final-gradient accuracy.
4. **MegaLibm and Chassis:** Attempt to encode the same complete backward
   expressions, BF16 output contract, and target costs; report unsupported
   constructs and search outcomes before claiming a method distinction.

## Tentative novelty (use only if table rows stay accurate)

> **Hypothesis (not an established novelty claim):** jointly selecting a
> derivative polynomial and its reconstruction for a complete backward product
> against a BF16 contract may address cases not covered by the cited
> elementary-function and numerical-compilation systems. In current host
> experiments, a generated quadratic used 1,123 intersected constraints across
> 128 significands and 12 projection sweeps. The old boundary policy produced
> 36 false zeros per function. Signed-minimum-normal saturation had 0 failures
> and at most 1 ULP error over 51,072,728 tanh and 50,497,896 sigmoid
> normal-reference cases, including both input signs. These are sampled host
> results, not a proof or device measurements. Verification-domain reduction
> remains unproved (`claimed_sound=false`); device behavior is unmeasured.

## Search log

| Query | Used for |
|-------|----------|
| Poseidon CGO 2026 numerical rewriting | Row above; PDF [ece.is](https://ece.is/assets/pdf/poseidon-cgo26.pdf) |
| Herbie floating point expression rewriting | Row above; [herbie.uwplse.org](https://herbie.uwplse.org/) |
| RLibm correctly rounded elementary polynomials | Row above; POPL/PLDI RLibm line |
| MegaLibm implementation and synthesis of math library functions | Row above; [arXiv:2311.01515](https://arxiv.org/abs/2311.01515) |
| Chassis target-aware implementation of real expressions | Row above; [ASPLOS 2025 paper](https://arxiv.org/abs/2410.14025) |
| low-precision backward-expression synthesis | No direct hit matching this combination (gap support) |
| scale-aware floating-point computation synthesis | Hits scaled BLAS / GSL; not backward IR synthesis |

Update this file when direct same-task comparisons provide evidence about the remaining differentiation hypotheses.
