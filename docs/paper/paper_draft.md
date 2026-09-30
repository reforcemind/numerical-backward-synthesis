# Representation-Aware Synthesis of Numerically Safe Low-Precision Backward Programs

*Draft toward PLDI 2027.*

## Status

| Claim | Maturity |
|-------|----------|
| Host false-zero (45,4) → `0x008f` | demonstrated |
| Contract + verifier | implemented |
| Scale-aware IR + frexp host eval | implemented |
| Transforms with precondition flags | implemented |
| Synth IR mutations | implemented |
| Device measurement | **you run on board** — see `docs/HARDWARE.md` §2 |
| Host ablation + synth cost (tanh+sigmoid) | `results/paper/` via `export_results` |
| Verification-domain reduction | unproved hypothesis (`claimed_sound=false`) |

## Problem

Synthesize \(P\) under \(P(x,g)\approx\mathrm{round}(g\cdot f'(x))\) without materializing unsafe BF16 \(f'\).
Here \(\approx\) means the host contract vs `exact_product_f64`, not correctly rounded reals.

## Pointers

- Semantics: `docs/numerical_semantics.md`
- Proof sketch: `docs/paper/appendix_proofs.md`
- Prior art: `docs/prior_art_matrix.md`
- Hardware: `docs/HARDWARE.md`
- Tables: `./build/export_results --out results`
