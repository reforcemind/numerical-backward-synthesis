# PLDI 2027 research plan

## Positioning

**Not:** “An optimized Tenstorrent tanh kernel.”

**Yes:** An approach for synthesizing low-precision backward programs whose intermediate representations may be numerically unsafe, using explicit scale-aware representations and verified transformations.

Tenstorrent SFPU provides a concrete accelerator where the problem matters and where generated programs can be evaluated at instruction/register/cycle level.

## Core research question

Can we automatically synthesize low-precision implementations of complete backward computations that:

1. preserve required numerical semantics,
2. avoid unsafe intermediate representations,
3. explicitly reason about significands and exponents,
4. satisfy a specified error/exception contract,
5. can be formally or exhaustively verified over the target domain,
6. and are selected using hardware-aware cost models?

Motivating fact (central, not a side anecdote):

> A derivative such as `sech²(x)` can underflow in the target format even when `g * sech²(x)` is representable.

This is a **representation** problem: a mathematically valid intermediate is not necessarily a valid computational representation in low precision.

## Contributions (target: 3-4)

1. **Numerical IR + synthesis framework** for low-precision backward computations under an explicit numerical contract.
2. **Scale-aware semantics-preserving transformations** that avoid materializing unsafe intermediate derivatives, with stated preconditions.
3. **Verification method**, including investigation of a sound reduction of the activation-gradient search space (hypothesis -> proof or delimited conditions).
4. **Hardware evaluation** on Tenstorrent SFPU across ≥2 backward functions (tanh, sigmoid; then erf/ELU).

## What is *not* claimed as novel

Scale separation / scaled special functions alone (GSL, etc.). Novelty requires the **combination** of backward-expression synthesis + representation constraints + scale-aware transforms + verification + hardware-aware selection - only if literature search supports the distinction (see `prior_art_matrix.md`).

## Phase gate (strict order)

| Phase | Gate |
|------:|------|
| 1 | Reproduce motivating case on host; hardware TBD |
| 2 | Hand scale-separated tanh: correctness + cost proxy -> **GO/NO-GO** |
| 3 | Minimal IR expresses the hand solution |
| 4 | Transforms + preconditions documented |
| 5 | Reproducible verifier + contract |
| 6 | Grammar search + cost selection |
| 7 | Sigmoid via same framework |
| 8 | Broader eval / paper |

**Phase 2 status (host):** GO - scale-separated matches contract; baseline shows false zeros.

## PLDI emphasis hierarchy

```
numerical specification
        ↓
representation-aware program synthesis
        ↓
semantics-preserving transformations
        ↓
verification
        ↓
target-specific implementation
```

## Explicit non-goals (unless later forced by experiment)

- LLVM/MLIR integration
- CUDA backend
- RL / LLM synthesis
- Large compiler infrastructure

## Risks (from overview; still active)

- Rescaling useful engineering without enough algorithmic novelty
- Verification reduction may not hold under real hardware semantics
- Extra exponent handling may erase performance benefits
- Tiny tail-gradient improvements ≠ training benefits
