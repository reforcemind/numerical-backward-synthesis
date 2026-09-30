# Numerical contract and semantics

## Central specification

For a backward map with mathematical derivative \(f'\),

\[
P(x,g) \approx \mathrm{round}\bigl(g \cdot f'(x)\bigr)
\]

The meaning of \(\approx\) is entirely defined by the `NumericalContract`
(`include/bw_syn/contract.hpp`): \(P\) must lie within `max_ulp_error` of the
**host** reference `round(exact_product_f64(...))` (f64 scaled formulas + RNE to BF16).
This is **not** a claim of correctly rounded real arithmetic.

## Default BF16 contract (`NumericalContract::bf16_default`)

| Field | Default | Meaning |
|-------|---------|---------|
| Input formats | BF16 | `x`, `g` already in BF16 |
| Output format | BF16 | |
| Rounding | RNE | `RoundMode::ToNearestEven` |
| Output subnormals | kept | `flush_output_subnormals = false` |
| Baseline derivative FTZ | on | **host model** of an unsafe intermediate path (not device FTZ) |
| Max ULP | 1 | finite vs host reference |
| Signed zero | required | when reference is ±0 |
| Exceptions | policy below | NaN / Inf handled explicitly |
| Domain | all finite BF16 | optional `max_abs_x` |

**We do not invent stronger guarantees than the implementation supports.** Device semantics
may differ and must be measured (`label=device`).

## Reference oracle

`exact_product_f64` evaluates \(g f'(x)\) with scaled formulas so the product remains
accurate even when \(f'(x)\) underflows BF16. Extreme tails where the **f64** formula
underflows yield a rounded reference of zero.

| kind | scaled product? | hard zero / f64 limit |
|------|-----------------|------------------------|
| tanh | yes | `|x|>300` → `copysign(0,g)` |
| sigmoid | yes | `|x|>700` → `copysign(0,g)` |
| erf | scaled via frexp on `g` | `exp` arg ≲ −700 → `copysign(0,g)` |
| elu | scaled via frexp on `g` for `x≤0` | same exp cutoff |

Functions:

- **tanh:** \(f'(x)=\mathrm{sech}^2(x)=4e^{-2|x|}/(1+e^{-2|x|})^2\)
- **sigmoid:** \(f'(x)=e^{-|x|}/(1+e^{-|x|})^2\)
- **erf:** \(f'(x)=(2/\sqrt{\pi})e^{-x^2}\)
- **ELU:** \(f'(x)=1\) if \(x>0\) else \(\alpha e^{x}\)

## Unsafe baseline semantics (host model)

1. Compute \(d = \mathrm{round}_{\mathrm{BF16}}(f'(x))\)
2. Optionally flush subnormals of \(d\) to zero (`baseline_flush_derivative_subnormals`)
3. Return \(\mathrm{round}_{\mathrm{BF16}}(d \otimes g)\)

This can produce **false zeros** when \(f'(x)\) underflows but \(g f'(x)\) does not.
This is a **model** of an unsafe intermediate path — not a measurement of device FTZ.

IR `ir_direct()` rounds \(f'\) to BF16 without an explicit Flush op; synthesis
`DirectMaterialize` **evaluation** uses `baseline_materialize_derivative` (FTZ model),
not IR round-only.

## Scale-separated intent

Host evaluator `detail::scale_separated_product` (not the oracle):

1. Compute \(f'(x)\) in f64 with a scaled formula (no BF16 store of \(f'\)).
2. `frexp(g)`, `frexp(f')` -> `(m_g,e_g)`, `(m_d,e_d)`.
3. `ScaleMul` / `AddExp` / `ldexp` / final `round_to_bf16`.

This is intentionally **independent** of `contract_reference` so verification is non-tautological.
Device SFPU may use a wider DST accum without explicit frexp; measure before claiming equivalence.

`TransformResult::sound_under_preconditions` is a **local rewrite flag** (structural match to
appendix B sketches: no prior BF16 of \(f'\`, final-round-only). It is **not** a proof.
Keep `ReducedDomain::claimed_sound=false` until a proof exists. Failure modes: premature
BF16/FTZ of \(f'\), mid-stream mantissa round, subnormal flush, exp-sum overflow, specials.

## Exceptional / edge cases (explicit)

- NaN in either input → NaN out
- `|x|=∞`, finite `g`, derivative → 0 (tanh/sigmoid/erf; ELU at −∞) → signed zero of `g`
- `|x|=+∞`, finite `g`, ELU → round(`g`)
- `|g|=∞` with vanishing derivative, or both infinite → NaN (avoid IEEE `0·∞`)
- Subnormals at output kept unless `flush_output_subnormals`
- Exponent overflow/underflow at reconstruction
- Derivative singularities (none for tanh/sigmoid; ELU kink at 0)

Scaling alone does **not** guarantee correct rounding if intermediate rounding is reintroduced.
