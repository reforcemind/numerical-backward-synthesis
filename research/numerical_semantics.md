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
| Scoped domain | off | `min_abs_x`, `finite_inputs_only`, and `normal_reference_output_only` can restrict an exploratory candidate; skipped pairs are reported |

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

IR `ir_direct()` rounds \(f'\) to BF16 and applies an explicit `FlushBF16` node.
Synthesis verifies this same IR, so cost and numerical checks refer to one program.

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
- Finite `x`, `|g|=∞`, nonzero derivative → signed infinity (for ELU the sign also depends on `alpha` when `x≤0`)
- `|g|=∞` with a zero derivative, or infinite `x` with a vanishing derivative → NaN for `0·∞`
- ELU at `x=+∞` has derivative one, so infinite `g` passes through
- Subnormals at output kept unless `flush_output_subnormals`
- Exponent overflow/underflow at reconstruction
- Derivative singularities (none for tanh/sigmoid; ELU kink at 0)

Scaling alone does **not** guarantee correct rounding if intermediate rounding is reintroduced.

## Exploratory tanh tail schedule

For finite BF16 `g`, let `a=|x|` and `q=exp(-a/2)`. The experimental tail
candidate evaluates `g*q`, multiplies by 4, then multiplies by `q` three more
times. It is evaluated only for `a>=4` and **normal** BF16 reference outputs.
The exact tanh backward product is `4*g*q^4/(1+q^4)^2`; the candidate omits
the denominator. Its relative overestimate is at most
`2*exp(-8)+exp(-16)<0.000672` on this domain, before operation and exponential
approximation error. This bound alone does not prove the device's 1-ULP contract.

For a normal output and a normal, finite BF16 gradient, `q` and every scheduled
FP32 product are normal in ideal arithmetic. The output condition and the BF16
maximum bound `a` to approximately 88.7, where `q` is far above FP32's minimum
normal. At `a>=4`, `4*q<1`, so the first two operations do not overflow and
all later products decrease toward the normal output. This is a range argument
for the schedule, not a guarantee about the Wormhole exponential or packer.
The scoped IR candidate is `tanh_bw::ir_tail_split4`; synthesis includes it only
when the contract requests `min_abs_x>=4`, finite inputs, and normal reference
outputs. The hand-written device candidate is in
`tt/kernels/compute/tanh_bw_tail_split4.cpp`; the independent FP32 host
calculation is `tanh_bw::tail_split4_host`. Source correspondence and device
semantics are still unverified. The normal-output filter is a verification
scope, not an implemented runtime dispatch predicate.

BF16 subnormal outputs remain outside this candidate's declared domain. The
`--format-probe` hardware path compares raw BF16 transport, compute copy, and
normal-times-normal subnormal production. Do not infer a device FTZ stage from
the prior tanh run before this probe is measured.

The synthesis verifier evaluates candidate IR on exceptional inputs instead of
substituting the reference. Tanh and sigmoid candidates carry a
`PositiveDerivativeInf` guard: because their analytic derivatives are strictly
positive for finite \(x\), infinite \(g\) retains its sign even when an f64
exponential underflows. Two distinct IRs per function pass the full default
boundary sample, and the selected IR passes two 196,608-pair fixed-axis BF16
sweeps plus three 16-bit pairing permutations (196,608 pairs). The 952 finite
high-precision probes checked by
`experiments/scripts/check_oracle_mpmath.py` agree with the f64 reference;
none of these checks covers every \((x,g)\) pair. The standalone host helper has
separate exception handling and must be reported separately. IR evaluation uses the contract's
rounding mode, but its transcendental operations are still f64 host approximations;
directed-rounding guarantees over real arithmetic are not established.
