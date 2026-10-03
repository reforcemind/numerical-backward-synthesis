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

## Fused exponential-product candidate (wh4)

`detail::tail_exp_product<Ops,Degree,Rate,Scale>` shares one arithmetic body
between the IEEE FP32 host adapter and the Wormhole SFPI adapter. It approximates
`g * 2^Scale * exp(-Rate*abs(x))` with the following schedule:

1. Reduce `t=-Rate*abs(x)` to `k*ln(2)+r`, using an integer rounding step and
   a two-part `ln(2)` constant.
2. Evaluate a degree-three or degree-four Taylor polynomial on `|r| <= .347`.
   The constant terms are raised to `1.0009` and `1.00006`, respectively.
3. Replace the exponent of normal `g` by 127, multiply its signed significand
   by the polynomial, and round that bounded product to BF16 precision.
4. Compose the product exponent with the original gradient exponent, `k`, and
   `Scale`. Reconstruct only the final value. No tiny derivative is materialized.

For tanh, `(Rate,Scale)=(2,2)` and `4 <= |x| <= 88.5`. For sigmoid,
`(Rate,Scale)=(1,0)` and `8 <= |x| <= 177`. The denominator `(1+exp(t))^2` is
omitted. Only finite normal BF16 gradients and **normal BF16 reference outputs**
are covered. Tanh has a device implementation; sigmoid currently has host checks
only. These are specialized candidates, not full-domain replacements or generated
lowerings of the existing synthesis IR. The output restriction is a verification
scope, not a runtime dispatch test. Outside that scope, results are unspecified.

The real-arithmetic Taylor remainder bounds on this interval are less than
`0.000855` (cubic) and `0.0000594` (quartic). The upward offsets exceed these
bounds. Including the omitted denominator gives conservative relative error
bounds below `0.003156` and `0.000841` in real arithmetic. FP32 evaluation,
coefficient rounding, target instructions, and final BF16 rounding must still be
checked separately; these bounds are not device correctness proofs.

### Repair at the rounded normal boundary

Reconstructing an FP32 subnormal before rounding can lose an output that ought
to round to BF16 minimum normal. Rounding the bounded significand first avoids
most such failures. A remaining case occurs when the composed biased exponent
is zero and the rounded normalized magnitude is at least `1.9921875`: the
adapter returns signed minimum normal. This accounts for the coarser spacing on
the BF16 subnormal side. It can promote a true subnormal reference to normal;
it is valid only within the stated normal-reference-output scope. Tests cover
both signs and gradients immediately around the boundary for every tested `x`.

### Evidence and limits

`verify_tail_product OUT_CSV` enumerates every positive BF16 `x` in the specified
interval and every positive normal BF16 `g`, filtering by the rounded oracle
output. On the recorded Windows GCC 13.2 Release build:

| Function | Checks per degree | Degrees | Failures | Maximum BF16 ULP |
|----------|------------------:|---------|---------:|-----------------:|
| tanh backward | 12,768,182 | 3 and 4 | 0 | 1 |
| sigmoid backward | 12,624,474 | 3 and 4 | 0 | 1 |

These 50,785,312 **host** checks are against the independent scaled-f64 oracle,
not correctly rounded real arithmetic. Negative inputs/gradients have selected
regression tests, not a second exhaustive sweep. `--quick` is a sampled CI gate.
The full CSV and source hashes are in `results/host/tail_product/`.

The shared body prevents separately maintained formulas from drifting, but
`std::fma` is not a bit-exact model of
[Wormhole SFPMAD](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SFPMAD.md),
which is partially fused. Device compilation, numerical behavior, cycles, and
register pressure remain unmeasured for this candidate.

### Attribution and next research gate

The device experiment compares the candidate to split4, the materialized helper,
the vendor derivative primitive, and the pinned vendor tail polynomial fused
with `g` in the same traversal. All paths use the same reader and buffer capacity.
`BW_SYN_CB_TILES=1` versus `2` isolates buffer capacity, though both use the new
dual-read reader. A 50% cycle reduction means a factor-two throughput increase;
it is a target, not a prediction or measured result.

Range reduction, polynomial approximation, exponent reconstruction, and fusion
are established techniques. [MegaLibm](https://arxiv.org/abs/2311.01515) already
supports composable math-library implementations and synthesis;
[RLibm](https://people.cs.rutgers.edu/~sn349/papers/rlibm32-pldi-2021-preprint.pdf)
already targets final rounding constraints when constructing polynomials.
This candidate alone does not establish a new PLDI contribution.

The stronger direction is a synthesis rule for complete backward products with
an exponent-equivariance proof: reduce interior gradient exponents to significands
and verify boundary cases separately. The proof must cover signs, rounding,
normal boundaries, and target arithmetic, followed by generated schedules for
multiple functions and hardware comparison against equally accurate baselines.
That rule is not proved or implemented here; `claimed_sound` remains false.

## Generated quadratic with contract-specific reconstruction (wh5)

`synthesize_tail_product` now searches a bounded program family: one quadratic
shared by tanh and sigmoid, with either wh4 boundary repair or signed
minimum-normal saturation. It starts from the second-order Taylor coefficients
of `2^r`, generates final-product constraints, and uses cyclic projections onto
those linear intervals. It does not find a global optimum or prove that a
rejected polynomial family is infeasible.

### Product constraints and acceptance

For every activation encoding, the generator visits all 128 normal gradient
significands at biased exponent 254. If even this gradient produces a subnormal
reference, that pair cannot enter the normal-output contract at lower exponents.
For a normal reference with positive BF16 code `b`, the allowed output codes are
`b-1`, `b`, and `b+1`. The midpoints between `b-2`/`b-1` and `b+1`/`b+2` bound the
corresponding real-valued product interval. Endpoints are moved inward by a
relative margin of `1e-6` before fitting.

Writing the normalized gradient as `m`, and using the FP32 range reduction
`z=(-Rate*abs(x))*log2(e)`, `k=round(z)`, `r=z-k`, the candidate product at this
gradient exponent is `m*p(r)*2^(k+Scale+127)`. Inverting that expression maps each
allowed product interval to a linear constraint on the polynomial coefficients.
The generator intersects the intervals over all significands at each activation.
There are 1,123 resulting activation/function constraints across tanh and sigmoid.

These constraints guide search; they do not certify FP32 evaluation or establish
a sound exponent reduction. Acceptance separately enumerates every BF16 gradient
encoding and both signs of every activation encoding in the declared tail ranges.
It checks all normal reference outputs against the existing scaled-f64 oracle.
Only a passing full sweep emits `tail_exp2_coefficients.hpp`; sampled runs never
emit a selected header. `claimed_sound=false` remains unchanged.

The generated shared coefficients, after 12 projection sweeps, are:

```text
p(r) = (0x1.ebaceep-3 * r + 0x1.657c1cp-1) * r + 0x1.ff9dap-1
```

The shared arithmetic body uses two polynomial FMAs and one residual subtraction.
The wh4 cubic uses three polynomial FMAs and two FMAs for residual reduction.
These are source-operation counts, not measured instructions or cycles.

### Why reconstruction is part of the search

For the generated quadratic, retaining wh4's boundary repair causes false zeros.
The alternative reconstructs a normal result when its composed exponent is
positive and otherwise returns signed BF16 minimum normal. For a positive normal
reference `R >= N` and a nonnegative approximation `Y < N`, clamping to `N` cannot
increase BF16 ULP distance: `round(Y) <= N <= R`, and positive BF16 codes are
monotonic. The negative case follows by magnitudes if the sign is preserved.
This local observation does not prove the polynomial's approximation error.

| Host variant | Tanh checks | Sigmoid checks | Failed tanh / sigmoid |
|--------------|------------:|---------------:|----------------------:|
| Generated quadratic + wh4 repair | 51,072,728 | 50,497,896 | 36 / 36 false zeros |
| Same quadratic + normal saturation | 51,072,728 | 50,497,896 | 0 / 0; maximum 1 ULP |

The selected policy deliberately changes outputs outside the normal-reference
scope. It does not implement gradual underflow or a full-domain activation
backward operation. The output-domain test remains a verification restriction,
not a runtime dispatch predicate. Both functions share the generated polynomial
and arithmetic template; only tanh has a device path at present.

This ablation establishes a concrete interaction between approximation and
reconstruction in this bounded host experiment. Polynomial fitting, range
reduction, final-rounding constraints, and saturation are established techniques.
A new general synthesis or verification contribution still needs a precise
comparison with RLibm, MegaLibm, and Chassis, and target-level evidence. See
`prior_art_matrix.md`. The device experiment compares the generated quadratic
with the wh4 cubic in the same fused kernel, plus the existing baselines.

The full manifest and verification CSV are in `results/host/tail_synthesis`.
The verifier now includes false-zero distances when reporting maximum ULP;
previous reports could show `max_ulp=1` alongside much larger false-zero errors.
