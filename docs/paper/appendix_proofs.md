# Appendix: numerical semantics and proof obligations

## A. Motivating calculation

Let \(x=45\), \(g=4\). Then

\[
\mathrm{sech}^2(x)=4e^{-2x}/(1+e^{-2x})^2 \approx 3.277605\times 10^{-39}.
\]

BF16 min normal \(= 2^{-126}\approx 1.175494\times 10^{-38}\). Product \(\approx 1.311042\times 10^{-38}\) rounds (RNE) to BF16 bits `0x008f` (host confirmed).

If the derivative is flushed to zero before multiply, the output is \(0\) - a false zero relative to \(\mathrm{round}(g f'(x))\).

## B. ScaleSeparateMul - sufficient conditions (sketch)

**Setup.** Working real arithmetic \(\mathbb{R}\) with a final rounding map \(\mathrm{fl}:\mathbb{R}\to\mathrm{BF16}\) (RNE/RTZ/RU/RD). Let \(g,d\in\mathbb{R}\) be finite and nonzero. Write frexp forms \(g=m_g 2^{e_g}\), \(d=m_d 2^{e_d}\) with \(m_\bullet\in[1/2,1)\).

**Claim (identity before rounding).** \(g\cdot d = (m_g m_d) 2^{e_g+e_d}\).

**Claim (final rounding).** If the implementation returns \(\mathrm{fl}((m_g m_d)2^{e_g+e_d})\) with *no other rounding points*, then it equals \(\mathrm{fl}(g\cdot d)\).

**Failure modes (must be handled outside the claim).**

1. \(d\) rounded/flushed to BF16 before multiply (destroys claim).
2. Intermediate round of \(m_g m_d\) in a format that overflows/underflows differently than \(\mathrm{fl}(g\cdot d)\).
3. Subnormal \(g\) or \(d\): frexp still works mathematically, but hardware paths that flush subnormals break the premise.
4. Overflow of exponent sum in a bounded integer type.
5. Signed zero / NaN / Inf: excluded; use separate exception rules.

**Status:** sufficient-condition sketch for the research system; not a machine-checked proof.

## C. Verification reduction hypothesis

**Hypothesis.** For positive even \(f'\) (tanh/sigmoid) and implementations that are homogeneous in \(g\) up to final rounding, failures of scale-separated templates concentrate on:

- significand grids of \(|x|\) and \(|g|\),
- exponent bands near underflow/overflow,
- special values.

**Not proved.** The repository exposes `build_reduced_domain` with `claimed_sound = false`. A sound reduction must account for intermediate rounding in the concrete IR, output FTZ policy, and non-homogeneous approximations (polynomials).

**Proof obligation for a future theorem.**

> If \(P\) is drawn from template class \(\mathcal{T}\) with property \(\Phi\) (e.g., exact scaled-exp, final-only round, no FTZ mid-stream), then \(\mathrm{Verify}(P; X\times G)=\mathsf{true}\) iff \(\mathrm{Verify}(P; R)=\mathsf{true}\) for an explicitly constructed finite \(R\).

Until \(\Phi\) and \(R\) are fixed and proved, exhaustive or large boundary suites remain the trustworthy path.

## D. Hardware semantics obligation

Any paper table that attributes false zeros to a **device** kernel must include:

- pinned `tt-metal` commit,
- board (Wormhole/Blackhole),
- dtype path (BF16 tile formats),
- measured intermediate storage (DST vs BF16),
- FTZ / denormal behavior tests.

Host FTZ *models* are labeled as models.
