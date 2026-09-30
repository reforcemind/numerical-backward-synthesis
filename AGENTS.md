# Agent roles for this repository

Standing review roles. Prefer invoking them before claiming novelty, hardware
results, numerical guarantees, or merging non-trivial code.

| Role | Rule file | Mandate |
|------|-----------|---------|
| **PLDI Reviewer** | `.cursor/rules/pldi-reviewer.mdc` | Skeptical PC member; reject overclaims |
| **TT Systems Engineer** | `.cursor/rules/tt-systems-engineer.mdc` | Hardware path, SFPU, build/run on board |
| **Numerical Researcher** | `.cursor/rules/numerical-researcher.mdc` | Contracts, proofs, oracle honesty |
| **Code Standards** | `.cursor/rules/code-standards.mdc` | Ladder, reuse, dead code, format/lint |

Always-on baseline: `.cursor/rules/project-standards.mdc`  
Ladder details: `docs/agents/CODE_STANDARDS.md`

## How to use in Cursor

Ask explicitly, for example:

- “Act as the PLDI reviewer and critique the paper draft.”
- “Act as the TT systems engineer and harden the hardware harness.”
- “Act as the numerical researcher and audit the BF16 contract.”
- “Act as the code standards agent and clean this change.”

Or `@`-mention the corresponding rule if your Cursor build supports it.

## Non-negotiables (all roles)

1. Do **not** claim device FTZ / cycles / TTNN speedups without measured data on a pinned `tt-metal` revision.
2. Do **not** claim scale separation is novel by itself.
3. Do **not** mark verification-domain reduction as sound until proved (`claimed_sound` stays false).
4. Host results are labeled **host**; hardware results are labeled **device**.
5. Prefer C++ for research code; Python only for optional cross-checks.
6. Walk the **Ladder** before adding code (`docs/agents/CODE_STANDARDS.md`).

## Pull-and-run path (humans)

1. Host CI/smoke: `docs/HARDWARE.md` § Host
2. Device: `docs/HARDWARE.md` § Device (requires `TT_METAL_HOME`)
3. Standards: `experiments/scripts/check_standards.ps1` (or `.sh`)
