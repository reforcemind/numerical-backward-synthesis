# bw_kernel_syn

Research prototype for representation-aware synthesis of low-precision backward
programs (C++20): IR → transforms → verify → select. Host results are **host** /
**host_sim**; device rows require a pinned `tt-metal` board run
(`tt/HARDWARE.md`). Scale separation alone is not the claim.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/export_results --out results
python experiments/scripts/check_oracle_mpmath.py  # optional: mpmath 1.3.0
```

| Tool | Role |
|------|------|
| `bw_syn_cli` | `version` `doctor` `reproduce` `verify` `synth` `emit` `tt` |
| `export_results` | CSV under `results/` (host only; no fake device files) |
| `run_tt_harness` | `--device` / host CSV |

The paper source stays local under `docs/` and is excluded from Git. Hardware runbook:
[`tt/HARDWARE.md`](tt/HARDWARE.md). Research notes:
[`research/numerical_semantics.md`](research/numerical_semantics.md).

When local paper source is present, build the anonymous review draft:

```text
cmake -E make_directory build/paper
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error '-outdir=../build/paper' paper.tex
```
The PLDI 2027 paper-format call is pending; this uses the PLDI 2026 ACM small
review style provisionally.

Current status: host synthesis checks 10 distinct IRs per function and selects
one of two sampled-contract-passing programs for each of tanh and sigmoid.
Selected programs pass fixed-axis and paired-bit BF16 sweeps; 952 finite probes agree with
an optional independent mpmath check. The SFPI emitter rejects unsupported
scale and rounding operations, so these programs are not device-linked. The
device tanh kernel is hand-written and unmeasured. Its ten-case harness is a
smoke test, not a PLDI hardware evaluation.
