# bw_kernel_syn

Representation-aware synthesis of numerically safe low-precision backward programs (C++20):
IR → transforms → verify → select. Host results are **host** / **host_sim**; device rows require
a pinned `tt-metal` board run (`docs/HARDWARE.md`). Scale separation alone is not the claim.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/export_results --out results
```

| Tool | Role |
|------|------|
| `bw_syn_cli` | `version` `doctor` `reproduce` `verify` `synth` `emit` `tt` |
| `export_results` | CSV under `results/` (host only; no fake device files) |
| `run_tt_harness` | `--device` / host CSV |

Docs: [`docs/HARDWARE.md`](docs/HARDWARE.md) · [`docs/numerical_semantics.md`](docs/numerical_semantics.md) · [`AGENTS.md`](AGENTS.md)
