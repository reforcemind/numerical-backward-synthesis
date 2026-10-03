"""Measure ttnn.tanh_bw on a bw_syn tail sweep: correctness on every tile, device cycles on the timed set.

Cross-check only (device, TTNN). Inputs, oracle bits and scope come from a bw_syn tail.csv.
Timing gives every core exactly the recorded timed inputs, so per-core cycles/tile are
comparable in tile count with the single-core bw_syn zones (not in dispatch or core count).
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ordered(bits: int) -> int:
    return -(bits & 0x7FFF) if bits & 0x8000 else bits


def is_nonfinite(bits: int) -> bool:
    return (bits & 0x7F80) == 0x7F80


def passes(got: int, oracle: int) -> bool:
    # Same rule the bw_syn contract applies to normal finite references: <= 1 BF16 ulp.
    if is_nonfinite(got):
        return False
    return abs(ordered(got) - ordered(oracle)) <= 1


def tiles_from_bits(bits: list[int]) -> torch.Tensor:
    t = torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return t.repeat_interleave(1024).reshape(len(bits) * 32, 32)


def run(device, xs: list[int], gs: list[int]) -> list[int]:
    x = ttnn.from_torch(tiles_from_bits(xs), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(tiles_from_bits(gs), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.tanh_bw(g, x)
    ttnn.synchronize_device(device)
    host = ttnn.to_torch(out[0]).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    host = host.reshape(len(xs), 1024)
    lanes_equal = (host == host[:, :1]).all(dim=1)
    if not bool(lanes_equal.all()):
        raise RuntimeError("non-uniform lanes in a constant tile")
    return [int(v) for v in host[:, 0]]


def kernel_spans(log: Path) -> dict[int, dict[tuple[str, str], int]]:
    """run host ID -> core -> TRISC-KERNEL span (earliest TRISC start to latest TRISC end)."""
    rows = list(csv.reader(open(log)))
    header = [h.strip() for h in rows[1]]
    col = {name: header.index(name) for name in header}
    opened: dict = {}
    per: dict = defaultdict(lambda: defaultdict(lambda: [None, None]))
    for r in rows[2:]:
        r = [c.strip() for c in r]
        if len(r) < len(header) or r[col["zone name"]] != "TRISC-KERNEL":
            continue
        run_id = int(r[col["run host ID"]])
        core = (r[col["core_x"]], r[col["core_y"]])
        key = (run_id, core, r[col["RISC processor type"]])
        tick = int(r[col["time[cycles since reset]"]])
        if r[col["type"]] == "ZONE_START":
            opened[key] = tick
        else:
            start = opened.pop(key)
            span = per[run_id][core]
            span[0] = start if span[0] is None else min(span[0], start)
            span[1] = tick if span[1] is None else max(span[1], tick)
    return {rid: {core: e - s for core, (s, e) in cores.items()} for rid, cores in per.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tail-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measured", type=int, default=20)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TT_METAL_PROFILER_DIR", str(out / "profiler"))
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        raise SystemExit("set TT_METAL_DEVICE_PROFILER=1 before starting Python")

    cases = list(csv.DictReader(open(a.tail_csv)))
    xs = [int(r["x_bits"], 16) for r in cases]
    gs = [int(r["g_bits"], 16) for r in cases]
    timed = [i for i, r in enumerate(cases) if r.get("timed_input") == "1"]

    device = ttnn.open_device(device_id=a.device_id)
    try:
        got = run(device, xs, gs)
        ttnn.ReadDeviceProfiler(device)
        grid = device.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        tx = [xs[i] for i in timed] * cores
        tg = [gs[i] for i in timed] * cores
        for _ in range(a.warmup + a.measured):
            run(device, tx, tg)
            ttnn.ReadDeviceProfiler(device)
    finally:
        ttnn.close_device(device)

    in_scope = n_pass = n_zero = n_wrong = 0
    with open(out / "ttnn_tail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "x_bits", "g_bits", "oracle_bits", "ttnn_bits", "scope", "pass", "timed_input"])
        for r, bits in zip(cases, got):
            oracle = int(r["oracle_bits"], 16)
            scoped = r["scope"] == "finite_normal_output"
            ok = scoped and passes(bits, oracle)
            if scoped:
                in_scope += 1
                n_pass += ok
                if not ok:
                    if bits & 0x7FFF == 0:
                        n_zero += 1
                    else:
                        n_wrong += 1
            w.writerow(["ttnn_device", r["x_bits"], r["g_bits"], r["oracle_bits"], f"0x{bits:04x}",
                        r["scope"], ("1" if ok else "0") if scoped else "", r.get("timed_input", "")])

    log = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "profile_log_device.csv"
    spans = kernel_spans(log)
    measured_ids = sorted(spans)[-a.measured:]
    per_tile = [span / len(timed) for rid in measured_ids for span in spans[rid].values()]
    per_launch_median = [statistics.median(spans[rid].values()) / len(timed) for rid in measured_ids]
    n_cores = {len(spans[rid]) for rid in measured_ids}
    summary = {
        "in_scope": in_scope, "pass": n_pass, "false_zero": n_zero, "wrong_nonzero": n_wrong,
        "cores_per_launch": sorted(n_cores), "tiles_per_core": len(timed),
        "median_cycles_per_tile_all_cores": round(statistics.median(per_tile), 1),
        "min_core_cycles_per_tile": round(min(per_tile), 1),
        "p95_cycles_per_tile": round(sorted(per_tile)[int(0.95 * len(per_tile)) - 1], 1),
        "median_of_launch_medians": round(statistics.median(per_launch_median), 1),
    }
    with open(out / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")
    for k, v in summary.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
