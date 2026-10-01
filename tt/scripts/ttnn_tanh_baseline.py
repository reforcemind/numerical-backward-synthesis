"""Measure TTNN tanh backward on the same BF16 tiles as the Metal smoke test."""

import argparse
import csv
import math
import os
import statistics
import subprocess
import time
from pathlib import Path


def read_device_cases(path: Path, arch: str, commit: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or any(
        row["label"] != "device"
        or row["arch"] != arch
        or row["tt_metal_commit"] != commit
        for row in rows
    ):
        raise ValueError(
            "expected ten device rows from this architecture and tt-metal commit"
        )
    return rows


def ulp_distance(a: int, b: int) -> int:
    def ordered(bits: int) -> int:
        return (~bits & 0xFFFF) if bits & 0x8000 else (bits | 0x8000)

    return abs(ordered(a) - ordered(b))


def is_nan(bits: int) -> bool:
    return (bits & 0x7F80) == 0x7F80 and (bits & 0x007F) != 0


def is_inf(bits: int) -> bool:
    return (bits & 0x7FFF) == 0x7F80


def is_zero(bits: int) -> bool:
    return (bits & 0x7FFF) == 0


def passes_contract(got: int, oracle: int) -> bool:
    if is_nan(oracle):
        return is_nan(got)
    if is_inf(oracle):
        return got == oracle
    if is_nan(got) or is_inf(got):
        return False
    if is_zero(got) and not is_zero(oracle):
        return False
    if is_zero(got) and is_zero(oracle) and got != oracle:
        return False
    return ulp_distance(got, oracle) <= 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")

    home = os.environ.get("TT_METAL_HOME")
    arch = os.environ.get("BW_SYN_ARCH")
    if not home or arch not in ("wormhole", "blackhole"):
        parser.error("TT_METAL_HOME and BW_SYN_ARCH=wormhole|blackhole are required")
    commit = subprocess.check_output(
        ["git", "-C", home, "rev-parse", "HEAD"], text=True
    ).strip()
    cases = read_device_cases(args.device_csv, arch, commit)

    try:
        import torch
        import ttnn
    except ImportError as exc:
        raise SystemExit(
            "TTNN baseline requires torch and ttnn from the pinned source build"
        ) from exc

    xs = torch.tensor([float(row["x"]) for row in cases], dtype=torch.bfloat16)
    gs = torch.tensor([float(row["g"]) for row in cases], dtype=torch.bfloat16)
    x_tiles = xs.repeat_interleave(1024).reshape(1, 1, len(cases) * 32, 32)
    g_tiles = gs.repeat_interleave(1024).reshape(1, 1, len(cases) * 32, 32)

    device = ttnn.open_device(device_id=args.device_id)
    try:
        x_device = ttnn.from_torch(
            x_tiles, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        g_device = ttnn.from_torch(
            g_tiles, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        for _ in range(args.warmup):
            ttnn.tanh_bw(g_device, x_device)
            ttnn.synchronize_device(device)

        times_ns = []
        output = None
        for _ in range(args.repeat):
            start = time.perf_counter_ns()
            output = ttnn.tanh_bw(g_device, x_device)
            ttnn.synchronize_device(device)
            times_ns.append(time.perf_counter_ns() - start)

        if not isinstance(output, (list, tuple)) or len(output) != 1:
            raise RuntimeError("ttnn.tanh_bw did not return one output tensor")
        host = ttnn.to_torch(output[0])
        if host.dtype != torch.bfloat16 or host.numel() != len(cases) * 1024:
            raise RuntimeError("TTNN output is not the expected ten BF16 tiles")
        bits = host.contiguous().reshape(-1, 32, 32).view(torch.int16)
        got_bits = [int(bits[i, 0, 0].item()) & 0xFFFF for i in range(len(cases))]
        nonuniform = [
            int((bits[i] != bits[i, 0, 0]).sum().item()) for i in range(len(cases))
        ]
    finally:
        ttnn.close_device(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    module_path = Path(ttnn.__file__).resolve()
    source_root = Path(home).resolve()
    module_under_source = (
        module_path == source_root or source_root in module_path.parents
    )
    with (args.out_dir / "ttnn_tanh_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label",
                "arch",
                "tt_metal_commit",
                "ttnn_module_under_source",
                "x",
                "g",
                "oracle_bits",
                "ttnn_bits",
                "ulp",
                "pass",
                "exact_match",
                "false_zero",
                "nonuniform_lanes",
            ]
        )
        for row, got, nonuniform_lanes in zip(cases, got_bits, nonuniform):
            oracle = int(row["oracle_bits"], 16)
            ulp = ulp_distance(got, oracle)
            false_zero = (
                is_zero(got)
                and not is_zero(oracle)
                and not (is_nan(oracle) or is_inf(oracle))
            )
            writer.writerow(
                [
                    "ttnn_device",
                    arch,
                    commit,
                    int(module_under_source),
                    row["x"],
                    row["g"],
                    row["oracle_bits"],
                    f"0x{got:04x}",
                    ulp,
                    int(passes_contract(got, oracle) and nonuniform_lanes == 0),
                    int(got == oracle),
                    int(false_zero),
                    nonuniform_lanes,
                ]
            )

    with (args.out_dir / "ttnn_tanh_latency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label",
                "arch",
                "tt_metal_commit",
                "scope",
                "tiles",
                "warmup",
                "iteration",
                "ns",
            ]
        )
        for index, elapsed in enumerate(times_ns):
            writer.writerow(
                [
                    "ttnn_device",
                    arch,
                    commit,
                    "host_enqueue_to_sync",
                    len(cases),
                    args.warmup,
                    index,
                    elapsed,
                ]
            )

    sorted_ns = sorted(times_ns)
    p95_ns = sorted_ns[math.ceil(0.95 * len(sorted_ns)) - 1]
    with (args.out_dir / "ttnn_tanh_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label",
                "arch",
                "tt_metal_commit",
                "scope",
                "tiles",
                "warmup",
                "repeat",
                "median_ns",
                "p95_ns",
                "ttnn_module_path",
                "ttnn_module_under_source",
            ]
        )
        writer.writerow(
            [
                "ttnn_device",
                arch,
                commit,
                "host_enqueue_to_sync",
                len(cases),
                args.warmup,
                args.repeat,
                statistics.median(times_ns),
                p95_ns,
                module_path,
                int(module_under_source),
            ]
        )
    print(
        f"TTNN device output: {len(cases)} tiles; host enqueue-to-sync median {statistics.median(times_ns)} ns"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
