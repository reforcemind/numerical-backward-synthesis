"""Optional independent high-precision check of exported host oracle probes."""

import argparse
import csv
import struct
from functools import cache
from pathlib import Path

import mpmath as mp


@cache
def bf16_value(bits: int) -> mp.mpf:
    return mp.mpf(struct.unpack("!f", struct.pack("!I", bits << 16))[0])


def round_bf16_positive(value: mp.mpf) -> int:
    if value == 0:
        return 0
    maximum = bf16_value(0x7F7F)
    overflow_midpoint = maximum + (maximum - bf16_value(0x7F7E)) / 2
    if value >= overflow_midpoint:
        return 0x7F80
    if value >= maximum:
        return 0x7F7F

    lo, hi = 0, 0x7F7F
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if bf16_value(mid) <= value:
            lo = mid
        else:
            hi = mid
    low_error = value - bf16_value(lo)
    high_error = bf16_value(hi) - value
    return (
        lo
        if low_error < high_error or (low_error == high_error and lo % 2 == 0)
        else hi
    )


def reference_bits(kind: str, x_bits: int, g_bits: int) -> int:
    if g_bits & 0x7FFF == 0:
        return g_bits
    x = bf16_value(x_bits)
    g = bf16_value(g_bits)
    if kind == "tanh_backward":
        e = mp.exp(-2 * abs(x))
        derivative = 4 * e / (1 + e) ** 2
    elif kind == "sigmoid_backward":
        e = mp.exp(-abs(x))
        derivative = e / (1 + e) ** 2
    else:
        raise ValueError(f"unsupported backward kind: {kind}")
    product = g * derivative
    sign = 0x8000 if product < 0 else 0
    return sign | round_bf16_positive(abs(product))


def ordered_bits(bits: int) -> int:
    return 0x8000 - (bits & 0x7FFF) if bits & 0x8000 else 0x8000 + bits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("results/paper/oracle_probes.csv"),
    )
    args = parser.parse_args()
    mp.mp.dps = 200
    count = 0
    disagreements = []
    candidate_failures = []
    with args.csv_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            x_bits = int(row["x_bits"], 16)
            g_bits = int(row["g_bits"], 16)
            expected = reference_bits(row["kind"], x_bits, g_bits)
            host = int(row["host_ref_bits"], 16)
            candidate = int(row["candidate_bits"], 16)
            if host != expected:
                disagreements.append(
                    (row["kind"], row["x_bits"], row["g_bits"], host, expected)
                )
            false_zero = candidate & 0x7FFF == 0 and expected & 0x7FFF != 0
            wrong_zero_sign = (
                candidate & 0x7FFF == 0
                and expected & 0x7FFF == 0
                and candidate != expected
            )
            wrong_infinity = (candidate & 0x7FFF == 0x7F80) != (
                expected & 0x7FFF == 0x7F80
            )
            if (
                abs(ordered_bits(candidate) - ordered_bits(expected)) > 1
                or false_zero
                or wrong_zero_sign
                or wrong_infinity
            ):
                candidate_failures.append(
                    (row["kind"], row["x_bits"], row["g_bits"], candidate, expected)
                )
            count += 1
    for kind, x_bits, g_bits, got, expected in disagreements[:10]:
        print(
            f"host reference mismatch {kind} x={x_bits} g={g_bits}: {got:#06x} != {expected:#06x}"
        )
    for kind, x_bits, g_bits, got, expected in candidate_failures[:10]:
        print(
            f"candidate mismatch {kind} x={x_bits} g={g_bits}: {got:#06x} vs {expected:#06x}"
        )
    print(
        f"mpmath dps=200 checked={count} host_reference_disagreements={len(disagreements)} "
        f"candidate_contract_failures={len(candidate_failures)}"
    )
    return 1 if disagreements or candidate_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
