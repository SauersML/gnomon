#!/usr/bin/env python3
"""Compare two gnomon .sscore files for a parity check.

Identifier and count columns (FID, IID, *_CT, *MISSING*) must match exactly, under the
same header and in the same row order. Every other column is a score and must agree
within 1e-10 relative, with an absolute floor of 1e-12 times the column's largest
magnitude so near-zero values are not held to an impossible bound.

Before comparing, the script confirms that it rejects a planted 1e-8 relative change to
a score and a count that is off by one, so the check cannot pass vacuously.
"""

import math
import sys
from pathlib import Path

RELATIVE = 1e-10
FLOOR = 1e-12


def read(path):
    lines = [line for line in Path(path).read_text().splitlines() if not line.startswith("##")]
    if not lines:
        raise SystemExit(f"{path} has no header")
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]
    for number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise SystemExit(f"{path} row {number} has {len(row)} fields; the header has {len(header)}")
    return header, rows


def is_exact(name):
    return name.lstrip("#") in ("FID", "IID") or name.endswith("_CT") or "MISSING" in name


def as_float(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def mismatches(a, b):
    (header_a, rows_a), (header_b, rows_b) = a, b
    if header_a != header_b:
        return [f"header: {header_a} vs {header_b}"]
    if len(rows_a) != len(rows_b):
        return [f"row count: {len(rows_a)} vs {len(rows_b)}"]
    problems = []
    for column, name in enumerate(header_a):
        pairs = [(x[column], y[column]) for x, y in zip(rows_a, rows_b)]
        if is_exact(name):
            problems += [f"{name} row {r}: {x} vs {y}" for r, (x, y) in enumerate(pairs) if x != y]
            continue
        floats = [(as_float(x), as_float(y)) for x, y in pairs]
        scale = max((abs(v) for pair in floats for v in pair if v is not None), default=0.0)
        for r, ((x, y), (p, q)) in enumerate(zip(pairs, floats)):
            if p is None or q is None:
                if x != y:
                    problems.append(f"{name} row {r}: {x} vs {y}")
            elif abs(p - q) > max(RELATIVE * max(abs(p), abs(q)), FLOOR * scale):
                problems.append(f"{name} row {r}: {x} vs {y}")
    return problems


def self_check(table):
    header, rows = table
    scores = [(r, c) for c, name in enumerate(header) if not is_exact(name)
              for r, row in enumerate(rows) if as_float(row[c])]
    counts = [c for c, name in enumerate(header) if name.endswith("_CT")]
    if not scores or not counts:
        raise SystemExit(f"self-check needs a nonzero score and a count column; header is {header}")
    r, c = scores[0]
    perturbed = [row[:] for row in rows]
    perturbed[r][c] = repr(float(rows[r][c]) * (1 + 1e-8))
    shifted = [row[:] for row in rows]
    shifted[0][counts[0]] = str(int(rows[0][counts[0]]) + 1)
    for label, planted in (("a 1e-8 relative score change", perturbed), ("a count off by one", shifted)):
        if not mismatches(table, (header, planted)):
            raise SystemExit(f"self-check failed: the comparison accepted {label}")


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: compare_sscore.py EXPECTED.sscore ACTUAL.sscore")
    expected, actual = read(argv[1]), read(argv[2])
    self_check(actual)
    problems = mismatches(expected, actual)
    if problems:
        print(f"{argv[1]} and {argv[2]} differ in an id or count, or beyond {RELATIVE:g} relative:")
        print("\n".join(problems[:20]))
        return 1
    print(f"{len(actual[1])} rows agree: ids and counts exact, scores within {RELATIVE:g} relative.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
