"""Run only on MSI: bounded exact-kernel checks against an independent Fraction oracle."""
from fractions import Fraction
from pathlib import Path
import math
import random
import struct
import subprocess

root = Path('/projects/standard/hsiehph/sauer354/gnomon/target/score-map')
source = root / 'src/benches/probes/exact_score_safety.rs'
flags = ['rustc', '--edition=2024', '-C', 'opt-level=3', '-C', 'target-cpu=x86-64-v3',
         '-C', 'codegen-units=4', str(source)]
tests = root / 'exact-score-safety-tests'
oracle = root / 'exact-score-rounding'
subprocess.run(flags + ['--test', '-o', str(tests)], check=True, timeout=20)
subprocess.run([str(tests), '--test-threads=4'], check=True, timeout=15)
subprocess.run(flags + ['-o', str(oracle)], check=True, timeout=20)

rng = random.Random(20260914)
cases = []
for scale, divisor in [(1, 1), (1, 3), (1 << 63, 2), (2**64 - 1, 2**64 - 1),
                       (2**63 + 1, 3), (3, 2**63 + 1)]:
    for value in [0, 1, -1, 2**127 - 1, -(2**127), 2**53 + 1, 2**54 - 1]:
        for exponent in [-1200, -1075, -1074, -1022, -64, 0, 960, 1024]:
            cases.append((value, exponent, scale, divisor))
for _ in range(30_000):
    value = rng.getrandbits(rng.randint(1, 127)) * rng.choice([-1, 1])
    scale = max(1, rng.getrandbits(rng.randint(1, 64)))
    divisor = max(1, rng.getrandbits(rng.randint(1, 64)))
    cases.append((value, rng.randint(-1200, 1100), scale, divisor))
payload = ''.join(' '.join(map(str, case)) + '\n' for case in cases)
result = subprocess.run([str(oracle)], input=payload, text=True, capture_output=True,
                        check=True, timeout=15)
actual = result.stdout.splitlines()
assert len(actual) == len(cases)
for case, received in zip(cases, actual):
    value, exponent, scale, divisor = case
    rational = Fraction(value, scale * divisor)
    rational *= Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)
    try:
        rounded = float(rational)
    except OverflowError:
        rounded = math.copysign(math.inf, value)
    expected = struct.pack('>d', rounded).hex()
    assert received == expected, (case, received, expected)
print(f'Independent exact-rational rounding: {len(cases)} cases matched bit for bit', flush=True)
