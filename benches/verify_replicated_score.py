"""Compare a repeated reference cohort's complete score output with its seed."""
import csv
import math
import sys

with open(sys.argv[1]) as reference_file, open(sys.argv[2]) as actual_file:
    reference = csv.DictReader(reference_file, delimiter='\t')
    expected = {row['#IID']: row for row in reference}
    actual = csv.DictReader(actual_file, delimiter='\t')
    assert actual.fieldnames == reference.fieldnames
    count = 0
    maximum_error = 0.0
    for row in actual:
        seed, replicate = row['#IID'].rsplit('_t', 1)
        assert replicate.isdigit()
        baseline = expected[seed]
        for name in actual.fieldnames[1:]:
            if name.endswith('_MISSING_PCT'):
                assert row[name] == baseline[name]
            else:
                a, b = float(row[name]), float(baseline[name])
                assert math.isfinite(a) and math.isfinite(b)
                maximum_error = max(maximum_error, abs(a-b))
                assert abs(a-b) <= 2e-6 * max(1e-6, abs(a), abs(b)), (seed, name, a, b)
        count += 1
    assert count > len(expected) and count % len(expected) == 0
    print('verified_samples', count, 'max_abs_difference', maximum_error)
