"""Compare a cohort_score probe's sums with independent gscore averages."""
import argparse
import csv
import json
import math
import struct
from pathlib import Path

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('production', type=Path)
parser.add_argument('reference', type=Path)
parser.add_argument('--tolerance', type=float, default=1e-12,
                    help='relative tolerance (default: 1e-12)')
parser.add_argument('--absolute-tolerance', type=float, default=0.0,
                    help='absolute allowance near zero (default: 0)')
args = parser.parse_args()
assert math.isfinite(args.tolerance) and args.tolerance >= 0
assert math.isfinite(args.absolute_tolerance) and args.absolute_tolerance >= 0
metadata = json.loads(args.production.with_suffix('.meta.json').read_text())
names = metadata['scores']
denominators = metadata['denominators']
assert len(names) == len(denominators) and names
assert len(set(names)) == len(names), 'duplicate score name'
reference = {}
with args.reference.open() as stream:
    for row in csv.DictReader(stream, delimiter='\t'):
        assert row['#IID'] not in reference, 'duplicate reference IID'
        reference[row['#IID']] = row
maximum_absolute = 0.0
maximum_relative = 0.0
worst = None
worst_absolute = None
per_score = {name: {'max_absolute': 0.0, 'max_relative': 0.0, 'outside_tolerance': 0}
             for name in names}
outside_tolerance = 0
people = 0
with args.production.open() as stream:
    for row in csv.DictReader(stream, delimiter='\t'):
        person = row['IID']
        expected = reference.pop(person)
        people += 1
        for name, denominator in zip(names, denominators):
            missing = int(row[name + '_MISSING_COUNT'])
            assert 0 <= missing <= denominator
            percentage = float(expected[name + '_MISSING_PCT'])
            # gscore reports percentages in f32. Reproduce that representation
            # instead of pretending a rounded percentage is an exact count.
            expected_percentage = f32(f32(f32(missing) / f32(denominator)) * 100.0) if denominator else 0.0
            assert f32(percentage) == expected_percentage, (person, name, missing, percentage, expected_percentage)
            actual = float(row[name]) / (denominator - missing) if denominator > missing else 0.0
            target = float(expected[name + '_AVG'])
            assert math.isfinite(actual) and math.isfinite(target)
            absolute = abs(actual - target)
            relative = absolute / max(abs(actual), abs(target), 1e-30)
            if absolute > maximum_absolute:
                maximum_absolute = absolute
                worst_absolute = {'person': person, 'score': name, 'actual': actual, 'expected': target}
            if relative > maximum_relative:
                maximum_relative = relative
                worst = {'person': person, 'score': name, 'actual': actual, 'expected': target}
            outside = absolute > args.absolute_tolerance + args.tolerance * max(abs(actual), abs(target))
            outside_tolerance += outside
            metrics = per_score[name]
            metrics['max_absolute'] = max(metrics['max_absolute'], absolute)
            metrics['max_relative'] = max(metrics['max_relative'], relative)
            metrics['outside_tolerance'] += outside
assert people and not reference, 'person sets differ'
print(json.dumps({'people': people, 'scores': len(names), 'max_absolute': maximum_absolute,
                  'max_relative': maximum_relative, 'outside_tolerance': outside_tolerance,
                  'tolerance': args.tolerance, 'absolute_tolerance': args.absolute_tolerance,
                  'worst_relative': worst, 'worst_absolute': worst_absolute,
                  'per_score': per_score}, indent=2))
raise SystemExit(1 if outside_tolerance else 0)
