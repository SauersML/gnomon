"""Create deterministic, irregular keep lists from the full-marker MSI seed."""
from pathlib import Path

root = Path('/projects/standard/hsiehph/sauer354/gnomon/target/score-map')
data = Path('/scratch.global/sauer354/gnomon-swarm/data/score')
ids = [line.split()[1] for line in (data / 'src/array3200.fam').read_text().splitlines()]
for count, step, wobble in [(512, 6, 5), (65, 47, 7)]:
    indices = [i * step + i % wobble for i in range(count)]
    assert indices == sorted(set(indices)) and indices[-1] < len(ids)
    (root / f'round12-keep{count}.txt').write_text(''.join(ids[i] + '\n' for i in indices))
single = root / 'real-genome/round12-single-score'
single.mkdir(exist_ok=True)
weights = data / 'pgs/txt/PGS000018_hmPOS_GRCh38.gnomon.sorted.gnomon.tsv'
target = single / weights.name
if not target.is_symlink():
    target.symlink_to(weights)
else:
    assert target.resolve() == weights.resolve()
