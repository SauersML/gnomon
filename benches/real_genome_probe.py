"""One bounded full-marker MSI experiment per invocation; no truncated BIMs.

python3 benches/real_genome_probe.py {score,project} N {before,after} [SCORESET]
Use taskset externally. Inputs are existing public-reference/PGS fixtures.
"""
from pathlib import Path
import hashlib
import json
import mmap
import shutil
import subprocess
import sys
import time

root = Path('/projects/standard/hsiehph/sauer354/gnomon/target/score-map')
data = Path('/scratch.global/sauer354/gnomon-swarm/data')
task, n, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
scoreset = sys.argv[4] if len(sys.argv) > 4 else 'PGS000018'
assert task in ['score', 'project'] and label in ['before', 'after']
if task == 'score':
    source = data / ('score/src/array3200' if n <= 3200 else 'score/panels/large204800')
    weights = data / ('score/scoresets/' + scoreset if not scoreset.startswith('PGS') else 'score/pgs/txt/' + scoreset + '_hmPOS_GRCh38.txt')
else:
    source = data / ('map/real/bed1x/ref1kg_gsa' if n <= 3200 else 'map/real/bed32x/ref1kg_gsa_32x' if n <= 102400 else 'map/real/bed141x/ref1kg_gsa_141x')
case = '{}-n{}-{}'.format(task, n, scoreset if task == 'score' else 'reference')
work = root / 'real-genome' / case
work.mkdir(parents=True, exist_ok=True)
prefix = work / 'panel'
fam = source.with_suffix('.fam').read_bytes().splitlines(keepends=True)
assert n <= len(fam)
stride = (len(fam) + 3)//4
size = source.with_suffix('.bed').stat().st_size
variants = (size - 3)//stride
assert 3 + variants*stride == size
if not prefix.with_suffix('.bed').exists():
    prefix.with_suffix('.bim').symlink_to(source.with_suffix('.bim'))
    if n == len(fam):
        prefix.with_suffix('.bed').symlink_to(source.with_suffix('.bed'))
        prefix.with_suffix('.fam').symlink_to(source.with_suffix('.fam'))
    else:
        with source.with_suffix('.bed').open('rb') as src, prefix.with_suffix('.bed').open('wb') as dst:
            mapped = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
            dst.write(mapped[:3])
            for variant in range(variants):
                start = 3 + variant*stride
                dst.write(mapped[start:start+(n+3)//4])
            mapped.close()
        prefix.with_suffix('.fam').write_bytes(b''.join(fam[:n]))
if task == 'project' and not prefix.with_suffix('.hwe.json').exists():
    prefix.with_suffix('.hwe.json').symlink_to(data / 'map/real/model/all_hg38.hwe.json')
binary = root/'real-genome'/label/('gnomon-score' if task == 'score' else 'gnomon-map')
if task == 'score':
    command = [str(binary), str(weights), str(prefix)]
    suffix = weights.stem
    outputs = [work/('panel_' + suffix + '.sscore')]
    for checkpoint in work.glob('panel_' + suffix + '.sscore.gnomon-checkpoint*'):
        checkpoint.unlink()
else:
    command = [str(binary), 'project', str(prefix)]
    outputs = [work/'panel.projection_scores.bin', work/'panel.projection_scores.metadata.json']
for output in outputs:
    if output.exists(): output.unlink()
log_path, rss_path = work/(label+'.log'), work/(label+'.rss')
start = time.monotonic()
with log_path.open('wb') as log:
    result = subprocess.run(['timeout','--kill-after=2','35','/usr/bin/time','-f','%M','-o',str(rss_path)]+command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
record = dict(task=task, samples=n, variants=variants, scoreset=scoreset, label=label,
              wall_s=time.monotonic()-start, returncode=result.returncode,
              rss_kib=rss_path.read_text().strip() if rss_path.exists() else None)
if result.returncode == 0:
    primary = outputs[0]
    record['sha256'] = hashlib.sha256(primary.read_bytes()).hexdigest()
    shutil.copyfile(primary, work/(label + primary.suffix))
log = log_path.read_text()
record['stages'] = [line for line in log.splitlines() if any(s in line.lower() for s in ['time:', 'took ', 'complete in', 'backend', 'storage:', 'overlapping', 'principal components'])]
(work/(label+'.json')).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
if result.returncode: print(log[-2500:], flush=True)
sys.exit(result.returncode)
