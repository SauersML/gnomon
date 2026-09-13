from pathlib import Path
import array
import math
import shutil
import hashlib
import json
import mmap
import re
import statistics
import subprocess
import sys
import time

root = Path('/projects/standard/hsiehph/sauer354/gnomon/.validation/score-map-20260913')
large = '--large' in sys.argv or '--biobank' in sys.argv
single = '--single' in sys.argv
biobank = '--biobank' in sys.argv
missing_data = '--missing' in sys.argv
log_prefix = 'missing-' if missing_data else 'biobank-' if biobank else 'single-' if single else 'large-' if large else ''
work = root/('work-' + log_prefix.rstrip('-') if log_prefix else 'work')
work.mkdir(exist_ok=True)
for label, score, mapper in [('before',root/'baseline-score',root/'baseline-map'),('after',root/'target/release/gnomon-score',root/'after-map')]:
    (root/label).mkdir(exist_ok=True)
    for name,target in [('gnomon-score',score),('gnomon-map',mapper)]:
        link=root/label/name
        if not link.exists(): link.symlink_to(target)
source = Path('/scratch.global/sauer354/gnomon-swarm/data/map/synth/c50k_20k')
n = 1025 if missing_data else 500000 if biobank else 1 if single else 50000 if large else 16384
rows = Path(str(source)+'.bim').read_text().splitlines()[:8192 if large else 4096]
fam = Path(str(source)+'.fam').read_text().splitlines()
prefix = work/'panel'
stride = (len(fam)+3)//4
if not prefix.with_suffix('.bed').exists():
    with Path(str(source)+'.bed').open('rb') as infile, prefix.with_suffix('.bed').open('wb') as out:
        bed = mmap.mmap(infile.fileno(),0,access=mmap.ACCESS_READ)
        out.write(bed[:3])
        for j in range(len(rows)):
            row = bed[3+j*stride:3+(j+1)*stride]
            row = bytearray((row * ((n + len(fam) - 1)//len(fam)))[:(n+3)//4])
            if missing_data:
                for sample in range((-j*7) % 13, n, 13):
                    shift = (sample % 4) * 2
                    row[sample//4] = (row[sample//4] & ~(3 << shift)) | (1 << shift)
            out.write(row)
        bed.close()
    prefix.with_suffix('.bim').write_text('\n'.join(rows)+'\n')
    prefix.with_suffix('.fam').write_text(''.join('F{0} I{0} 0 0 0 -9\n'.format(i) for i in range(n)))
if (single or biobank or missing_data) and not prefix.with_suffix('.hwe.json').exists():
    shutil.copyfile(root/('work-large' if biobank else 'work')/'panel.hwe.json', prefix.with_suffix('.hwe.json'))
if not prefix.with_suffix('.hwe.json').exists():
    result=subprocess.run([str(root/'before/gnomon-map'),'fit','--components','4','--threads','4',str(prefix)],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=25)
    (root/'fit-fixture.log').write_text(result.stdout)
    if result.returncode: print(result.stdout[-5000:],flush=True)
    result.check_returncode()
weights=work/'weights.tsv'
k=9
with weights.open('w') as out:
    out.write('variant_id\teffect_allele\tother_allele\t'+'\t'.join('S'+str(i) for i in range(k))+'\n')
    for j,line in enumerate(rows[:8192 if large else 2048]):
        fields=line.split()
        out.write(fields[0]+':'+fields[3]+'\t'+fields[4]+'\t'+fields[5]+'\t'+'\t'.join(str(((j*13+i*7)%101-50)/1000) for i in range(k))+'\n')
results=[]
for task in (['score'] if '--score-only' in sys.argv else ['project'] if '--project-only' in sys.argv else ['score','project']):
    reference=None
    for rep in range(1 if '--once' in sys.argv else 3):
        for label in (['before','after'] if rep%2==0 else ['after','before']):
            for name in (['panel_weights.sscore'] if task=='score' else ['panel.projection_scores.bin','panel.projection_scores.metadata.json']):
                output_path=work/name
                if output_path.exists(): output_path.unlink()
            if task=='score':
                binary=root/label/'gnomon-score'
                command=[str(binary),str(weights),str(prefix)]
            else:
                binary=root/label/'gnomon-map'
                command=[str(binary),'project',str(prefix)]
            start=time.monotonic()
            rss_path=root/f'{log_prefix}{task}-{label}-{rep}.rss'
            command=['/usr/bin/time','-f','%M','-o',str(rss_path)]+command
            result=subprocess.run(command,cwd=work,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
            wall=time.monotonic()-start
            log=result.stdout+'\n'+result.stderr
            (root/f'{log_prefix}{task}-{label}-{rep}.log').write_text(log)
            if result.returncode:
                print(log[-5000:],flush=True)
                result.check_returncode()
            stage=re.findall(r'(?:Total pipeline time: |Projection compute time: )([^\n]+)',log)
            paths=sorted(work.glob('*.sscore' if task=='score' else '*.projection_scores.bin'))
            assert len(paths)==1,paths
            output=paths[0].read_bytes()
            digest=hashlib.sha256(output).hexdigest()
            if reference is None:
                reference=output
            numeric_error=0.0
            if task=='project':
                assert len(output)==len(reference) and output[:32]==reference[:32]
                extent=32+n*4*8
                assert output[extent:]==reference[extent:]
                a=array.array('d'); a.frombytes(reference[32:extent])
                b=array.array('d'); b.frombytes(output[32:extent])
                for x,y in zip(a,b):
                    assert math.isfinite(x) and math.isfinite(y)
                    numeric_error=max(numeric_error,abs(x-y))
                    assert abs(x-y)<=1e-10*(1+abs(x)),(x,y)
            elif output!=reference:
                for a,b in zip(reference.decode().splitlines()[1:],output.decode().splitlines()[1:]):
                    av=a.split(); bv=b.split()
                    assert len(av)==len(bv)
                    for x,y in zip(av,bv):
                        if x==y: continue
                        numeric_error=max(numeric_error,abs(float(x)-float(y)))
                assert numeric_error<1e-5,numeric_error
            entry=dict(max_rss_kib=int(rss_path.read_text().strip()),samples=n,task=task,label=label,rep=rep,wall_s=wall,compute=stage,sha256=digest,max_numeric_difference=numeric_error)
            results.append(entry)
            print(json.dumps(entry),flush=True)
(root/(log_prefix+'cli-results.json')).write_text(json.dumps(results,indent=2)+'\n')
for task in sorted(set(r['task'] for r in results)):
    before=statistics.median(r['wall_s'] for r in results if r['task']==task and r['label']=='before')
    after=statistics.median(r['wall_s'] for r in results if r['task']==task and r['label']=='after')
    print(f'{task}: before={before:.3f}s after={after:.3f}s speedup={before/after:.3f}x')
