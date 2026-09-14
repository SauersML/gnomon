"""Validate score precision on MSI against a successful warm Cargo library build."""
import argparse
import subprocess
from pathlib import Path
from build_cached_probe import build, root

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('library_log', type=Path)
parser.add_argument('mode', choices=['tests', 'cohort', 'plan', 'kernels'])
parser.add_argument('--candidate-prepare', action='store_true')
parser.add_argument('--candidate-batch', action='store_true')
parser.add_argument('--output', default='cohort-score-f64')
args = parser.parse_args()

if args.mode == 'kernels':
    build(root / 'src/benches/probes/exact_score_kernel.rs', 'exact-score-kernel',
          ['-C', 'panic=abort'], library_log=args.library_log)
elif args.mode == 'plan':
    build(root / 'src/benches/probes/exact_score_plan.rs', 'exact-score-plan',
          ['-C', 'panic=abort'], library_log=args.library_log)
elif args.mode == 'cohort':
    path = root / 'src/benches/probes/cohort_score.rs'
    if args.candidate_prepare or args.candidate_batch:
        modules = 'pub use gnomon::{adapt_plink2, memory, pipeline_error, output};\n'
        modules += 'pub mod shared { pub use gnomon::files; }\n'
        selected = ['prepare'] if args.candidate_prepare else []
        if args.candidate_batch:
            selected += ['batch', 'pipeline', 'io']
            modules += '#[allow(dead_code)] #[path="%s"] pub mod genotype_table;\n' % (root / 'src/shared/genotype_table.rs')
        for name in selected:
            modules += '#[path="%s"] pub mod candidate_%s;\n' % (root / ('src/score/' + name + '.rs'), name)
        modules += 'pub mod score { pub use gnomon::score::*;\n'
        for name in selected:
            modules += 'pub use crate::candidate_%s as %s;\n' % (name, name)
        modules += '}\n'
        source = '#![feature(portable_simd)]\n' + path.read_text().replace('use gnomon::score;', modules)
        path = root / 'cohort_score_precision_candidate.rs'
        path.write_text(source)
    build(path, args.output,
          ['-C', 'panic=abort'], library_log=args.library_log)
else:
    source = '#![feature(portable_simd)]\n'
    source += '#[path="%s"] pub mod memory;\n' % (root / 'src/shared/memory.rs')
    source += 'pub use gnomon::{adapt_plink2, pipeline_error, output};\n'
    source += 'pub mod shared { pub use gnomon::files; }\n'
    # This shared module also defines map-only constants outside this score harness.
    source += '#[allow(dead_code)] #[path="%s"] pub mod genotype_table;\n' % (root / 'src/shared/genotype_table.rs')
    modules = ['batch', 'io', 'pipeline', 'prepare', 'kernel']
    for name in modules:
        source += '#[path="%s"] pub mod candidate_%s;\n' % (root / ('src/score/' + name + '.rs'), name)
    source += 'pub mod score { pub use gnomon::score::{types, complex, decide, reformat};\n'
    for name in modules:
        source += 'pub use crate::candidate_%s as %s;\n' % (name, name)
    source += '}\n'
    path = root / 'precision_score_tests.rs'
    path.write_text(source)
    tempfile = list((root / 'target/release/build/tempfile').glob('*/out/libtempfile*.rmeta'))
    assert len(tempfile) == 1, tempfile
    extra = ['--test', '-C', 'panic=abort', '-Z', 'panic-abort-tests', '--extern', 'tempfile=' + str(tempfile[0])]
    for dependency in ['tempfile', 'rustix', 'linux-raw-sys', 'fastrand', 'getrandom', 'once_cell', 'errno', 'bitflags', 'libc']:
        for directory in (root / 'target/release/build' / dependency).glob('*/out'):
            extra += ['-L', 'dependency=' + str(directory)]
    build(path, 'precision-score-tests', extra, library_log=args.library_log)
    subprocess.run([str(root / 'precision-score-tests'), '--test-threads', '4'], check=True, timeout=25)
