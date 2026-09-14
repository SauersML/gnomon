"""Test current scoring modules on MSI without rebuilding the dev dependency graph."""
import subprocess
from build_cached_probe import build, root

source = '#![feature(portable_simd)]\n'
source += '#[path="%s"] pub mod memory;\n' % (root/'src/shared/memory.rs')
source += 'pub use gnomon::{adapt_plink2, pipeline_error, output};\npub mod shared { pub use gnomon::files; }\n'
source += '#[allow(dead_code)]\n#[path="%s"] pub mod genotype_table;\n' % (root/'src/shared/genotype_table.rs')
for name in ['batch', 'io', 'pipeline', 'prepare']:
    source += '#[path="%s"] pub mod candidate_%s;\n' % (root/('src/score/'+name+'.rs'), name)
source += 'pub mod score { pub use gnomon::score::{types, kernel, complex, decide, reformat};\n'
for name in ['batch', 'io', 'pipeline', 'prepare']:
    source += 'pub use crate::candidate_%s as %s;\n' % (name, name)
source += '}\n'
(root/'preparation_memory_tests.rs').write_text(source)
tempfile = list((root/'target/release/build/tempfile').glob('*/out/libtempfile*.rmeta'))
assert len(tempfile) == 1, tempfile
extra = ['--test', '-C', 'panic=abort', '-Z', 'panic-abort-tests', '--extern', 'tempfile='+str(tempfile[0])]
for dependency in ['tempfile', 'rustix', 'linux-raw-sys', 'fastrand', 'getrandom', 'once_cell', 'errno', 'bitflags', 'libc']:
    for directory in (root/'target/release/build'/dependency).glob('*/out'):
        extra += ['-L', 'dependency='+str(directory)]
build(root/'preparation_memory_tests.rs', 'preparation-memory-tests', extra)
subprocess.run([str(root/'preparation-memory-tests'), '--test-threads', '4'], check=True, timeout=20)
