"""Build a cohort probe against current scoring modules and warm MSI dependencies."""
from build_cached_probe import build, root
import sys

if sys.argv[1:] == ['--baseline']:
    build(root / 'src/benches/probes/cohort_score.rs', 'cohort-score-baseline', ['-C', 'panic=abort'])
    raise SystemExit(0)

modules = 'pub use gnomon::{adapt_plink2, memory, output, pipeline_error};\npub mod shared { pub use gnomon::files; }\n'
modules += '#[allow(dead_code)] #[path="%s"] pub mod genotype_table;\n' % (root / 'src/shared/genotype_table.rs')
for name in ['batch', 'io', 'pipeline', 'prepare']:
    modules += '#[path="%s"] pub mod candidate_%s;\n' % (root / ('src/score/' + name + '.rs'), name)
modules += 'pub mod score { pub use gnomon::score::{types, kernel, complex, decide, reformat};\n'
for name in ['batch', 'io', 'pipeline', 'prepare']:
    modules += 'pub use crate::candidate_%s as %s;\n' % (name, name)
modules += '}\n'
source = (root / 'src/benches/probes/cohort_score.rs').read_text()
source = '#![feature(portable_simd)]\n' + source.replace('use gnomon::score;', modules)
probe = root / 'cohort_score_candidate.rs'
probe.write_text(source)
build(probe, 'cohort-score-candidate', ['-C', 'panic=abort'])
