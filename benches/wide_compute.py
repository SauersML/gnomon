"""Build a paired full-marker pipeline probe using the warm pre-change library."""
from build_cached_probe import build, root

source = (root / 'src/benches/probes/wide_plan.rs').read_text()
modules = 'pub use gnomon::{adapt_plink2, memory, output, pipeline_error};\npub mod shared { pub use gnomon::files; }\n'
modules += '#[allow(dead_code)] #[path="%s"] pub mod genotype_table;\n' % (root / 'src/shared/genotype_table.rs')
for name in ['batch', 'io', 'pipeline', 'prepare']:
    modules += '#[path="%s"] pub mod candidate_%s;\n' % (root / ('src/score/' + name + '.rs'), name)
modules += 'pub mod score { pub use gnomon::score::{types, kernel, checkpoint, complex, decide, reformat};\n'
for name in ['batch', 'io', 'pipeline', 'prepare']:
    modules += 'pub use crate::candidate_%s as %s;\n' % (name, name)
modules += '}\n'
source = '#![feature(portable_simd)]\n' + source.replace('pub use gnomon::{pipeline_error, score};', modules)
source = source.replace('before_prepare::prepare_for_computation(', 'gnomon::score::prepare::prepare_for_computation(')
source = source.replace('let before_context = score::pipeline::PipelineContext::new', 'let before_context = gnomon::score::pipeline::PipelineContext::new')
source = source.replace('let before_output = score::pipeline::run(&before_context).unwrap();',
    'let before_start = Instant::now();\nlet before_output = gnomon::score::pipeline::run(&before_context).unwrap();\nlet before_compute = before_start.elapsed();')
source = source.replace('assert_eq!(before_output, after_output);', '''
println!("timing_before_ms={:.3} timing_after_ms={:.3}", before_compute.as_secs_f64()*1000.0, compute.as_secs_f64()*1000.0);
assert_eq!(before_output.1, after_output.1);
let mut max_abs = 0.0f64;
let mut max_relative = 0.0f64;
for (&left, &right) in before_output.0.iter().zip(&after_output.0) {
    assert!(left.is_finite() && right.is_finite());
    let error = (left - right).abs();
    let relative = error / left.abs().max(right.abs()).max(1e-6);
    max_abs = max_abs.max(error);
    max_relative = max_relative.max(relative);
    assert!(relative <= 2e-6, "before={left} after={right} relative={relative}");
}
println!("before_compute_ms={:.3} max_abs={max_abs:.12e} max_relative={max_relative:.12e}", before_compute.as_secs_f64()*1000.0);
''')
probe = root / 'wide_compute.rs'
probe.write_text(source)
build(probe, 'wide-compute', ['-C', 'panic=abort'])
