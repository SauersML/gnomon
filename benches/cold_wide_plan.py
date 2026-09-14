"""Compare uncached full-marker compilers on MSI using the existing warm artifacts."""
from build_cached_probe import build, root

# Add a benchmark entry point to an MSI-only source copy, without changing the
# production API or disabling content validation in the shipped cache path.
candidate = root / 'src/score/prepare_probe_candidate.rs'
candidate.write_text((root / 'src/score/prepare.rs').read_text() + '''
pub fn compile_uncached(prefixes: &[PathBuf], scores: &[PathBuf]) -> PreparationResult {
    prepare_for_computation_with_retry(prefixes, scores, None, None, 1).unwrap().0
}
''')
source = (root / 'src/benches/probes/wide_plan.rs').read_text()
source = source.replace('pub use gnomon::{pipeline_error, score};',
    'pub use gnomon::{pipeline_error, score, memory, output};\n'
    '#[path="%s"] pub mod candidate_prepare;' % candidate)
source = source.replace(
    'score::prepare::prepare_for_computation(&args[..1], &files, None, None).unwrap()',
    'candidate_prepare::compile_uncached(&args[..1], &files)')
probe = root / 'cold_wide_plan.rs'
probe.write_text(source)
try:
    build(probe, 'cold-wide-plan', ['-C', 'panic=abort'])
finally:
    candidate.unlink()
