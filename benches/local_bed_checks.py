"""Exercise local range planning and copying against the warm MSI dependencies."""
import subprocess
from build_cached_probe import build, root

text = (root/'src/shared/files.rs').read_text()
start = text.index('struct PlannedLocalSource')
end = text.index('\nfn remote_pgen_block_size', start)
local = text[start:end]
test_start = text.index('    fn planned_local_reads_reassemble_ranges_and_reject_truncation')
test_end = text.index('\n    }', test_start) + len('\n    }')
local += '\n#[test]\n' + text[test_start:test_end].replace('        use super::*;\n', '')
source = '''
pub mod pipeline_error { pub use gnomon::pipeline_error::*; }
#[allow(dead_code)]
#[path="%s"] mod range_fetch;
mod local {
use gnomon::pipeline_error::PipelineError;
use gnomon::files::ByteRangeSource;
use crate::range_fetch::{BedReadPlan, PlannedReader};
use std::sync::Arc;
%s
}
''' % (root/'src/shared/range_fetch.rs', local)
(root/'local_bed_tests.rs').write_text(source)
build(root/'local_bed_tests.rs', 'local-bed-tests', ['--test', '-C', 'panic=abort', '-Z', 'panic-abort-tests'])
subprocess.run([str(root/'local-bed-tests'), 'local_', '--test-threads', '2'], check=True, timeout=10)
