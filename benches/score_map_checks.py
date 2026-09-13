"""Compile focused production checks using an existing MSI release dependency graph."""
import subprocess
from build_cached_probe import args, build, root

def function(source, name):
    start=source.index('fn '+name)
    body=source.index('{',start)
    depth=1
    end=body+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}')
        end+=1
    return source[start:end]

kernel = (root/'src/score/kernel.rs').read_text().split('#[cfg(test)]')[0]
batch = root/'src/score/batch.rs'
project = (root/'src/map/project.rs').read_text()
projection = 'use std::simd::Simd; use std::sync::OnceLock; use rayon::prelude::*; use std::mem::size_of; use crate::genotype_table;\n'
for name in ['add_score_vector','packed_score_pair_tables','accumulate_packed_scores','packed_bytes_per_variant','plink_missing_lane_masks','packed_tri_size','use_grouped_projection','accumulate_grouped_projection','count_packed_missing_calls','accumulate_packed_cpu_block_row_major_sparse_missing','accumulate_packed_cpu_block_row_major_dense_missing']:
    projection += function(project,name)+'\n'
projection += '#[test]\n'+function(project,'packed_projection_kernels_match_scalar_calls_and_missingness')+'\n#[test]\n'+function(project,'packed_missing_count_excludes_padding')
source = '#![feature(portable_simd)]\n#[path="'+str(root/'src/shared/genotype_table.rs')+'"] mod genotype_table;\npub mod score { pub mod types { pub use gnomon::score::types::*; }\npub mod kernel {\n'+kernel+'\n}\n#[path="'+str(batch)+'"] pub mod batch; }\nmod projection {\n'+projection+'\n}\n'
(root/'focused_tests.rs').write_text(source)
io = (root/'src/map/io.rs').read_text()
selection = 'use std::{fmt, str, path::{Path, PathBuf}}; use gnomon::files::TextSource; use gnomon::map::io::PlinkIoError; use gnomon::map::variant_filter::{VariantKey, VariantSelection, MatchKind};\n'
selection += function(io, 'select_plink_variant_records_by_keys') + '\n' + function(io, 'selected_model_key') + '\n'
selection += io[io.index('pub struct PlinkVariantRecordIter'):io.index('#[derive(Debug)]\npub struct PlinkVariantBlockSource')]
for name in ['from_source', 'path', 'line']:
    selection = selection.replace(function(selection, name), '')
selection += '\n#[test]\n' + function(io, 'packed_marker_selection_preserves_allele_priority_and_record_errors')
source += '\npub mod pipeline_error { pub use gnomon::pipeline_error::*; }\nmod marker_selection {\n' + selection + '\n}'
(root/'focused_tests.rs').write_text(source)
build(root/'focused_tests.rs','focused-tests',['--test','-C','panic=abort','-Z','panic-abort-tests'])
subprocess.run([str(root/'focused-tests'),'--test-threads','4'],check=True,timeout=30)
(root/'kernel_tests.rs').write_text('#![feature(portable_simd)]\n#[path="'+str(root/'src/score/kernel.rs')+'"] mod kernel;')
subprocess.run([args[0], '--edition=2024','--test','-C','opt-level=3','-C','target-cpu=x86-64-v3',str(root/'kernel_tests.rs'),'-o',str(root/'kernel-tests')],check=True,timeout=15)
subprocess.run([str(root/'kernel-tests')],check=True,timeout=10)
build(root/'src/cli/main.rs','after-map',['--cfg','feature="map"','-C','panic=abort'])
