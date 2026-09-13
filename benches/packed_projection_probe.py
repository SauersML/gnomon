"""Bounded, dependency-free MSI probe of packed projection kernels.

Usage: python3 benches/packed_projection_probe.py BASELINE_PROJECT_RS OUTPUT_DIR
Run on an allocated CPU, using the same compiler flags for both implementations.
"""
from pathlib import Path
import subprocess
import sys

source = Path(__file__).resolve().parents[1]
baseline = Path(sys.argv[1])
output = Path(sys.argv[2])

def function(code, name):
    start = code.index('fn ' + name)
    body = code.index('{', start)
    depth, end = 1, body + 1
    while depth:
        depth += (code[end] == '{') - (code[end] == '}')
        end += 1
    return code[start:end]

modules = []
for name, path in [('before', baseline), ('after', source / 'map/project.rs')]:
    code = path.read_text()
    functions = ['add_score_vector', 'packed_score_pair_tables', 'accumulate_packed_scores',
                 'packed_bytes_per_variant', 'plink_missing_lane_masks', 'packed_tri_size',
                 'accumulate_packed_cpu_block_row_major_sparse_missing',
                 'accumulate_packed_cpu_block_row_major_dense_missing']
    if name == 'after':
        functions += ['use_grouped_projection', 'accumulate_grouped_projection']
    body = 'use std::simd::Simd; use std::sync::OnceLock;\n'
    if name == 'after':
        body += 'use std::mem::size_of; use crate::genotype_table;\n'
    for f in functions:
        inline = '#[inline(always)]\n' if f in ['add_score_vector', 'accumulate_packed_scores'] else ''
        body += inline + function(code, f).replace('fn ', 'pub fn ', 1).replace('.par_chunks_mut(', '.chunks_mut(') + '\n'
    body += 'pub ' + function(code, 'packed_projection_kernels_match_scalar_calls_and_missingness')
    modules.append('mod ' + name + ' {\n' + body + '\n}')

main = r'''
use std::hint::black_box;
use std::time::Instant;
fn next(seed: &mut u64) -> u64 { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17; *seed }
fn main() {
    before::packed_projection_kernels_match_scalar_calls_and_missingness();
    after::packed_projection_kernels_match_scalar_calls_and_missingness();
    println!("scalar score, missing-list and missing-information checks passed");
    let mut seed = 42;
    for (n,k) in [(1usize,4usize),(32,4),(1024,4),(50000,1),(50000,4),(50000,20),(50000,64),(250000,4)] {
        let m = 257;
        let data: Vec<Vec<u8>> = (0..m).map(|_| (0..n.div_ceil(4)).map(|_| {
            let mut b = 0;
            for lane in 0..4 {
                let r = next(&mut seed) % 10000;
                let code = if r < 3500 {0} else if r < 8200 {2} else if r < 9800 {3} else {1};
                b |= code << (lane * 2);
            }
            b
        }).collect()).collect();
        let bytes: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
        let vectors: Vec<f64> = (0..m*3*k).map(|i| ((i*37%127) as f64 - 43.0)/127.0).collect();
        let swaps: Vec<bool> = (0..m).map(|i| i%3 == 0).collect();
        for rep in 0..3 {
            let mut reference: Option<(Vec<f64>, Vec<Vec<u32>>)> = None;
            for (label, f) in [("before", before::accumulate_packed_cpu_block_row_major_sparse_missing as fn(&[&[u8]],&[f64],&[bool],usize,usize,usize,&mut[f64],&mut[Vec<u32>])),
                               ("after", after::accumulate_packed_cpu_block_row_major_sparse_missing)] {
                let mut scores = vec![0.0; n*k];
                let mut missing = vec![Vec::with_capacity(8); n];
                let start = Instant::now();
                f(black_box(&bytes), &vectors, &swaps, 17, 1024, k, &mut scores, &mut missing);
                println!("n={n} k={k} {label} rep={rep} ms={:.3}", start.elapsed().as_secs_f64()*1000.0);
                if let Some((expected, absent)) = &reference {
                    for (a,b) in scores.iter().zip(expected) { assert!((a-b).abs() < 1e-10); }
                    assert_eq!(&missing, absent);
                } else { reference = Some((scores, missing)); }
            }
        }
    }
}
'''
probe = output / 'packed_projection_probe.rs'
probe.write_text('#![feature(portable_simd)]\n#[path="' + str(source / 'shared/genotype_table.rs') + '"] mod genotype_table;\n' + '\n'.join(modules) + main)
binary = output / 'packed_projection_probe'
subprocess.run(['rustc', '+nightly-2026-08-31', '--edition=2024', '-C', 'opt-level=3', '-C', 'target-cpu=x86-64-v3', str(probe), '-o', str(binary)], check=True, timeout=30)
subprocess.run([str(binary.resolve())], check=True, timeout=30)
