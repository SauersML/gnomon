# Score and packed projection performance

The memory-planning startup query was subsequently narrowed to the fields
actually used. On the shared MSI node, full system refresh took 267–492 ms;
memory plus process names took 32–37 ms and found the same 17 gnomon processes.
PCA's memory-only query took 0.37–0.41 ms. Memory availability and scoring's
concurrent-process fair-share checks remain active. These are query timings,
not measurements of the complete commands below.

Measured on MSI on 2026-09-13 using four pinned AMD EPYC Milan cores,
Rust nightly-2026-08-31, `target-cpu=x86-64-v3`, release optimization, and
the existing iteration cache with LTO disabled. Each number below is the
median of three runs, alternating before/after order. Inputs were local,
synthetic PLINK hard calls with warm filesystem caches.

| Workload | Before wall | After wall | Speedup | Before compute | After compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Score: 50,000 samples, 8,192 variants, 9 scores | 2.013 s | 1.422 s | 1.42× | 1.380 s | 0.806 s |
| Project: 50,000 samples, 8,192 variants, 4 PCs | 2.230 s | 0.917 s | 2.43× | 2.146 s | 0.829 s |
| Score: 16,384 samples, 2,048 variants, 9 scores | 0.709 s | 0.758 s | 0.94× | 0.123 s | 0.076 s |
| Project: 16,384 samples, 4,096 variants, 4 PCs | 0.514 s | 0.283 s | 1.82× | 0.457 s | 0.208 s |

Wall time includes process startup, preparation, computation, and output.
Compute time is the command's reported pipeline/projection time. The smaller
score workload illustrates the limit: saving 47 ms of computation did not
overcome variation in the rest of the command. These measurements do not
establish gains for PCA fitting, GPU execution, or remote genotype storage.

The scoring changes classify 32 dosages into separate SIMD masks and select
constant-width accumulator loops once per score stripe. The projection change
uses a four-bit lookup to add two samples' contribution vectors together and
walks missing-call masks separately. Cohorts smaller than 128 samples index
the model's vectors directly, avoiding the cost of constructing pair tables.
Variant accumulation order and missing-call corrections are preserved.

All before/after score TSVs and projection binaries were byte-identical,
including repeated runs. SHA-256 for the larger workload:

- Score: `46e0971ddac9e6d0cf7414421bb718949f31f8fdd9d11fdaf617435cc4db5f49`
- Projection: `409ac9f3073d7fd9085c13b80cccee97678166f545d2874ed427f91741c0a0ef`

Ten focused tests passed using the production score modules and projection
kernels. Coverage includes all 256 packed bytes, allele swaps, both missing
information representations, partial sample bytes, SIMD and mini-batch
boundaries, every accumulator width, and checked kernel bounds. The broad
Cargo test build was stopped when its development dependencies began rebuilding;
the complete test suite was not run. Production scoring and projection binaries
were built and exercised on MSI.

Benchmark scripts, logs, source snapshots, and raw JSON results are retained
on MSI at
`/projects/standard/hsiehph/sauer354/gnomon/.validation/score-map-20260913`.
`bench_cli.py` runs the smaller workload; `bench_cli.py --large` runs the larger
one. Run under `taskset -c 24-27` with a 60-second timeout. The fixture is a
fixed prefix of `gnomon-swarm/data/map/synth/c50k_20k`, with nine deterministic
score columns and a four-PC model fitted once by the baseline executable.
