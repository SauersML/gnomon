use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use gnomon::calibrate::faer_ndarray::FaerArrayView;
use ndarray::Array2;
use rand::distributions::Standard;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn manual_frobenius(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn faer_frobenius(matrix: &Array2<f64>) -> f64 {
    FaerArrayView::new(matrix).as_ref().norm_l2()
}

fn random_matrix(size: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(0x5EED_F64 + size as u64);
    Array2::from_shape_fn((size, size), |_| rng.sample(Standard))
}

fn benchmark_frobenius(c: &mut Criterion) {
    let sizes = [50_usize, 100, 200];
    let matrices: Vec<_> = sizes
        .iter()
        .map(|&size| (size, random_matrix(size)))
        .collect();

    let mut group = c.benchmark_group("frobenius_norm");
    for (size, matrix) in matrices.iter() {
        let elements = (*size * *size) as u64;
        group.throughput(Throughput::Elements(elements));

        group.bench_with_input(BenchmarkId::new("manual", size), matrix, |b, input| {
            b.iter(|| {
                let norm = manual_frobenius(black_box(input));
                black_box(norm);
            });
        });

        group.bench_with_input(BenchmarkId::new("faer", size), matrix, |b, input| {
            b.iter(|| {
                let norm = faer_frobenius(black_box(input));
                black_box(norm);
            });
        });
    }
    group.finish();
}

criterion_group!(frobenius_norm, benchmark_frobenius);
criterion_main!(frobenius_norm);
