//! MSI-only public-API checks for PCA's minimum streaming memory guard.
use gnomon::map::fit::{DenseBlockSource, HwePcaModel, VariantBlockSource};

struct ImpossibleSource {
    samples: usize,
}

impl VariantBlockSource for ImpossibleSource {
    type Error = std::convert::Infallible;
    fn n_samples(&self) -> usize {
        self.samples
    }
    fn n_variants(&self) -> usize {
        4
    }
    fn block_storage_samples(&self) -> usize {
        usize::MAX
    }
    fn reset(&mut self) -> Result<(), Self::Error> {
        panic!("infeasible source was opened")
    }
    fn next_block_into(&mut self, _: usize, _: &mut [f64]) -> Result<usize, Self::Error> {
        panic!("infeasible source was decoded")
    }
}

fn main() {
    for samples in [2, usize::MAX] {
        let error = HwePcaModel::fit_k(&mut ImpossibleSource { samples }, 1).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot hold one streamed variant")
        );
    }
    let data = [
        0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0,
    ];
    let mut source = DenseBlockSource::new(&data, 4, 4).unwrap();
    let fit = HwePcaModel::fit_k(&mut source, 2).unwrap();
    assert_eq!(fit.components(), 2);
    assert!(
        fit.explained_variance()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    );
    println!("PCA rejects infeasible logical/physical cohorts before reading; small fit succeeds");
}
