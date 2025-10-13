#![feature(portable_simd)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

pub mod files;

pub mod shared {
    pub use super::files;
}

pub mod adapt_plink2;

#[path = "../score/mod.rs"]
pub mod score;

pub mod batch {
    pub use crate::score::batch::*;

    #[cfg_attr(not(feature = "no-inline-profiling"), inline)]
    #[cfg_attr(feature = "no-inline-profiling", inline(never))]
    pub fn process_tile<'a>(
        tile: &'a [crate::score::types::EffectAlleleDosage],
        prep_result: &'a crate::score::types::PreparationResult,
        weights_for_batch: &'a [f32],
        flips_for_batch: &'a [u8],
        reconciled_variant_indices_for_batch:
            &'a [crate::score::types::ReconciledVariantIndex],
        block_scores_out: &mut [f64],
        block_missing_counts_out: &mut [u32],
    ) {
        crate::score::batch::process_tile_impl(
            tile,
            prep_result,
            weights_for_batch,
            flips_for_batch,
            reconciled_variant_indices_for_batch,
            block_scores_out,
            block_missing_counts_out,
        );
    }
}

pub use score::{complex, decide, download, io, kernel, pipeline, prepare, reformat, types};

#[path = "../map/mod.rs"]
pub mod map;

#[path = "../calibrate/mod.rs"]
pub mod calibrate;
