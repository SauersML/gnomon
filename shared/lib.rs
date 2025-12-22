#![feature(portable_simd)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]
// Acceptable patterns for numerical/scientific computing code
#![allow(clippy::type_complexity)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::same_item_push)]
#![allow(clippy::redundant_locals)]
// Additional acceptable patterns
#![allow(clippy::drop_non_drop)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_strip)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::useless_asref)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::iter_filter_is_ok)]
#![allow(clippy::redundant_guards)]
#![allow(clippy::if_then_some_else_none)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::iter_filter_is_some)]
#![allow(clippy::unnecessary_literal_unwrap)]
#![allow(clippy::lines_filter_map_ok)]
#![allow(clippy::unnecessary_unwrap)]

pub mod files;

pub mod shared {
    pub use super::files;
}

pub mod adapt_plink2;

#[path = "../score/mod.rs"]
pub mod score;

#[path = "../terms/mod.rs"]
pub mod terms;

pub mod batch {
    pub use crate::score::batch::*;
}

pub use score::{complex, decide, download, io, kernel, pipeline, prepare, reformat, types};

#[path = "../map/mod.rs"]
pub mod map;

#[path = "../calibrate/mod.rs"]
pub mod calibrate;
