use super::fit::{
    DEFAULT_BLOCK_WIDTH, DenseBlockSource, HardCallPacked, HwePcaError, HwePcaModel, HweScaler,
    VariantBlockSource,
};
use super::progress::{
    NoopProjectionProgress, ProjectionProgressObserver, ProjectionProgressStage,
};
use super::variant_filter::MatchKind;
use core::cmp::min;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, Par};
use rayon::prelude::*;
use std::error::Error;
use std::mem::size_of;
use std::path::Path;
use std::sync::OnceLock;

pub struct HwePcaProjector<'model> {
    model: &'model HwePcaModel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ZeroAlignmentAction {
    #[default]
    Zero,
    NaN,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectionOptions {
    pub missing_axis_renormalization: bool,
    pub return_alignment: bool,
    pub on_zero_alignment: ZeroAlignmentAction,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            missing_axis_renormalization: true,
            return_alignment: false,
            on_zero_alignment: ZeroAlignmentAction::NaN,
        }
    }
}

#[derive(Debug)]
pub struct ProjectionResult {
    pub scores: Mat<f64>,
    pub alignment: Option<Mat<f64>>,
}

const DEFAULT_PROJECT_CUDA_MIN_WORK: usize = 50_000_000;
const WLS_RIDGE: f64 = 1.0e-5;
const PROJECT_CUDA_UNPACK_KERNELS: &str = r#"
extern "C" __global__ void unpack_weighted_plink(
    const unsigned char* packed,
    const float* coeffs,
    int num_people,
    int batch_variants,
    int bytes_per_variant,
    float* out_matrix,
    unsigned int* missing_flags
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long total =
        (unsigned long long)num_people * (unsigned long long)batch_variants;
    if (idx >= total) return;

    unsigned long long variant = idx / (unsigned long long)num_people;
    unsigned long long person = idx % (unsigned long long)num_people;

    unsigned long long byte_idx = (unsigned long long)person >> 2;
    int bit_shift = (int)((person & 3ull) << 1);
    unsigned long long packed_offset =
        variant * (unsigned long long)bytes_per_variant + byte_idx;
    unsigned char b = packed[packed_offset];
    unsigned char gt = (b >> bit_shift) & 0x3u;

    float value = 0.0f;
    unsigned char miss = 0u;
    if (gt == 0u) {
        value = coeffs[(size_t)variant * 3u + 0u];
    } else if (gt == 2u) {
        value = coeffs[(size_t)variant * 3u + 1u];
    } else if (gt == 3u) {
        value = coeffs[(size_t)variant * 3u + 2u];
    } else if (gt == 1u) {
        miss = 1u;
    }

    out_matrix[idx] = value;
    if (miss != 0u) {
        unsigned long long word = idx >> 5;
        unsigned int bit = 1u << (idx & 31ull);
        atomicOr(&missing_flags[word], bit);
    }
}

extern "C" __global__ void clear_u32(
    unsigned int* data,
    unsigned long long n_words
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= n_words) return;
    data[idx] = 0u;
}

extern "C" __global__ void accumulate_missing_info(
    const unsigned int* missing_flags,
    const float* contrib,
    int num_people,
    int batch_variants,
    int packed_info_size,
    float* missing_info
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long total =
        (unsigned long long)num_people * (unsigned long long)batch_variants;
    if (idx >= total) return;
    unsigned long long word = idx >> 5;
    unsigned int bit = 1u << (idx & 31ull);
    if ((missing_flags[word] & bit) == 0u) return;

    unsigned long long variant = idx / (unsigned long long)num_people;
    unsigned long long person = idx % (unsigned long long)num_people;
    size_t src_off = (size_t)variant * (size_t)packed_info_size;
    size_t dst_off = (size_t)person * (size_t)packed_info_size;
    for (int t = 0; t < packed_info_size; ++t) {
        atomicAdd(&missing_info[dst_off + (size_t)t], contrib[src_off + (size_t)t]);
    }
}
"#;

struct ProjectionCudaRhs {
    _ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    blas: CudaBlas,
    d_a: Option<CudaSlice<f64>>,
    d_b: Option<CudaSlice<f64>>,
    d_c: Option<CudaSlice<f64>>,
    a_cap: usize,
    b_cap: usize,
    c_cap: usize,
    h_c: Vec<f64>,
}

struct ProjectionCudaPacked {
    _ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    blas: CudaBlas,
    unpack_kernel: CudaFunction,
    clear_u32_kernel: CudaFunction,
    missing_accum_kernel: CudaFunction,
    d_packed: Option<CudaSlice<u8>>,
    d_coeffs: Option<CudaSlice<f32>>,
    d_a: Option<CudaSlice<f32>>,
    d_b: Option<CudaSlice<f32>>,
    d_scores_accum: Option<CudaSlice<f32>>,
    d_missing_flags: Option<CudaSlice<u32>>,
    d_contrib: Option<CudaSlice<f32>>,
    d_missing_info: Option<CudaSlice<f32>>,
    packed_cap: usize,
    coeffs_cap: usize,
    a_cap: usize,
    b_cap: usize,
    scores_cap: usize,
    missing_flags_words_cap: usize,
    contrib_cap: usize,
    missing_info_cap: usize,
    h_scores: Vec<f32>,
    h_missing_info: Vec<f32>,
}

impl ProjectionCudaPacked {
    fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        ctx.bind_to_thread()
            .map_err(|e| format!("Failed to bind CUDA context: {e:?}"))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| format!("Failed to create CUDA stream: {e:?}"))?;
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("Failed to initialize cuBLAS: {e:?}"))?;
        let ptx = compile_ptx(PROJECT_CUDA_UNPACK_KERNELS)
            .map_err(|e| format!("NVRTC compile failed for projection kernel: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Failed to load projection CUDA module: {e:?}"))?;
        let unpack_kernel = module
            .load_function("unpack_weighted_plink")
            .map_err(|e| format!("Failed to load unpack_weighted_plink kernel: {e:?}"))?;
        let missing_accum_kernel = module
            .load_function("accumulate_missing_info")
            .map_err(|e| format!("Failed to load accumulate_missing_info kernel: {e:?}"))?;
        Ok(Self {
            _ctx: ctx,
            stream,
            blas,
            unpack_kernel,
            clear_u32_kernel: module
                .load_function("clear_u32")
                .map_err(|e| format!("Failed to load clear_u32 kernel: {e:?}"))?,
            missing_accum_kernel,
            d_packed: None,
            d_coeffs: None,
            d_a: None,
            d_b: None,
            d_scores_accum: None,
            d_missing_flags: None,
            d_contrib: None,
            d_missing_info: None,
            packed_cap: 0,
            coeffs_cap: 0,
            a_cap: 0,
            b_cap: 0,
            scores_cap: 0,
            missing_flags_words_cap: 0,
            contrib_cap: 0,
            missing_info_cap: 0,
            h_scores: Vec::new(),
            h_missing_info: Vec::new(),
        })
    }

    fn ensure_capacity(
        &mut self,
        packed_len: usize,
        coeffs_len: usize,
        a_len: usize,
        b_len: usize,
        scores_len: usize,
    ) -> Result<(), String> {
        if self.packed_cap < packed_len {
            self.d_packed = Some(
                self.stream
                    .alloc_zeros::<u8>(packed_len)
                    .map_err(|e| format!("Failed to allocate packed CUDA buffer: {e:?}"))?,
            );
            self.packed_cap = packed_len;
        }
        if self.coeffs_cap < coeffs_len {
            self.d_coeffs = Some(
                self.stream
                    .alloc_zeros::<f32>(coeffs_len)
                    .map_err(|e| format!("Failed to allocate coeff CUDA buffer: {e:?}"))?,
            );
            self.coeffs_cap = coeffs_len;
        }
        if self.a_cap < a_len {
            self.d_a = Some(
                self.stream
                    .alloc_zeros::<f32>(a_len)
                    .map_err(|e| format!("Failed to allocate A CUDA buffer: {e:?}"))?,
            );
            self.a_cap = a_len;
        }
        if self.b_cap < b_len {
            self.d_b = Some(
                self.stream
                    .alloc_zeros::<f32>(b_len)
                    .map_err(|e| format!("Failed to allocate B CUDA buffer: {e:?}"))?,
            );
            self.b_cap = b_len;
        }
        if self.scores_cap < scores_len {
            self.d_scores_accum = Some(
                self.stream
                    .alloc_zeros::<f32>(scores_len)
                    .map_err(|e| format!("Failed to allocate score-accum CUDA buffer: {e:?}"))?,
            );
            self.scores_cap = scores_len;
        }
        let missing_words = missing_flag_words(a_len);
        if self.missing_flags_words_cap < missing_words {
            self.d_missing_flags = Some(
                self.stream
                    .alloc_zeros::<u32>(missing_words)
                    .map_err(|e| format!("Failed to allocate missing-flag CUDA buffer: {e:?}"))?,
            );
            self.missing_flags_words_cap = missing_words;
        }
        if self.h_scores.len() < scores_len {
            self.h_scores.resize(scores_len, 0.0);
        }
        Ok(())
    }

    fn ensure_missing_capacity(
        &mut self,
        contrib_len: usize,
        missing_info_len: usize,
    ) -> Result<(), String> {
        if self.contrib_cap < contrib_len {
            self.d_contrib = Some(
                self.stream
                    .alloc_zeros::<f32>(contrib_len)
                    .map_err(|e| format!("Failed to allocate contrib CUDA buffer: {e:?}"))?,
            );
            self.contrib_cap = contrib_len;
        }
        if self.missing_info_cap < missing_info_len {
            self.d_missing_info = Some(
                self.stream
                    .alloc_zeros::<f32>(missing_info_len)
                    .map_err(|e| format!("Failed to allocate missing-info CUDA buffer: {e:?}"))?,
            );
            self.missing_info_cap = missing_info_len;
        }
        if self.h_missing_info.len() < missing_info_len {
            self.h_missing_info.resize(missing_info_len, 0.0);
        }
        Ok(())
    }

    fn init_missing_info(
        &mut self,
        n_samples: usize,
        packed_info_size: usize,
    ) -> Result<(), String> {
        let missing_info_len = n_samples
            .checked_mul(packed_info_size)
            .ok_or_else(|| "projection CUDA overflow for missing_info".to_string())?;
        self.ensure_missing_capacity(1, missing_info_len)?;
        self.d_missing_info = Some(
            self.stream
                .alloc_zeros::<f32>(missing_info_len)
                .map_err(|e| format!("Failed to zero-init missing-info CUDA buffer: {e:?}"))?,
        );
        self.missing_info_cap = missing_info_len;
        Ok(())
    }

    fn init_scores_accum(&mut self, n_samples: usize, components: usize) -> Result<(), String> {
        let score_len = n_samples
            .checked_mul(components)
            .ok_or_else(|| "projection CUDA overflow for score_accum".to_string())?;
        self.ensure_capacity(1, 1, 1, 1, score_len)?;
        self.d_scores_accum = Some(
            self.stream
                .alloc_zeros::<f32>(score_len)
                .map_err(|e| format!("Failed to zero-init score-accum CUDA buffer: {e:?}"))?,
        );
        self.scores_cap = score_len;
        Ok(())
    }

    fn compute_scores_block(
        &mut self,
        packed: &[u8],
        coeffs: &[f32],
        loadings_col_major: &[f32],
        n_samples: usize,
        filled: usize,
        components: usize,
    ) -> Result<(), String> {
        let a_len = n_samples
            .checked_mul(filled)
            .ok_or_else(|| "projection CUDA overflow for A".to_string())?;
        let b_len = filled
            .checked_mul(components)
            .ok_or_else(|| "projection CUDA overflow for B".to_string())?;
        let score_len = n_samples
            .checked_mul(components)
            .ok_or_else(|| "projection CUDA overflow for score_accum".to_string())?;
        self.ensure_capacity(packed.len(), coeffs.len(), a_len, b_len, score_len)?;

        let d_packed = self
            .d_packed
            .as_mut()
            .expect("packed CUDA buffer must be allocated");
        let d_coeffs = self
            .d_coeffs
            .as_mut()
            .expect("coeff CUDA buffer must be allocated");
        let d_a = self.d_a.as_mut().expect("A CUDA buffer must be allocated");
        let d_b = self.d_b.as_mut().expect("B CUDA buffer must be allocated");
        let d_scores_accum = self
            .d_scores_accum
            .as_mut()
            .expect("score-accum CUDA buffer must be allocated");
        let d_missing_flags = self
            .d_missing_flags
            .as_mut()
            .expect("missing-flag CUDA buffer must be allocated");

        let mut d_packed_view = d_packed.slice_mut(0..packed.len());
        let mut d_coeffs_view = d_coeffs.slice_mut(0..coeffs.len());
        let mut d_b_view = d_b.slice_mut(0..b_len);
        let mut d_a_view = d_a.slice_mut(0..a_len);
        let mut d_scores_view = d_scores_accum.slice_mut(0..score_len);
        let missing_words = missing_flag_words(a_len);
        let mut d_missing_flags_view = d_missing_flags.slice_mut(0..missing_words);

        self.stream
            .memcpy_htod(packed, &mut d_packed_view)
            .map_err(|e| format!("Failed to copy packed block to GPU: {e:?}"))?;
        self.stream
            .memcpy_htod(coeffs, &mut d_coeffs_view)
            .map_err(|e| format!("Failed to copy coeffs to GPU: {e:?}"))?;
        self.stream
            .memcpy_htod(loadings_col_major, &mut d_b_view)
            .map_err(|e| format!("Failed to copy loadings to GPU: {e:?}"))?;

        let missing_words_u32 = u32::try_from(missing_words)
            .map_err(|_| format!("missing flag words too large: {missing_words}"))?;
        let missing_words_u64 = u64::try_from(missing_words)
            .map_err(|_| format!("missing flag words exceed u64: {missing_words}"))?;
        unsafe {
            self.stream
                .launch_builder(&self.clear_u32_kernel)
                .arg(&mut d_missing_flags_view)
                .arg(&missing_words_u64)
                .launch(LaunchConfig::for_num_elems(missing_words_u32))
                .map_err(|e| format!("Failed to launch clear_u32 for missing flags: {e:?}"))?;
        }

        let n_samples_i32 =
            i32::try_from(n_samples).map_err(|_| format!("n_samples too large: {n_samples}"))?;
        let filled_i32 =
            i32::try_from(filled).map_err(|_| format!("filled too large: {filled}"))?;
        let bytes_per_variant_i32 = i32::try_from(packed_bytes_per_variant(n_samples))
            .map_err(|_| format!("bytes_per_variant too large for n_samples={n_samples}"))?;
        let unpack_elems_u32 =
            u32::try_from(a_len).map_err(|_| format!("unpack element count too large: {a_len}"))?;
        unsafe {
            self.stream
                .launch_builder(&self.unpack_kernel)
                .arg(&d_packed.slice(0..packed.len()))
                .arg(&d_coeffs.slice(0..coeffs.len()))
                .arg(&n_samples_i32)
                .arg(&filled_i32)
                .arg(&bytes_per_variant_i32)
                .arg(&mut d_a_view)
                .arg(&mut d_missing_flags_view)
                .launch(LaunchConfig::for_num_elems(unpack_elems_u32))
                .map_err(|e| format!("Failed to launch unpack_weighted_plink: {e:?}"))?;
        }

        let m = i32::try_from(n_samples).map_err(|_| format!("m too large: {n_samples}"))?;
        let n = i32::try_from(components).map_err(|_| format!("n too large: {components}"))?;
        let k = i32::try_from(filled).map_err(|_| format!("k too large: {filled}"))?;
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            alpha: 1.0f32,
            lda: m,
            ldb: k,
            beta: 1.0f32,
            ldc: m,
        };
        unsafe {
            let a_view = d_a.slice(0..a_len);
            let b_view = d_b.slice(0..b_len);
            self.blas
                .gemm(cfg, &a_view, &b_view, &mut d_scores_view)
                .map_err(|e| format!("projection cuBLAS GEMM failed: {e:?}"))?;
        }
        Ok(())
    }

    fn accumulate_missing_block(
        &mut self,
        block_contrib: &[f32],
        n_samples: usize,
        filled: usize,
        packed_info_size: usize,
    ) -> Result<(), String> {
        let contrib_len = filled
            .checked_mul(packed_info_size)
            .ok_or_else(|| "projection CUDA overflow for contrib".to_string())?;
        let missing_info_len = n_samples
            .checked_mul(packed_info_size)
            .ok_or_else(|| "projection CUDA overflow for missing_info".to_string())?;
        self.ensure_missing_capacity(contrib_len, missing_info_len)?;
        let d_contrib = self
            .d_contrib
            .as_mut()
            .expect("contrib CUDA buffer must be allocated");
        let d_missing_info = self
            .d_missing_info
            .as_mut()
            .expect("missing-info CUDA buffer must be allocated");
        let d_missing_flags = self
            .d_missing_flags
            .as_ref()
            .expect("missing-flag CUDA buffer must be allocated");

        let mut d_contrib_view = d_contrib.slice_mut(0..contrib_len);
        self.stream
            .memcpy_htod(block_contrib, &mut d_contrib_view)
            .map_err(|e| format!("Failed to copy missing contrib to GPU: {e:?}"))?;

        let n_samples_i32 =
            i32::try_from(n_samples).map_err(|_| format!("n_samples too large: {n_samples}"))?;
        let filled_i32 =
            i32::try_from(filled).map_err(|_| format!("filled too large: {filled}"))?;
        let packed_info_i32 = i32::try_from(packed_info_size)
            .map_err(|_| format!("packed_info_size too large: {packed_info_size}"))?;
        let elems = n_samples
            .checked_mul(filled)
            .ok_or_else(|| "projection CUDA overflow for missing launch".to_string())?;
        let missing_words = missing_flag_words(elems);
        let elems_u32 =
            u32::try_from(elems).map_err(|_| format!("missing launch too large: {elems}"))?;

        unsafe {
            self.stream
                .launch_builder(&self.missing_accum_kernel)
                .arg(&d_missing_flags.slice(0..missing_words))
                .arg(&d_contrib.slice(0..contrib_len))
                .arg(&n_samples_i32)
                .arg(&filled_i32)
                .arg(&packed_info_i32)
                .arg(&mut d_missing_info.slice_mut(0..missing_info_len))
                .launch(LaunchConfig::for_num_elems(elems_u32))
                .map_err(|e| format!("Failed to launch accumulate_missing_info: {e:?}"))?;
        }
        Ok(())
    }

    fn copy_missing_info_to_host(
        &mut self,
        n_samples: usize,
        packed_info_size: usize,
        out_f64: &mut [f64],
    ) -> Result<(), String> {
        let len = n_samples
            .checked_mul(packed_info_size)
            .ok_or_else(|| "projection CUDA overflow for missing-info dtoh".to_string())?;
        let d_missing_info = self
            .d_missing_info
            .as_ref()
            .ok_or_else(|| "missing-info CUDA buffer is not initialized".to_string())?;
        if out_f64.len() < len {
            return Err("missing_info_storage too small for dtoh".to_string());
        }
        if self.h_missing_info.len() < len {
            self.h_missing_info.resize(len, 0.0);
        }
        self.stream
            .memcpy_dtoh(
                &d_missing_info.slice(0..len),
                &mut self.h_missing_info[..len],
            )
            .map_err(|e| format!("Failed to copy missing-info from GPU: {e:?}"))?;
        for i in 0..len {
            out_f64[i] = self.h_missing_info[i] as f64;
        }
        Ok(())
    }

    fn copy_scores_to_host(
        &mut self,
        n_samples: usize,
        components: usize,
        out_f64: &mut [f64],
    ) -> Result<(), String> {
        let len = n_samples
            .checked_mul(components)
            .ok_or_else(|| "projection CUDA overflow for score dtoh".to_string())?;
        let d_scores = self
            .d_scores_accum
            .as_ref()
            .ok_or_else(|| "score-accum CUDA buffer is not initialized".to_string())?;
        if out_f64.len() < len {
            return Err("score output buffer too small for dtoh".to_string());
        }
        if self.h_scores.len() < len {
            self.h_scores.resize(len, 0.0);
        }
        self.stream
            .memcpy_dtoh(&d_scores.slice(0..len), &mut self.h_scores[..len])
            .map_err(|e| format!("Failed to copy projection scores from GPU: {e:?}"))?;
        for i in 0..len {
            out_f64[i] += self.h_scores[i] as f64;
        }
        Ok(())
    }
}

impl ProjectionCudaRhs {
    fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        ctx.bind_to_thread()
            .map_err(|e| format!("Failed to bind CUDA context: {e:?}"))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| format!("Failed to create CUDA stream: {e:?}"))?;
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("Failed to initialize cuBLAS: {e:?}"))?;
        Ok(Self {
            _ctx: ctx,
            stream,
            blas,
            d_a: None,
            d_b: None,
            d_c: None,
            a_cap: 0,
            b_cap: 0,
            c_cap: 0,
            h_c: Vec::new(),
        })
    }

    fn ensure_capacity(&mut self, a_len: usize, b_len: usize, c_len: usize) -> Result<(), String> {
        if self.a_cap < a_len {
            self.d_a = Some(
                self.stream
                    .alloc_zeros::<f64>(a_len)
                    .map_err(|e| format!("Failed to allocate CUDA A buffer: {e:?}"))?,
            );
            self.a_cap = a_len;
        }
        if self.b_cap < b_len {
            self.d_b = Some(
                self.stream
                    .alloc_zeros::<f64>(b_len)
                    .map_err(|e| format!("Failed to allocate CUDA B buffer: {e:?}"))?,
            );
            self.b_cap = b_len;
        }
        if self.c_cap < c_len {
            self.d_c = Some(
                self.stream
                    .alloc_zeros::<f64>(c_len)
                    .map_err(|e| format!("Failed to allocate CUDA C buffer: {e:?}"))?,
            );
            self.c_cap = c_len;
        }
        if self.h_c.len() < c_len {
            self.h_c.resize(c_len, 0.0);
        }
        Ok(())
    }

    fn compute_rhs(
        &mut self,
        a_col_major: &[f64],
        b_col_major: &[f64],
        m_rows: usize,
        k_shared: usize,
        n_cols: usize,
    ) -> Result<&[f64], String> {
        let a_len = m_rows
            .checked_mul(k_shared)
            .ok_or_else(|| "CUDA rhs overflow for A".to_string())?;
        let b_len = k_shared
            .checked_mul(n_cols)
            .ok_or_else(|| "CUDA rhs overflow for B".to_string())?;
        let c_len = m_rows
            .checked_mul(n_cols)
            .ok_or_else(|| "CUDA rhs overflow for C".to_string())?;
        self.ensure_capacity(a_len, b_len, c_len)?;

        let d_a = self
            .d_a
            .as_mut()
            .expect("CUDA A buffer must be allocated before use");
        let d_b = self
            .d_b
            .as_mut()
            .expect("CUDA B buffer must be allocated before use");
        let d_c = self
            .d_c
            .as_mut()
            .expect("CUDA C buffer must be allocated before use");
        let mut d_a_view: CudaViewMut<'_, f64> = d_a.slice_mut(0..a_len);
        let mut d_b_view: CudaViewMut<'_, f64> = d_b.slice_mut(0..b_len);
        let mut d_c_view: CudaViewMut<'_, f64> = d_c.slice_mut(0..c_len);

        self.stream
            .memcpy_htod(a_col_major, &mut d_a_view)
            .map_err(|e| format!("Failed to copy projection block to device: {e:?}"))?;
        self.stream
            .memcpy_htod(b_col_major, &mut d_b_view)
            .map_err(|e| format!("Failed to copy loading block to device: {e:?}"))?;

        let m = i32::try_from(m_rows).map_err(|_| format!("m_rows too large: {m_rows}"))?;
        let n = i32::try_from(n_cols).map_err(|_| format!("n_cols too large: {n_cols}"))?;
        let k = i32::try_from(k_shared).map_err(|_| format!("k_shared too large: {k_shared}"))?;
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            alpha: 1.0f64,
            lda: m,
            ldb: k,
            beta: 0.0f64,
            ldc: m,
        };
        unsafe {
            let a_view: CudaView<'_, f64> = d_a.slice(0..a_len);
            let b_view: CudaView<'_, f64> = d_b.slice(0..b_len);
            self.blas
                .gemm(cfg, &a_view, &b_view, &mut d_c_view)
                .map_err(|e| format!("cuBLAS GEMM failed for projection rhs: {e:?}"))?;
        }

        let d_c_ref = self
            .d_c
            .as_ref()
            .expect("CUDA C buffer must be allocated before dtoh");
        self.stream
            .memcpy_dtoh(&d_c_ref.slice(0..c_len), &mut self.h_c[..c_len])
            .map_err(|e| format!("Failed to copy projection rhs from device: {e:?}"))?;
        Ok(&self.h_c[..c_len])
    }
}

impl HwePcaModel {
    pub fn projector(&self) -> HwePcaProjector<'_> {
        HwePcaProjector { model: self }
    }

    pub fn project_dense(
        &self,
        data: &[f64],
        n_samples: usize,
        n_variants: usize,
    ) -> Result<Mat<f64>, HwePcaError> {
        let mut source = DenseBlockSource::new(data, n_samples, n_variants)?;
        self.projector().project(&mut source)
    }

    pub fn project_with_options<S>(
        &self,
        source: &mut S,
        opts: &ProjectionOptions,
    ) -> Result<ProjectionResult, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = NoopProjectionProgress;
        self.projector()
            .project_with_options_and_progress(source, opts, &progress)
    }
}
impl<'model> HwePcaProjector<'model> {
    pub fn project<S>(&self, source: &mut S) -> Result<Mat<f64>, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let options = ProjectionOptions::default();
        let result = self.project_with_options(source, &options)?;
        Ok(result.scores)
    }

    pub fn project_with_options<S>(
        &self,
        source: &mut S,
        opts: &ProjectionOptions,
    ) -> Result<ProjectionResult, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = NoopProjectionProgress;
        self.project_with_options_and_progress(source, opts, &progress)
    }

    pub fn project_with_options_and_progress<S, P>(
        &self,
        source: &mut S,
        opts: &ProjectionOptions,
        progress: &P,
    ) -> Result<ProjectionResult, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
        P: ProjectionProgressObserver,
    {
        let n_samples = source.n_samples();
        let mut scores = Mat::zeros(n_samples, self.model.components());
        let mut alignment = if opts.return_alignment {
            Some(Mat::zeros(n_samples, self.model.components()))
        } else {
            None
        };

        self.project_into_with_options_and_progress(
            source,
            scores.as_mut(),
            opts,
            alignment.as_mut().map(|mat| mat.as_mut()),
            progress,
        )?;

        Ok(ProjectionResult { scores, alignment })
    }

    pub fn project_into<S>(
        &self,
        source: &mut S,
        scores: MatMut<'_, f64>,
    ) -> Result<(), HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let options = ProjectionOptions::default();
        let progress = NoopProjectionProgress;
        self.project_into_with_options_and_progress(source, scores, &options, None, &progress)
    }

    pub(crate) fn project_into_with_options_and_progress<S, P>(
        &self,
        source: &mut S,
        mut scores: MatMut<'_, f64>,
        opts: &ProjectionOptions,
        alignment_out: Option<MatMut<'_, f64>>,
        progress: &P,
    ) -> Result<(), HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
        P: ProjectionProgressObserver,
    {
        let n_samples = source.n_samples();
        let variant_hint = source.n_variants();
        let model_variants = self.model.n_variants();
        let expected_variants = if variant_hint == 0 {
            model_variants
        } else {
            variant_hint
        };
        let components = self.model.components();

        if !opts.missing_axis_renormalization {
            return Err(HwePcaError::InvalidInput(
                "Projection requires missing axis renormalization",
            ));
        }

        if n_samples == 0 {
            return Err(HwePcaError::InvalidInput(
                "Projection requires at least one sample",
            ));
        }
        if variant_hint != 0 && variant_hint != model_variants {
            return Err(HwePcaError::InvalidInput(
                "Projection variant dimension must match fitted model",
            ));
        }
        if scores.nrows() != n_samples {
            return Err(HwePcaError::InvalidInput(
                "Projection output row count mismatch",
            ));
        }
        if scores.ncols() != components {
            return Err(HwePcaError::InvalidInput(
                "Projection output column count must equal number of components",
            ));
        }
        if let Some(ref alignment) = alignment_out
            && (alignment.nrows() != n_samples || alignment.ncols() != components)
        {
            return Err(HwePcaError::InvalidInput(
                "Alignment output shape must match projection output",
            ));
        }

        scores.fill(0.0);

        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;

        let packed_info_size = packed_tri_size(components);
        let mut global_info_packed = vec![0.0f64; packed_info_size];
        let scaler = self.model.scaler();
        let loadings = self.model.variant_loadings();
        let ld_weights = self.model.ld().map(|ld| ld.weights.as_slice());
        let normalization = self.model.component_weighted_norms_sq();
        let par = faer::get_global_parallelism();

        progress.on_stage_start(ProjectionProgressStage::Projection, variant_hint);
        if variant_hint == 0 && expected_variants > 0 {
            progress.on_stage_total(ProjectionProgressStage::Projection, expected_variants);
        }

        if let Some(packed) = source.hard_call_packed() {
            eprintln!("> Projection backend path: packed hard-call input (2-bit PLINK detected)");
            let mut missing_variants = vec![Vec::<u32>::new(); n_samples];
            let dense_missing_info = project_from_packed_hard_calls(
                packed,
                n_samples,
                expected_variants,
                components,
                scaler,
                loadings,
                ld_weights,
                packed_info_size,
                &mut global_info_packed,
                &mut scores,
                &mut missing_variants,
                progress,
            )?;
            progress.on_stage_finish(ProjectionProgressStage::Projection);
            match dense_missing_info {
                Some(missing_info_storage) => solve_projection_with_dense_missing_info(
                    scores,
                    alignment_out,
                    &missing_info_storage,
                    &global_info_packed,
                    normalization,
                    components,
                    packed_info_size,
                    opts.on_zero_alignment,
                ),
                None => solve_projection_with_sparse_missing_variants(
                    scores,
                    alignment_out,
                    &missing_variants,
                    &global_info_packed,
                    normalization,
                    loadings,
                    opts.on_zero_alignment,
                ),
            }
            return Ok(());
        } else {
            eprintln!(
                "> Projection backend path: dense/streaming input (no packed hard-call source)"
            );
            let mut missing_info_storage = vec![
                0.0f64;
                n_samples.checked_mul(packed_info_size).ok_or_else(
                    || { HwePcaError::InvalidInput("WLS info matrix storage overflow") }
                )?
            ];
            let block_capacity = projection_block_capacity(
                self.model.n_samples(),
                n_samples,
                expected_variants,
                components,
            );
            let elements = n_samples
                .checked_mul(block_capacity)
                .ok_or_else(|| HwePcaError::InvalidInput("Projection workspace size overflow"))?;
            let mut block_storage = vec![0.0f64; elements];
            // Storage for per-variant imputation quality scores (INFO/R²)
            let mut quality_storage = vec![1.0f64; block_capacity];
            let mut block_info_contrib = vec![0.0f64; block_capacity * packed_info_size];
            let mut processed = 0usize;
            let total_work = n_samples
                .saturating_mul(expected_variants)
                .saturating_mul(components);
            let mut cuda_rhs = match projection_gpu_rejection_reason(total_work) {
                Some(reason) => {
                    eprintln!("> Projection backend: CPU ({reason})");
                    None
                }
                None => match ProjectionCudaRhs::new() {
                    Ok(runtime) => {
                        eprintln!(
                            "> Projection backend: GPU (CUDA RHS acceleration enabled; CPU solves remain active)"
                        );
                        Some(runtime)
                    }
                    Err(reason) => {
                        eprintln!("> Projection backend: CPU (CUDA init failed: {reason})");
                        None
                    }
                },
            };
            let mut logged_cuda_fallback = false;
            let mut packed_loadings = Vec::<f64>::new();
            let mut packed_block = Vec::<f64>::new();

            loop {
                let filled = source
                    .next_block_into(block_capacity, &mut block_storage)
                    .map_err(|e| HwePcaError::Source(Box::new(e)))?;
                if filled == 0 {
                    break;
                }
                if processed + filled > expected_variants
                    && (variant_hint != 0 || processed + filled > model_variants)
                {
                    return Err(HwePcaError::InvalidInput(
                        "VariantBlockSource returned more variants than reported",
                    ));
                }

                // Get per-variant imputation quality for this block
                source.variant_quality(filled, &mut quality_storage[..filled]);
                let loadings_block = loadings.submatrix(processed, 0, filled, components);

                // Build per-variant packed contributions and accumulate global ideal info:
                // C_j = quality_j * vech(L_j L_j^T), A_global += C_j
                let block_contrib_len = filled * packed_info_size;
                if block_info_contrib.len() < block_contrib_len {
                    block_info_contrib.resize(block_contrib_len, 0.0);
                } else {
                    block_info_contrib[..block_contrib_len].fill(0.0);
                }
                for j_local in 0..filled {
                    let quality = quality_storage[j_local];
                    let contrib = &mut block_info_contrib
                        [j_local * packed_info_size..(j_local + 1) * packed_info_size];
                    let mut idx = 0usize;
                    for k in 0..components {
                        let l_jk = loadings_block[(j_local, k)];
                        for l in k..components {
                            let l_jl = loadings_block[(j_local, l)];
                            let value = quality * l_jk * l_jl;
                            contrib[idx] = value;
                            global_info_packed[idx] += value;
                            idx += 1;
                        }
                    }
                }

                // Sparse missingness complement:
                // A_missing_sample += C_j only when the raw value is missing.
                // Raw block is column-major with n_samples contiguous values per variant.
                let raw_block = &block_storage[..n_samples * filled];
                let scan = if let Some(packed) = source.hard_call_packed() {
                    if packed.n_variants() >= processed + filled {
                        scan_block_missingness_from_packed(packed, processed, n_samples, filled)
                    } else {
                        scan_block_missingness(raw_block, n_samples, filled)
                    }
                } else {
                    scan_block_missingness(raw_block, n_samples, filled)
                };
                let total_calls = n_samples.saturating_mul(filled);
                let present_calls = total_calls.saturating_sub(scan.missing_count);
                if scan.missing_count <= present_calls {
                    if let Some(mut missing_coords) = scan.coords {
                        apply_missing_coordinates_sparse(
                            &mut missing_coords,
                            packed_info_size,
                            &block_info_contrib[..block_contrib_len],
                            &mut missing_info_storage,
                            par,
                        );
                    } else {
                        accumulate_missing_info_from_raw(
                            raw_block,
                            n_samples,
                            filled,
                            packed_info_size,
                            &block_info_contrib[..block_contrib_len],
                            &mut missing_info_storage,
                            par,
                        );
                    }
                } else {
                    accumulate_missing_info_from_present(
                        raw_block,
                        n_samples,
                        filled,
                        packed_info_size,
                        &block_info_contrib[..block_contrib_len],
                        &mut missing_info_storage,
                        par,
                    );
                }

                // Standardize raw genotypes in-place for RHS matmul.
                let mut block = MatMut::from_column_major_slice_mut(
                    &mut block_storage[..n_samples * filled],
                    n_samples,
                    filled,
                );
                standardize_projection_block(scaler, block.as_mut(), processed, filled, par);

                // WLS: Apply omega weights to block for RHS calculation.
                // omega = quality × w (linear weight)
                // This ensures RHS matches standard projection (which uses x * w)
                let weights_slice = ld_weights.unwrap_or(&[]);
                for j in 0..filled {
                    let j_global = processed + j;
                    let quality = quality_storage[j];
                    let w = if j_global < weights_slice.len() {
                        weights_slice[j_global]
                    } else {
                        1.0
                    };
                    let omega = quality * w; // Linear weight!

                    // Apply omega to this column
                    for i in 0..n_samples {
                        block[(i, j)] *= omega;
                    }
                }

                let standardized = block.as_ref();

                // RHS: scores += (x × omega) × V = V^T × Ω × x
                let mut used_cuda = false;
                if let Some(runtime) = cuda_rhs.as_mut() {
                    let packed_len = filled.saturating_mul(components);
                    if packed_loadings.len() < packed_len {
                        packed_loadings.resize(packed_len, 0.0);
                    }
                    for col in 0..components {
                        let col_offset = col * filled;
                        for row in 0..filled {
                            packed_loadings[col_offset + row] = loadings_block[(row, col)];
                        }
                    }

                    let a_len = n_samples.saturating_mul(filled);
                    if packed_block.len() < a_len {
                        packed_block.resize(a_len, 0.0);
                    }
                    for col in 0..filled {
                        let col_offset = col * n_samples;
                        for row in 0..n_samples {
                            packed_block[col_offset + row] = block[(row, col)];
                        }
                    }
                    match runtime.compute_rhs(
                        &packed_block[..a_len],
                        &packed_loadings[..packed_len],
                        n_samples,
                        filled,
                        components,
                    ) {
                        Ok(rhs) => {
                            for col in 0..components {
                                let col_offset = col * n_samples;
                                for row in 0..n_samples {
                                    scores[(row, col)] += rhs[col_offset + row];
                                }
                            }
                            used_cuda = true;
                        }
                        Err(_) => {
                            if !logged_cuda_fallback {
                                eprintln!(
                                    "> Projection backend switch: CPU (CUDA RHS step failed during execution)"
                                );
                                logged_cuda_fallback = true;
                            }
                            cuda_rhs = None;
                        }
                    }
                }

                if !used_cuda {
                    matmul(
                        scores.as_mut(),
                        Accum::Add,
                        standardized,
                        loadings_block,
                        1.0,
                        par,
                    );
                }

                processed += filled;
                progress.on_stage_advance(ProjectionProgressStage::Projection, processed);
            }

            if processed != expected_variants {
                if variant_hint != 0 {
                    return Err(HwePcaError::InvalidInput(
                        "VariantBlockSource terminated early during projection",
                    ));
                }
                if processed != model_variants {
                    return Err(HwePcaError::InvalidInput(
                        "VariantBlockSource terminated early during projection",
                    ));
                }
            }

            progress.on_stage_finish(ProjectionProgressStage::Projection);
            solve_projection_with_dense_missing_info(
                scores,
                alignment_out,
                &missing_info_storage,
                &global_info_packed,
                normalization,
                components,
                packed_info_size,
                opts.on_zero_alignment,
            );
        }
        Ok(())
    }

    pub fn model(&self) -> &'model HwePcaModel {
        self.model
    }
}

struct ProjectionSolveBase {
    diag_indices: Vec<usize>,
    global_diag: Vec<f64>,
    base_info: Vec<f64>,
    base_factor: Vec<f64>,
    base_factor_ready: bool,
    global_trace: f64,
    base_alignment: Vec<f64>,
}

fn build_projection_solve_base(
    global_info_packed: &[f64],
    normalization: &[f64],
    components: usize,
) -> ProjectionSolveBase {
    let diag_indices: Vec<usize> = (0..components)
        .map(|k| packed_tri_index(components, k, k))
        .collect();
    let global_diag: Vec<f64> = diag_indices
        .iter()
        .map(|&idx| global_info_packed[idx])
        .collect();
    let base_info = build_dense_lower_info_matrix(global_info_packed, components, WLS_RIDGE);
    let mut base_factor = base_info.clone();
    let base_factor_ready = factorize_spd_lower_in_place(&mut base_factor, components);
    let global_trace = global_diag.iter().sum();
    let base_alignment = global_diag
        .iter()
        .enumerate()
        .map(|(k, &mass)| {
            let denom = normalization.get(k).copied().unwrap_or(0.0);
            if mass > 0.0 && denom > 0.0 {
                (mass / denom).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    ProjectionSolveBase {
        diag_indices,
        global_diag,
        base_info,
        base_factor,
        base_factor_ready,
        global_trace,
        base_alignment,
    }
}

fn solve_projection_with_dense_missing_info(
    mut scores: MatMut<'_, f64>,
    mut alignment_out: Option<MatMut<'_, f64>>,
    missing_info_storage: &[f64],
    global_info_packed: &[f64],
    normalization: &[f64],
    components: usize,
    packed_info_size: usize,
    zero_action: ZeroAlignmentAction,
) {
    let n_samples = scores.nrows();
    let solve_base = build_projection_solve_base(global_info_packed, normalization, components);
    let scores_ptr = {
        let scores_col_major = scores
            .rb_mut()
            .try_as_col_major_mut()
            .expect("projection output matrix must be contiguous column-major");
        scores_col_major.as_ptr_mut() as usize
    };
    let alignment_ptr = alignment_out.as_mut().map(|alignment| {
        let alignment_col_major = alignment
            .rb_mut()
            .try_as_col_major_mut()
            .expect("projection alignment matrix must be contiguous column-major");
        alignment_col_major.as_ptr_mut() as usize
    });
    let solve_chunk = projection_solve_sample_chunk(n_samples);
    let solve_chunks = n_samples.div_ceil(solve_chunk);

    (0..solve_chunks).into_par_iter().for_each_init(
        || {
            (
                vec![0.0f64; components * components],
                vec![0.0f64; components],
            )
        },
        |(info_matrix, rhs), chunk_idx| {
            let scores_ptr = scores_ptr as *mut f64;
            let alignment_ptr = alignment_ptr.map(|ptr| ptr as *mut f64);
            let sample_start = chunk_idx * solve_chunk;
            let sample_end = (sample_start + solve_chunk).min(n_samples);

            for sample in sample_start..sample_end {
                let info_offset = sample * packed_info_size;
                let mut missing_trace = 0.0f64;
                for &diag_idx in &solve_base.diag_indices {
                    missing_trace += missing_info_storage[info_offset + diag_idx];
                }
                let trace = solve_base.global_trace - missing_trace;

                if trace <= WLS_RIDGE * (components as f64) {
                    write_zero_projection_sample(
                        scores_ptr,
                        alignment_ptr,
                        n_samples,
                        components,
                        sample,
                        zero_action,
                    );
                    continue;
                }

                for k in 0..components {
                    rhs[k] = unsafe { *scores_ptr.add(k * n_samples + sample) };
                }

                let solved = if missing_trace == 0.0 && solve_base.base_factor_ready {
                    solve_cholesky_factor_in_place(
                        &solve_base.base_factor,
                        &mut rhs[..components],
                        components,
                    )
                } else {
                    fill_sample_info_matrix(
                        info_matrix,
                        &solve_base.base_info,
                        &missing_info_storage[info_offset..info_offset + packed_info_size],
                        components,
                    );
                    solve_spd_lower_in_place(info_matrix, &mut rhs[..components], components)
                };

                if solved {
                    for k in 0..components {
                        unsafe {
                            *scores_ptr.add(k * n_samples + sample) = rhs[k];
                        }
                    }
                    if let Some(alignment_ptr) = alignment_ptr {
                        if missing_trace == 0.0 {
                            for (k, &value) in solve_base.base_alignment.iter().enumerate() {
                                unsafe {
                                    *alignment_ptr.add(k * n_samples + sample) = value;
                                }
                            }
                        } else {
                            for (k, &diag_idx) in solve_base.diag_indices.iter().enumerate() {
                                let mass = global_info_packed[diag_idx]
                                    - missing_info_storage[info_offset + diag_idx];
                                let denom = normalization.get(k).copied().unwrap_or(0.0);
                                let value = if mass > 0.0 && denom > 0.0 {
                                    (mass / denom).sqrt()
                                } else {
                                    0.0
                                };
                                unsafe {
                                    *alignment_ptr.add(k * n_samples + sample) = value;
                                }
                            }
                        }
                    }
                } else {
                    write_zero_projection_sample(
                        scores_ptr,
                        alignment_ptr,
                        n_samples,
                        components,
                        sample,
                        zero_action,
                    );
                }
            }
        },
    );
}

fn solve_projection_with_sparse_missing_variants(
    mut scores: MatMut<'_, f64>,
    mut alignment_out: Option<MatMut<'_, f64>>,
    missing_variants: &[Vec<u32>],
    global_info_packed: &[f64],
    normalization: &[f64],
    loadings: faer::MatRef<'_, f64>,
    zero_action: ZeroAlignmentAction,
) {
    let n_samples = scores.nrows();
    let components = scores.ncols();
    let solve_base = build_projection_solve_base(global_info_packed, normalization, components);
    let scores_ptr = {
        let scores_col_major = scores
            .rb_mut()
            .try_as_col_major_mut()
            .expect("projection output matrix must be contiguous column-major");
        scores_col_major.as_ptr_mut() as usize
    };
    let alignment_ptr = alignment_out.as_mut().map(|alignment| {
        let alignment_col_major = alignment
            .rb_mut()
            .try_as_col_major_mut()
            .expect("projection alignment matrix must be contiguous column-major");
        alignment_col_major.as_ptr_mut() as usize
    });
    let solve_chunk = projection_solve_sample_chunk(n_samples);
    let solve_chunks = n_samples.div_ceil(solve_chunk);

    (0..solve_chunks).into_par_iter().for_each_init(
        || {
            (
                vec![0.0f64; components * components],
                vec![0.0f64; components],
                vec![0.0f64; components],
            )
        },
        |(info_matrix, rhs, diag_mass), chunk_idx| {
            let scores_ptr = scores_ptr as *mut f64;
            let alignment_ptr = alignment_ptr.map(|ptr| ptr as *mut f64);
            let sample_start = chunk_idx * solve_chunk;
            let sample_end = (sample_start + solve_chunk).min(n_samples);

            for sample in sample_start..sample_end {
                let missing = &missing_variants[sample];
                for k in 0..components {
                    rhs[k] = unsafe { *scores_ptr.add(k * n_samples + sample) };
                }

                let solved = if missing.is_empty() && solve_base.base_factor_ready {
                    solve_cholesky_factor_in_place(
                        &solve_base.base_factor,
                        &mut rhs[..components],
                        components,
                    )
                } else {
                    info_matrix[..components * components]
                        .copy_from_slice(&solve_base.base_info[..components * components]);
                    diag_mass[..components].copy_from_slice(&solve_base.global_diag[..components]);
                    for &variant in missing {
                        let variant = variant as usize;
                        for row in 0..components {
                            let row_loading = loadings[(variant, row)];
                            diag_mass[row] -= row_loading * row_loading;
                            for col in row..components {
                                info_matrix[col * components + row] -=
                                    row_loading * loadings[(variant, col)];
                            }
                        }
                    }
                    let trace: f64 = diag_mass.iter().sum();
                    if trace <= WLS_RIDGE * (components as f64) {
                        write_zero_projection_sample(
                            scores_ptr,
                            alignment_ptr,
                            n_samples,
                            components,
                            sample,
                            zero_action,
                        );
                        continue;
                    }
                    solve_spd_lower_in_place(info_matrix, &mut rhs[..components], components)
                };

                if solved {
                    for k in 0..components {
                        unsafe {
                            *scores_ptr.add(k * n_samples + sample) = rhs[k];
                        }
                    }
                    if let Some(alignment_ptr) = alignment_ptr {
                        if missing.is_empty() {
                            for (k, &value) in solve_base.base_alignment.iter().enumerate() {
                                unsafe {
                                    *alignment_ptr.add(k * n_samples + sample) = value;
                                }
                            }
                        } else {
                            for (k, &mass) in diag_mass.iter().enumerate() {
                                let denom = normalization.get(k).copied().unwrap_or(0.0);
                                let value = if mass > 0.0 && denom > 0.0 {
                                    (mass / denom).sqrt()
                                } else {
                                    0.0
                                };
                                unsafe {
                                    *alignment_ptr.add(k * n_samples + sample) = value;
                                }
                            }
                        }
                    }
                } else {
                    write_zero_projection_sample(
                        scores_ptr,
                        alignment_ptr,
                        n_samples,
                        components,
                        sample,
                        zero_action,
                    );
                }
            }
        },
    );
}

fn standardize_projection_block(
    scaler: &HweScaler,
    block: MatMut<'_, f64>,
    offset: usize,
    filled: usize,
    par: Par,
) {
    scaler.standardize_block(block, offset..offset + filled, par);
}

fn projection_block_capacity(
    fitted_samples: usize,
    projected_samples: usize,
    n_variants: usize,
    components: usize,
) -> usize {
    if n_variants == 0 {
        return 1;
    }
    let default = DEFAULT_BLOCK_WIDTH.max(1);
    let safe_reference = fitted_samples.saturating_mul(default);
    let mut capacity = if projected_samples == 0 {
        default
    } else {
        safe_reference / projected_samples
    };
    if capacity == 0 {
        capacity = 1;
    }
    capacity = min(capacity, default);
    capacity = min(capacity, n_variants);
    let budget = projection_block_budget_bytes(projected_samples, components);
    let bytes_per_column = projected_samples.saturating_mul(size_of::<f64>());
    if bytes_per_column > 0 {
        let budget_limited = (budget / bytes_per_column).max(1);
        capacity = min(capacity, budget_limited);
    }
    capacity
}

fn projection_block_budget_bytes(projected_samples: usize, components: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let lower = 2usize * 1024 * 1024;
    let upper = 64usize * 1024 * 1024;

    // Target a cache-resident stream budget: scale with active threads,
    // then bias upward for small N or larger K to keep GEMM efficiency.
    let mut budget = (threads * 4 * 1024 * 1024).clamp(lower, upper);
    if projected_samples <= 8_192 {
        budget = budget.max(32 * 1024 * 1024);
    }
    if projected_samples <= 1_024 {
        budget = upper;
    }
    if components >= 64 {
        budget = budget.saturating_mul(3) / 2;
    } else if components >= 32 {
        budget = budget.saturating_mul(5) / 4;
    }
    budget.clamp(lower, upper)
}

fn project_cuda_disabled() -> bool {
    std::env::var("GNOMON_PROJECT_DISABLE_CUDA")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn projection_gpu_rejection_reason(total_work: usize) -> Option<String> {
    let min_work = project_cuda_min_work();
    if total_work < min_work {
        return Some(format!(
            "workload below CUDA threshold ({total_work} < {min_work})"
        ));
    }
    if project_cuda_disabled() {
        return Some("GNOMON_PROJECT_DISABLE_CUDA is set".to_string());
    }
    if !cuda_driver_likely_available() {
        return Some("no CUDA driver/device detected".to_string());
    }
    None
}

fn project_cuda_min_work() -> usize {
    std::env::var("GNOMON_PROJECT_CUDA_MIN_WORK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_PROJECT_CUDA_MIN_WORK)
}

fn cuda_driver_likely_available() -> bool {
    if let Ok(devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
        let v = devices.trim();
        if v.is_empty() || v == "-1" || v.eq_ignore_ascii_case("none") {
            return false;
        }
    }

    if cfg!(target_os = "linux") && Path::new("/dev/nvidiactl").exists() {
        return true;
    }

    std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[inline]
fn packed_tri_size(components: usize) -> usize {
    components.saturating_mul(components.saturating_add(1)) / 2
}

#[inline]
fn packed_tri_index(components: usize, row: usize, col: usize) -> usize {
    debug_assert!(row <= col);
    debug_assert!(col < components);
    row * components - (row * row.saturating_sub(1)) / 2 + (col - row)
}

fn project_from_packed_hard_calls<P>(
    packed: HardCallPacked<'_>,
    n_samples: usize,
    expected_variants: usize,
    components: usize,
    scaler: &HweScaler,
    loadings: faer::MatRef<'_, f64>,
    ld_weights: Option<&[f64]>,
    packed_info_size: usize,
    global_info_packed: &mut [f64],
    scores: &mut MatMut<'_, f64>,
    missing_variants: &mut [Vec<u32>],
    progress: &P,
) -> Result<Option<Vec<f64>>, HwePcaError>
where
    P: ProjectionProgressObserver,
{
    if packed.n_variants() < expected_variants {
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource terminated early during projection",
        ));
    }
    if expected_variants > u32::MAX as usize {
        return Err(HwePcaError::InvalidInput(
            "Projection variant dimension exceeds packed missingness index capacity",
        ));
    }

    let freqs = scaler.allele_frequencies();
    let scales = scaler.variant_scales();
    if freqs.len() < expected_variants || scales.len() < expected_variants {
        return Err(HwePcaError::InvalidInput(
            "Projection variant dimension must match fitted model",
        ));
    }

    let weights_slice = ld_weights.unwrap_or(&[]);
    let block_variants = packed_projection_variant_block(components, n_samples, expected_variants);
    let sample_chunk = packed_projection_sample_chunk(n_samples);
    let total_work = n_samples
        .saturating_mul(expected_variants)
        .saturating_mul(components);
    let mut packed_cuda = match projection_gpu_rejection_reason(total_work) {
        Some(reason) => {
            eprintln!("> Projection backend: CPU ({reason})");
            None
        }
        None => match ProjectionCudaPacked::new() {
            Ok(runtime) => {
                eprintln!(
                    "> Projection backend: GPU (CUDA fused decode + SGEMM + GPU missing-info accumulation; CPU solves remain active)"
                );
                Some(runtime)
            }
            Err(reason) => {
                eprintln!("> Projection backend: CPU (CUDA init failed: {reason})");
                None
            }
        },
    };
    let mut scores_row_major = vec![0.0f64; n_samples * components];
    let mut dense_missing_info_storage = packed_cuda
        .as_ref()
        .map(|_| vec![0.0f64; n_samples * packed_info_size]);
    let mut gpu_loadings_col_major = vec![0.0f32; block_variants * components];
    let mut gpu_coeffs = vec![0.0f32; block_variants * 3];
    let mut gpu_block_contrib = vec![0.0f32; block_variants * packed_info_size];
    let mut gpu_missing_active = packed_cuda.is_some();
    let mut gpu_scores_active = packed_cuda.is_some();
    let mut gpu_scores_flushed = false;
    let mut logged_cuda_fallback = false;
    if let Some(runtime) = packed_cuda.as_mut()
        && runtime
            .init_missing_info(n_samples, packed_info_size)
            .is_err()
    {
        eprintln!(
            "> Projection backend switch: CPU (failed to initialize GPU missing-info buffer)"
        );
        packed_cuda = None;
        gpu_missing_active = false;
        gpu_scores_active = false;
        logged_cuda_fallback = true;
        dense_missing_info_storage = None;
    }
    if let Some(runtime) = packed_cuda.as_mut()
        && runtime.init_scores_accum(n_samples, components).is_err()
    {
        eprintln!("> Projection backend switch: CPU (failed to initialize GPU score accumulator)");
        packed_cuda = None;
        gpu_missing_active = false;
        gpu_scores_active = false;
        logged_cuda_fallback = true;
        dense_missing_info_storage = None;
    }

    let mut processed = 0usize;
    let mut block_loading = vec![0.0f64; block_variants * components];
    let mut block_coeffs = vec![0.0f64; block_variants * 3];
    let mut block_info_contrib = vec![0.0f64; block_variants * packed_info_size];
    let mut block_variant_bytes: Vec<&[u8]> = Vec::with_capacity(block_variants);

    while processed < expected_variants {
        let filled = (expected_variants - processed).min(block_variants);
        block_variant_bytes.clear();
        let load_len = filled * components;
        let coeff_len = filled * 3;
        let contrib_len = filled * packed_info_size;
        let packed_len = filled * packed_bytes_per_variant(n_samples);
        block_loading[..load_len].fill(0.0);
        block_coeffs[..coeff_len].fill(0.0);
        block_info_contrib[..contrib_len].fill(0.0);

        for j_local in 0..filled {
            let j_global = processed + j_local;
            let bytes = packed.slice(j_global, 1).ok_or(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early during projection",
            ))?;
            block_variant_bytes.push(bytes);

            let mean = 2.0 * freqs[j_global];
            let denom = scales[j_global];
            let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
            let w = if j_global < weights_slice.len() {
                weights_slice[j_global]
            } else {
                1.0
            };
            let coeff0 = (0.0 - mean) * inv * w;
            let coeff1 = (1.0 - mean) * inv * w;
            let coeff2 = (2.0 - mean) * inv * w;
            if packed.match_kind(j_global) == MatchKind::Swap {
                block_coeffs[j_local * 3] = coeff2;
                block_coeffs[j_local * 3 + 1] = coeff1;
                block_coeffs[j_local * 3 + 2] = coeff0;
            } else {
                block_coeffs[j_local * 3] = coeff0;
                block_coeffs[j_local * 3 + 1] = coeff1;
                block_coeffs[j_local * 3 + 2] = coeff2;
            }

            let loading_row = &mut block_loading[j_local * components..(j_local + 1) * components];
            for k in 0..components {
                loading_row[k] = loadings[(j_global, k)];
            }
        }

        for j_local in 0..filled {
            populate_packed_info_contrib(
                &block_loading,
                components,
                packed_info_size,
                j_local,
                &mut block_info_contrib,
            );
            let contrib =
                &block_info_contrib[j_local * packed_info_size..(j_local + 1) * packed_info_size];
            for idx in 0..packed_info_size {
                global_info_packed[idx] += contrib[idx];
            }
        }

        let mut used_gpu = false;
        let block_bytes_for_gpu = if packed_cuda.is_some() {
            packed.slice(processed, filled)
        } else {
            None
        };
        if packed_cuda.is_some() && block_bytes_for_gpu.is_none() {
            if let Some(runtime) = packed_cuda.as_mut() {
                if gpu_scores_active && !gpu_scores_flushed {
                    let _ =
                        runtime.copy_scores_to_host(n_samples, components, &mut scores_row_major);
                    gpu_scores_flushed = true;
                }
                if gpu_missing_active {
                    let _ = runtime.copy_missing_info_to_host(
                        n_samples,
                        packed_info_size,
                        dense_missing_info_storage
                            .as_mut()
                            .expect("dense missing-info storage must exist when GPU is active"),
                    );
                    gpu_missing_active = false;
                }
            }
            if !logged_cuda_fallback {
                eprintln!(
                    "> Projection backend switch: CPU (packed variant selection is not contiguous)"
                );
                logged_cuda_fallback = true;
            }
            packed_cuda = None;
            gpu_scores_active = false;
        }
        if let (Some(runtime), Some(block_bytes)) = (packed_cuda.as_mut(), block_bytes_for_gpu) {
            debug_assert_eq!(block_bytes.len(), packed_len);

            if gpu_loadings_col_major.len() < load_len {
                gpu_loadings_col_major.resize(load_len, 0.0);
            }
            if gpu_coeffs.len() < coeff_len {
                gpu_coeffs.resize(coeff_len, 0.0);
            }
            for col in 0..components {
                let col_offset = col * filled;
                for row in 0..filled {
                    gpu_loadings_col_major[col_offset + row] =
                        block_loading[row * components + col] as f32;
                }
            }
            for idx in 0..coeff_len {
                gpu_coeffs[idx] = block_coeffs[idx] as f32;
            }
            if gpu_block_contrib.len() < contrib_len {
                gpu_block_contrib.resize(contrib_len, 0.0);
            }
            for idx in 0..contrib_len {
                gpu_block_contrib[idx] = block_info_contrib[idx] as f32;
            }

            match runtime.compute_scores_block(
                block_bytes,
                &gpu_coeffs[..coeff_len],
                &gpu_loadings_col_major[..load_len],
                n_samples,
                filled,
                components,
            ) {
                Ok(()) => {
                    if gpu_missing_active {
                        match runtime.accumulate_missing_block(
                            &gpu_block_contrib[..contrib_len],
                            n_samples,
                            filled,
                            packed_info_size,
                        ) {
                            Ok(()) => {
                                used_gpu = true;
                            }
                            Err(_) => {
                                if gpu_scores_active && !gpu_scores_flushed {
                                    let _ = runtime.copy_scores_to_host(
                                        n_samples,
                                        components,
                                        &mut scores_row_major,
                                    );
                                    gpu_scores_flushed = true;
                                }
                                let _ = runtime.copy_missing_info_to_host(
                                    n_samples,
                                    packed_info_size,
                                    dense_missing_info_storage.as_mut().expect(
                                        "dense missing-info storage must exist when GPU is active",
                                    ),
                                );
                                if !logged_cuda_fallback {
                                    eprintln!(
                                        "> Projection backend switch: CPU (GPU missing-info accumulation failed during execution)"
                                    );
                                    logged_cuda_fallback = true;
                                }
                                gpu_missing_active = false;
                                packed_cuda = None;
                            }
                        }
                    } else {
                        used_gpu = true;
                    }
                }
                Err(_) => {
                    if gpu_scores_active && !gpu_scores_flushed {
                        let _ = runtime.copy_scores_to_host(
                            n_samples,
                            components,
                            &mut scores_row_major,
                        );
                        gpu_scores_flushed = true;
                    }
                    if gpu_missing_active {
                        let _ = runtime.copy_missing_info_to_host(
                            n_samples,
                            packed_info_size,
                            dense_missing_info_storage
                                .as_mut()
                                .expect("dense missing-info storage must exist when GPU is active"),
                        );
                        gpu_missing_active = false;
                    }
                    if !logged_cuda_fallback {
                        eprintln!(
                            "> Projection backend switch: CPU (GPU score block compute failed during execution)"
                        );
                        logged_cuda_fallback = true;
                    }
                    packed_cuda = None;
                    gpu_scores_active = false;
                }
            }
        }

        if !used_gpu {
            if let Some(missing_info_storage) = dense_missing_info_storage.as_mut() {
                accumulate_packed_cpu_block_row_major(
                    &block_variant_bytes,
                    &block_loading[..load_len],
                    &block_coeffs[..coeff_len],
                    &block_info_contrib[..contrib_len],
                    sample_chunk,
                    components,
                    packed_info_size,
                    &mut scores_row_major,
                    missing_info_storage,
                );
            } else {
                accumulate_packed_cpu_block_row_major_sparse_missing(
                    &block_variant_bytes,
                    &block_loading[..load_len],
                    &block_coeffs[..coeff_len],
                    processed,
                    sample_chunk,
                    components,
                    &mut scores_row_major,
                    missing_variants,
                );
            }
        }

        processed += filled;
        progress.on_stage_advance(ProjectionProgressStage::Projection, processed);
    }

    if gpu_missing_active
        && let Some(runtime) = packed_cuda.as_mut()
        && runtime
            .copy_missing_info_to_host(
                n_samples,
                packed_info_size,
                dense_missing_info_storage
                    .as_mut()
                    .expect("dense missing-info storage must exist when GPU is active"),
            )
            .is_err()
    {
        // Fall back to whatever CPU accumulation already has (currently none for
        // successful GPU runs). In this failure case we leave the existing values.
    }
    if gpu_scores_active
        && !gpu_scores_flushed
        && let Some(runtime) = packed_cuda.as_mut()
        && runtime
            .copy_scores_to_host(n_samples, components, &mut scores_row_major)
            .is_ok()
    {}

    for sample in 0..n_samples {
        let score_offset = sample * components;
        for k in 0..components {
            scores[(sample, k)] = scores_row_major[score_offset + k];
        }
    }

    Ok(dense_missing_info_storage)
}

fn accumulate_packed_cpu_block_row_major(
    block_variant_bytes: &[&[u8]],
    block_loading: &[f64],
    block_coeffs: &[f64],
    block_info_contrib: &[f64],
    sample_chunk: usize,
    components: usize,
    packed_info_size: usize,
    scores_row_major: &mut [f64],
    missing_info_storage: &mut [f64],
) {
    let samples = scores_row_major.len() / components;
    let chunk_scores = sample_chunk * components;
    let chunk_missing = sample_chunk * packed_info_size;
    scores_row_major
        .par_chunks_mut(chunk_scores)
        .zip(missing_info_storage.par_chunks_mut(chunk_missing))
        .enumerate()
        .for_each(|(chunk_idx, (score_chunk, missing_chunk))| {
            let sample_start = chunk_idx * sample_chunk;
            let chunk_samples = score_chunk.len() / components;
            let byte_start = sample_start >> 2;
            let byte_len = packed_bytes_per_variant(chunk_samples);

            for (variant, variant_bytes) in block_variant_bytes.iter().enumerate() {
                debug_assert_eq!(variant_bytes.len(), packed_bytes_per_variant(samples));
                let bytes = &variant_bytes[byte_start..byte_start + byte_len];
                let coeffs = &block_coeffs[variant * 3..variant * 3 + 3];
                let loading = &block_loading[variant * components..(variant + 1) * components];
                let contrib = &block_info_contrib
                    [variant * packed_info_size..(variant + 1) * packed_info_size];

                for (byte_idx, &byte) in bytes.iter().enumerate() {
                    let sample_base = byte_idx << 2;
                    let lanes = (chunk_samples - sample_base).min(4);
                    for lane in 0..lanes {
                        let sample = sample_base + lane;
                        let code = (byte >> (lane << 1)) & 0b11;
                        let score_offset = sample * components;
                        if code == 1 {
                            let missing_offset = sample * packed_info_size;
                            let dst = &mut missing_chunk
                                [missing_offset..missing_offset + packed_info_size];
                            for idx in 0..packed_info_size {
                                dst[idx] += contrib[idx];
                            }
                            continue;
                        }

                        let coeff = match code {
                            0 => coeffs[0],
                            2 => coeffs[1],
                            3 => coeffs[2],
                            _ => continue,
                        };
                        let dst = &mut score_chunk[score_offset..score_offset + components];
                        for k in 0..components {
                            dst[k] += coeff * loading[k];
                        }
                    }
                }
            }
        });
}

fn accumulate_packed_cpu_block_row_major_sparse_missing(
    block_variant_bytes: &[&[u8]],
    block_loading: &[f64],
    block_coeffs: &[f64],
    variant_offset: usize,
    sample_chunk: usize,
    components: usize,
    scores_row_major: &mut [f64],
    missing_variants: &mut [Vec<u32>],
) {
    let samples = scores_row_major.len() / components;
    let chunk_scores = sample_chunk * components;
    scores_row_major
        .par_chunks_mut(chunk_scores)
        .zip(missing_variants.par_chunks_mut(sample_chunk))
        .enumerate()
        .for_each(|(chunk_idx, (score_chunk, missing_chunk))| {
            let sample_start = chunk_idx * sample_chunk;
            let chunk_samples = score_chunk.len() / components;
            let byte_start = sample_start >> 2;
            let byte_len = packed_bytes_per_variant(chunk_samples);

            for (variant, variant_bytes) in block_variant_bytes.iter().enumerate() {
                debug_assert_eq!(variant_bytes.len(), packed_bytes_per_variant(samples));
                let bytes = &variant_bytes[byte_start..byte_start + byte_len];
                let coeffs = &block_coeffs[variant * 3..variant * 3 + 3];
                let loading = &block_loading[variant * components..(variant + 1) * components];
                let global_variant = (variant_offset + variant) as u32;

                for (byte_idx, &byte) in bytes.iter().enumerate() {
                    let sample_base = byte_idx << 2;
                    let lanes = (chunk_samples - sample_base).min(4);
                    for lane in 0..lanes {
                        let sample = sample_base + lane;
                        let code = (byte >> (lane << 1)) & 0b11;
                        let score_offset = sample * components;
                        if code == 1 {
                            missing_chunk[sample].push(global_variant);
                            continue;
                        }

                        let coeff = match code {
                            0 => coeffs[0],
                            2 => coeffs[1],
                            3 => coeffs[2],
                            _ => continue,
                        };
                        let dst = &mut score_chunk[score_offset..score_offset + components];
                        for k in 0..components {
                            dst[k] += coeff * loading[k];
                        }
                    }
                }
            }
        });
}

fn build_dense_lower_info_matrix(
    global_info_packed: &[f64],
    components: usize,
    ridge: f64,
) -> Vec<f64> {
    let mut dense = vec![0.0f64; components * components];
    let mut idx = 0usize;
    for row in 0..components {
        for col in row..components {
            let value = global_info_packed[idx];
            dense[col * components + row] = value;
            idx += 1;
        }
        dense[row * components + row] += ridge;
    }
    dense
}

fn fill_sample_info_matrix(
    dst: &mut [f64],
    base_info: &[f64],
    missing_info_packed: &[f64],
    components: usize,
) {
    dst[..components * components].copy_from_slice(&base_info[..components * components]);
    let mut idx = 0usize;
    for row in 0..components {
        for col in row..components {
            dst[col * components + row] -= missing_info_packed[idx];
            idx += 1;
        }
    }
}

fn factorize_spd_lower_in_place(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let diag_offset = j * n + j;
        let mut diag = a[diag_offset];
        for k in 0..j {
            let value = a[j * n + k];
            diag -= value * value;
        }
        if !diag.is_finite() || diag <= 0.0 {
            return false;
        }
        let diag = diag.sqrt();
        a[diag_offset] = diag;
        let inv_diag = diag.recip();

        for i in (j + 1)..n {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = value * inv_diag;
        }
    }

    true
}

fn solve_cholesky_factor_in_place(factor: &[f64], b: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        let mut value = b[i];
        for k in 0..i {
            value -= factor[i * n + k] * b[k];
        }
        let diag = factor[i * n + i];
        if !diag.is_finite() || diag == 0.0 {
            return false;
        }
        b[i] = value / diag;
    }

    for i in (0..n).rev() {
        let mut value = b[i];
        for k in (i + 1)..n {
            value -= factor[k * n + i] * b[k];
        }
        let diag = factor[i * n + i];
        if !diag.is_finite() || diag == 0.0 {
            return false;
        }
        b[i] = value / diag;
    }

    true
}

fn solve_spd_lower_in_place(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    factorize_spd_lower_in_place(a, n) && solve_cholesky_factor_in_place(a, b, n)
}

fn write_zero_projection_sample(
    scores_ptr: *mut f64,
    alignment_ptr: Option<*mut f64>,
    n_samples: usize,
    components: usize,
    sample: usize,
    zero_action: ZeroAlignmentAction,
) {
    let score_value = match zero_action {
        ZeroAlignmentAction::Zero => 0.0,
        ZeroAlignmentAction::NaN => f64::NAN,
    };
    for component in 0..components {
        unsafe {
            *scores_ptr.add(component * n_samples + sample) = score_value;
        }
    }
    if let Some(alignment_ptr) = alignment_ptr {
        for component in 0..components {
            unsafe {
                *alignment_ptr.add(component * n_samples + sample) = 0.0;
            }
        }
    }
}

fn projection_solve_sample_chunk(n_samples: usize) -> usize {
    if n_samples >= 250_000 {
        8192
    } else if n_samples >= 50_000 {
        4096
    } else if n_samples >= 10_000 {
        2048
    } else {
        512
    }
}

fn populate_packed_info_contrib(
    block_loading: &[f64],
    components: usize,
    packed_info_size: usize,
    variant: usize,
    block_info_contrib: &mut [f64],
) {
    let loading_row = &block_loading[variant * components..(variant + 1) * components];
    let contrib =
        &mut block_info_contrib[variant * packed_info_size..(variant + 1) * packed_info_size];
    let mut idx = 0usize;
    for k in 0..components {
        let l_jk = loading_row[k];
        for l in k..components {
            contrib[idx] = l_jk * loading_row[l];
            idx += 1;
        }
    }
}

#[inline]
fn packed_bytes_per_variant(n_samples: usize) -> usize {
    (n_samples + 3) >> 2
}

#[inline]
fn missing_flag_words(total_entries: usize) -> usize {
    (total_entries + 31) >> 5
}

fn packed_projection_variant_block(
    components: usize,
    n_samples: usize,
    expected_variants: usize,
) -> usize {
    if expected_variants == 0 {
        return 1;
    }
    let packed_info_size = packed_tri_size(components);
    let bytes_per_variant_meta = (components + packed_info_size + 3) * size_of::<f64>();
    let target = if n_samples >= 200_000 {
        256 * 1024
    } else if n_samples >= 50_000 {
        384 * 1024
    } else {
        512 * 1024
    };
    let mut block = (target / bytes_per_variant_meta.max(1)).max(16);
    block = block.min(DEFAULT_BLOCK_WIDTH);
    block.min(expected_variants)
}

fn packed_projection_sample_chunk(n_samples: usize) -> usize {
    if n_samples >= 250_000 {
        4096
    } else if n_samples >= 50_000 {
        2048
    } else if n_samples >= 10_000 {
        1024
    } else {
        256
    }
}

#[derive(Clone, Copy, Debug)]
struct MissingCoord {
    sample: u32,
    variant: u32,
}

struct BlockMissingnessScan {
    missing_count: usize,
    coords: Option<Vec<MissingCoord>>,
}

fn scan_block_missingness(
    raw_block: &[f64],
    n_samples: usize,
    filled: usize,
) -> BlockMissingnessScan {
    if n_samples == 0 || filled == 0 {
        return BlockMissingnessScan {
            missing_count: 0,
            coords: Some(Vec::new()),
        };
    }
    if n_samples > (u32::MAX as usize) || filled > (u32::MAX as usize) {
        return BlockMissingnessScan {
            missing_count: raw_block.iter().filter(|v| !v.is_finite()).count(),
            coords: None,
        };
    }

    // If missingness grows too large, building/sorting sparse coordinates
    // becomes slower and more memory-hungry than direct scan accumulation.
    let max_sparse = raw_block.len() / 5;
    let mut coords = Vec::with_capacity(raw_block.len() / 100 + 1);
    let mut missing_count = 0usize;
    let mut sparse_enabled = true;

    for j_local in 0..filled {
        let column = &raw_block[j_local * n_samples..(j_local + 1) * n_samples];
        for (sample, &raw) in column.iter().enumerate() {
            if raw.is_finite() {
                continue;
            }
            missing_count += 1;
            if sparse_enabled {
                coords.push(MissingCoord {
                    sample: sample as u32,
                    variant: j_local as u32,
                });
                if coords.len() > max_sparse {
                    sparse_enabled = false;
                    coords.clear();
                }
            }
        }
    }

    BlockMissingnessScan {
        missing_count,
        coords: if sparse_enabled { Some(coords) } else { None },
    }
}

fn scan_block_missingness_from_packed(
    packed: HardCallPacked<'_>,
    start_variant: usize,
    n_samples: usize,
    filled: usize,
) -> BlockMissingnessScan {
    if n_samples == 0 || filled == 0 {
        return BlockMissingnessScan {
            missing_count: 0,
            coords: Some(Vec::new()),
        };
    }
    if n_samples > (u32::MAX as usize) || filled > (u32::MAX as usize) {
        return BlockMissingnessScan {
            missing_count: 0,
            coords: None,
        };
    }

    let total = n_samples.saturating_mul(filled);
    let max_sparse = total / 5;
    let mut coords = Vec::with_capacity(total / 100 + 1);
    let mut missing_count = 0usize;
    let mut sparse_enabled = true;

    for j_local in 0..filled {
        let bytes = match packed.slice(start_variant + j_local, 1) {
            Some(bytes) => bytes,
            None => {
                return BlockMissingnessScan {
                    missing_count,
                    coords: None,
                };
            }
        };

        let mut sample = 0usize;
        for &byte in bytes {
            if sample >= n_samples {
                break;
            }
            for offset in 0..4 {
                if sample >= n_samples {
                    break;
                }
                let code = (byte >> (offset * 2)) & 0b11;
                if code == 1 {
                    missing_count += 1;
                    if sparse_enabled {
                        coords.push(MissingCoord {
                            sample: sample as u32,
                            variant: j_local as u32,
                        });
                        if coords.len() > max_sparse {
                            sparse_enabled = false;
                            coords.clear();
                        }
                    }
                }
                sample += 1;
            }
        }
    }

    BlockMissingnessScan {
        missing_count,
        coords: if sparse_enabled { Some(coords) } else { None },
    }
}

fn apply_missing_coordinates_sparse(
    missing_coords: &mut [MissingCoord],
    packed_info_size: usize,
    block_info_contrib: &[f64],
    missing_info_storage: &mut [f64],
    par: Par,
) {
    if missing_coords.is_empty() {
        return;
    }
    if par.degree() <= 1 || missing_coords.len() < 4_096 {
        apply_missing_coordinates_sparse_serial(
            missing_coords,
            packed_info_size,
            block_info_contrib,
            missing_info_storage,
        );
        return;
    }

    missing_coords.sort_unstable_by_key(|coord| coord.sample);
    let sample_chunk = sparse_sample_range_chunk(missing_info_storage.len() / packed_info_size);
    let chunk_width = sample_chunk * packed_info_size;
    missing_info_storage
        .par_chunks_mut(chunk_width)
        .enumerate()
        .for_each(|(chunk_idx, missing_chunk)| {
            let sample_start = chunk_idx * sample_chunk;
            let chunk_samples = missing_chunk.len() / packed_info_size;
            let sample_end = sample_start + chunk_samples;

            let coord_start = lower_bound_by_sample(missing_coords, sample_start as u32);
            let coord_end = lower_bound_by_sample(missing_coords, sample_end as u32);

            for coord in &missing_coords[coord_start..coord_end] {
                let sample = coord.sample as usize - sample_start;
                let variant = coord.variant as usize;
                let contrib = &block_info_contrib
                    [variant * packed_info_size..(variant + 1) * packed_info_size];
                let dst_offset = sample * packed_info_size;
                let dst = &mut missing_chunk[dst_offset..dst_offset + packed_info_size];
                for idx in 0..packed_info_size {
                    dst[idx] += contrib[idx];
                }
            }
        });
}

fn apply_missing_coordinates_sparse_serial(
    missing_coords: &[MissingCoord],
    packed_info_size: usize,
    block_info_contrib: &[f64],
    missing_info_storage: &mut [f64],
) {
    for coord in missing_coords {
        let sample = coord.sample as usize;
        let variant = coord.variant as usize;
        let sample_offset = sample * packed_info_size;
        let contrib =
            &block_info_contrib[variant * packed_info_size..(variant + 1) * packed_info_size];
        let dst = &mut missing_info_storage[sample_offset..sample_offset + packed_info_size];
        for idx in 0..packed_info_size {
            dst[idx] += contrib[idx];
        }
    }
}

fn sparse_sample_range_chunk(range_count: usize) -> usize {
    if range_count >= 1_000_000 {
        4_096
    } else if range_count >= 100_000 {
        2_048
    } else if range_count >= 10_000 {
        1_024
    } else {
        256
    }
}

fn lower_bound_by_sample(coords: &[MissingCoord], sample: u32) -> usize {
    let mut left = 0usize;
    let mut right = coords.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if coords[mid].sample < sample {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn accumulate_missing_info_from_raw(
    raw_block: &[f64],
    n_samples: usize,
    filled: usize,
    packed_info_size: usize,
    block_info_contrib: &[f64],
    missing_info_storage: &mut [f64],
    par: Par,
) {
    let prefer_serial = n_samples <= 128 || filled <= 16 || par.degree() <= 1;
    if prefer_serial {
        for j_local in 0..filled {
            let contrib =
                &block_info_contrib[j_local * packed_info_size..(j_local + 1) * packed_info_size];
            let column = &raw_block[j_local * n_samples..(j_local + 1) * n_samples];
            for (sample, &raw) in column.iter().enumerate() {
                if raw.is_finite() {
                    continue;
                }
                let sample_offset = sample * packed_info_size;
                let missing_info =
                    &mut missing_info_storage[sample_offset..sample_offset + packed_info_size];
                for idx in 0..packed_info_size {
                    missing_info[idx] += contrib[idx];
                }
            }
        }
        return;
    }

    let chunk_samples = missing_scan_chunk_samples();
    let chunk_width = chunk_samples * packed_info_size;
    missing_info_storage
        .par_chunks_mut(chunk_width)
        .enumerate()
        .for_each(|(chunk_idx, chunk_missing)| {
            let sample_start = chunk_idx * chunk_samples;
            let chunk_len = chunk_missing.len() / packed_info_size;

            for j_local in 0..filled {
                let contrib = &block_info_contrib
                    [j_local * packed_info_size..(j_local + 1) * packed_info_size];
                let column = &raw_block[j_local * n_samples..(j_local + 1) * n_samples];

                for local in 0..chunk_len {
                    let sample = sample_start + local;
                    let raw = column[sample];
                    if raw.is_finite() {
                        continue;
                    }
                    let dst_offset = local * packed_info_size;
                    let missing_info =
                        &mut chunk_missing[dst_offset..dst_offset + packed_info_size];
                    for idx in 0..packed_info_size {
                        missing_info[idx] += contrib[idx];
                    }
                }
            }
        });
}

fn accumulate_missing_info_from_present(
    raw_block: &[f64],
    n_samples: usize,
    filled: usize,
    packed_info_size: usize,
    block_info_contrib: &[f64],
    missing_info_storage: &mut [f64],
    par: Par,
) {
    if filled == 0 || n_samples == 0 {
        return;
    }

    let mut block_global = vec![0.0f64; packed_info_size];
    for j_local in 0..filled {
        let contrib =
            &block_info_contrib[j_local * packed_info_size..(j_local + 1) * packed_info_size];
        for idx in 0..packed_info_size {
            block_global[idx] += contrib[idx];
        }
    }

    let chunk_samples = missing_scan_chunk_samples();
    let chunk_width = chunk_samples * packed_info_size;

    let serial = par.degree() <= 1 || n_samples <= chunk_samples;
    if serial {
        let mut present = vec![0.0f64; packed_info_size];
        for sample in 0..n_samples {
            present.fill(0.0);
            for j_local in 0..filled {
                let raw = raw_block[j_local * n_samples + sample];
                if !raw.is_finite() {
                    continue;
                }
                let contrib = &block_info_contrib
                    [j_local * packed_info_size..(j_local + 1) * packed_info_size];
                for idx in 0..packed_info_size {
                    present[idx] += contrib[idx];
                }
            }
            let dst_offset = sample * packed_info_size;
            let dst = &mut missing_info_storage[dst_offset..dst_offset + packed_info_size];
            for idx in 0..packed_info_size {
                dst[idx] += block_global[idx] - present[idx];
            }
        }
        return;
    }

    missing_info_storage
        .par_chunks_mut(chunk_width)
        .enumerate()
        .for_each(|(chunk_idx, chunk_missing)| {
            let sample_start = chunk_idx * chunk_samples;
            let chunk_len = chunk_missing.len() / packed_info_size;
            let mut present_chunk = vec![0.0f64; chunk_len * packed_info_size];

            for j_local in 0..filled {
                let contrib = &block_info_contrib
                    [j_local * packed_info_size..(j_local + 1) * packed_info_size];
                let column = &raw_block[j_local * n_samples..(j_local + 1) * n_samples];
                for local in 0..chunk_len {
                    let sample = sample_start + local;
                    if !column[sample].is_finite() {
                        continue;
                    }
                    let p_offset = local * packed_info_size;
                    let present = &mut present_chunk[p_offset..p_offset + packed_info_size];
                    for idx in 0..packed_info_size {
                        present[idx] += contrib[idx];
                    }
                }
            }

            for local in 0..chunk_len {
                let p_offset = local * packed_info_size;
                let present = &present_chunk[p_offset..p_offset + packed_info_size];
                let dst = &mut chunk_missing[p_offset..p_offset + packed_info_size];
                for idx in 0..packed_info_size {
                    dst[idx] += block_global[idx] - present[idx];
                }
            }
        });
}

fn missing_scan_chunk_samples() -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("GNOMON_PROJECT_MISSING_CHUNK_SAMPLES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(256)
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::sync::Arc;

    use super::super::fit::{FitOptions, LdConfig, LdWindow};
    use super::super::progress::NoopFitProgress;
    use serde_json::Value;

    const N_SAMPLES: usize = 3;
    const N_VARIANTS: usize = 4;
    const TOLERANCE: f64 = 1e-10;
    const TEST_COMPONENTS: usize = 2;

    fn sample_data() -> Vec<f64> {
        vec![
            0.0, 1.0, 2.0, // variant 0
            1.0, 2.0, 0.0, // variant 1
            2.0, 1.0, 0.0, // variant 2
            1.0, 0.0, 2.0, // variant 3
        ]
    }

    fn fit_example_model() -> HwePcaModel {
        let data = sample_data();
        let mut source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("source");
        HwePcaModel::fit_k(&mut source, TEST_COMPONENTS).expect("model fit")
    }

    fn fit_example_model_with_ld() -> HwePcaModel {
        let data = sample_data();
        let mut source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("source");
        let options = FitOptions {
            ld: Some(LdConfig {
                window: Some(LdWindow::Sites(3)),
                ridge: Some(1.0e-3),
                variant_keys: None,
            }),
        };
        let progress = Arc::new(NoopFitProgress::default());
        HwePcaModel::fit_k_with_options_and_progress(
            &mut source,
            TEST_COMPONENTS,
            &options,
            &progress,
        )
        .expect("model fit with ld")
    }

    fn model_with_zero_norm_component() -> HwePcaModel {
        let model = fit_example_model();
        let components = model.components();
        assert!(
            components >= 2,
            "example model should expose multiple components"
        );
        let zero_idx = components - 1;
        let mut value = serde_json::to_value(&model).expect("serialize model");

        if let Some(loadings) = value.get_mut("loadings").and_then(Value::as_object_mut) {
            let nrows = loadings
                .get("nrows")
                .and_then(Value::as_u64)
                .expect("loadings nrows") as usize;
            if let Some(data) = loadings.get_mut("data").and_then(Value::as_array_mut) {
                for row in 0..nrows {
                    let idx = zero_idx * nrows + row;
                    data[idx] = Value::from(0.0);
                }
            }
        }

        if let Some(scores) = value
            .get_mut("sample_scores")
            .and_then(Value::as_object_mut)
        {
            let nrows = scores
                .get("nrows")
                .and_then(Value::as_u64)
                .expect("scores nrows") as usize;
            if let Some(data) = scores.get_mut("data").and_then(Value::as_array_mut) {
                for row in 0..nrows {
                    let idx = zero_idx * nrows + row;
                    data[idx] = Value::from(0.0);
                }
            }
        }

        if let Some(singular_values) = value
            .get_mut("singular_values")
            .and_then(Value::as_array_mut)
        {
            singular_values[zero_idx] = Value::from(0.0);
        }
        if let Some(eigenvalues) = value.get_mut("eigenvalues").and_then(Value::as_array_mut) {
            if zero_idx < eigenvalues.len() {
                eigenvalues[zero_idx] = Value::from(0.0);
            }
        }
        if let Some(norms) = value
            .get_mut("component_weighted_norms_sq")
            .and_then(Value::as_array_mut)
        {
            norms[zero_idx] = Value::from(0.0);
        }

        serde_json::from_value(value).expect("deserialize mutated model")
    }

    fn recompute_component_norms_sq(model: &HwePcaModel) -> Vec<f64> {
        let loadings = model.variant_loadings();
        let mut result = Vec::with_capacity(loadings.ncols());

        for col in 0..loadings.ncols() {
            let mut sum = 0.0f64;
            let mut compensation = 0.0f64;
            for row in 0..loadings.nrows() {
                let weight = 1.0;
                let value = loadings[(row, col)];
                let weighted = weight * value;
                let square = weighted * weighted;
                let y = square - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }

            if !sum.is_finite() || sum < 0.0 {
                sum = 0.0;
            }
            result.push(sum);
        }

        result
    }

    #[test]
    fn cached_component_norms_match_manual_computation() {
        let model = fit_example_model();
        let manual = recompute_component_norms_sq(&model);
        let cached = model.component_weighted_norms_sq();
        assert_eq!(manual.len(), cached.len());
        for (idx, (expected, actual)) in manual.iter().zip(cached.iter()).enumerate() {
            assert!(
                (expected - actual).abs() <= 1e-12,
                "component {idx} mismatch: expected {expected}, cached {actual}",
            );
        }
    }

    #[test]
    fn cached_component_norms_match_with_ld_weights() {
        let model = fit_example_model_with_ld();
        let manual = recompute_component_norms_sq(&model);
        let cached = model.component_weighted_norms_sq();
        assert_eq!(manual.len(), cached.len());
        for (idx, (expected, actual)) in manual.iter().zip(cached.iter()).enumerate() {
            assert!(
                (expected - actual).abs() <= 1e-12,
                "component {idx} mismatch: expected {expected}, cached {actual}",
            );
        }
    }

    #[test]
    fn cached_component_norms_handle_zero_norm_components() {
        let model = model_with_zero_norm_component();
        let manual = recompute_component_norms_sq(&model);
        let cached = model.component_weighted_norms_sq();
        assert_eq!(manual.len(), cached.len());
        let mut saw_zero = false;
        for (idx, (expected, actual)) in manual.iter().zip(cached.iter()).enumerate() {
            if expected.abs() <= 1e-12 {
                saw_zero = true;
            }
            assert!(
                (expected - actual).abs() <= 1e-12,
                "component {idx} mismatch: expected {expected}, cached {actual}",
            );
        }
        assert!(saw_zero, "expected at least one near-zero norm component");
    }

    fn assert_mats_close(a: &Mat<f64>, b: &Mat<f64>, tol: f64) {
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), b.ncols());
        for row in 0..a.nrows() {
            for col in 0..a.ncols() {
                let left = a[(row, col)];
                let right = b[(row, col)];
                if left.is_nan() && right.is_nan() {
                    continue;
                }
                assert!(
                    (left - right).abs() <= tol,
                    "mismatch at ({row}, {col}): {left} vs {right}"
                );
            }
        }
    }

    fn set_sample_to_nan(data: &mut [f64], sample_idx: usize) {
        for variant in 0..N_VARIANTS {
            data[variant * N_SAMPLES + sample_idx] = f64::NAN;
        }
    }

    fn set_variant_to_nan(data: &mut [f64], variant_idx: usize) {
        let start = variant_idx * N_SAMPLES;
        let end = start + N_SAMPLES;
        for value in &mut data[start..end] {
            *value = f64::NAN;
        }
    }
    #[test]
    fn renormalization_matches_baseline_without_missingness() {
        let model = fit_example_model();
        let data = sample_data();

        let mut baseline_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("baseline");
        let baseline = model
            .projector()
            .project(&mut baseline_source)
            .expect("baseline projection");

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let result = model
            .projector()
            .project_with_options(&mut renorm_source, &options)
            .expect("renormalized projection");
        assert_mats_close(&baseline, &result.scores, 1e-10);

        let alignment = result.alignment.expect("alignment");
        for row in 0..alignment.nrows() {
            for col in 0..alignment.ncols() {
                let norm = alignment[(row, col)];
                assert!(
                    (norm - 1.0).abs() <= 1e-12,
                    "alignment mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn ld_weighted_renormalization_matches_default_projection_without_missingness() {
        let model = fit_example_model_with_ld();
        let data = sample_data();

        let mut default_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("raw");
        let default_scores = model
            .projector()
            .project(&mut default_source)
            .expect("default projection");

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let renorm_options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let renorm_result = model
            .projector()
            .project_with_options(&mut renorm_source, &renorm_options)
            .expect("renormalized projection");

        assert_mats_close(&default_scores, &renorm_result.scores, 1e-10);

        let alignment = renorm_result.alignment.expect("alignment");
        for row in 0..alignment.nrows() {
            for col in 0..alignment.ncols() {
                let norm = alignment[(row, col)];
                assert!(
                    (norm - 1.0).abs() <= 1e-12,
                    "alignment mismatch at ({row}, {col})"
                );
            }
        }
    }
    #[test]
    fn zero_alignment_behavior_respected() {
        let model = fit_example_model();
        let mut data = sample_data();
        set_sample_to_nan(&mut data, 1);

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::NaN,
        };
        let result = model
            .projector()
            .project_with_options(&mut renorm_source, &options)
            .expect("renormalized projection");
        let alignment = result.alignment.expect("alignment");

        // Check Sample 1 (Zero Info) - expects NaN/Zero alignment
        for col in 0..result.scores.ncols() {
            assert!(result.scores[(1, col)].is_nan());
            assert_eq!(alignment[(1, col)], 0.0);
        }

        // Other samples should have finite scores (WLS might differ from Raw)
        for row in [0usize, 2usize] {
            for col in 0..result.scores.ncols() {
                assert!(result.scores[(row, col)].is_finite());
            }
        }
    }
    #[test]
    fn dropping_variant_matches_manual_renormalization() {
        let model = fit_example_model();
        let mut data = sample_data();
        let dropped_variant = 1;
        set_variant_to_nan(&mut data, dropped_variant);

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let result = model
            .projector()
            .project_with_options(&mut renorm_source, &options)
            .expect("renormalized projection");
        let alignment = result.alignment.expect("alignment");

        // WLS test: with one variant missing, alignment should be < 1.0 for
        // components where that variant contributes, and scores should be finite.
        let loadings = model.variant_loadings();
        for col in 0..result.scores.ncols() {
            let missing_loading = loadings[(dropped_variant, col)];
            for row in 0..result.scores.nrows() {
                let score = result.scores[(row, col)];
                let align = alignment[(row, col)];

                // Score should be finite (WLS succeeded)
                assert!(
                    score.is_finite(),
                    "score should be finite at ({row}, {col})"
                );

                // Alignment should reflect missing data
                if missing_loading.abs() > 1e-10 {
                    // If the dropped variant has non-zero loading, alignment < 1
                    assert!(
                        align < 1.0 - 1e-12,
                        "alignment should be < 1 at ({row}, {col}) when variant is missing"
                    );
                }
                assert!(
                    align > 0.0,
                    "alignment should be positive at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn ld_weighted_missingness_matches_manual_renormalization() {
        let model = fit_example_model_with_ld();
        let mut data = sample_data();
        // Make some variants missing for specific samples
        data[1 * N_SAMPLES + 0] = f64::NAN; // variant 1, sample 0
        data[3 * N_SAMPLES + 2] = f64::NAN; // variant 3, sample 2

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let renorm_options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let renorm_result = model
            .projector()
            .project_with_options(&mut renorm_source, &renorm_options)
            .expect("renormalized projection");

        let alignment = renorm_result.alignment.expect("alignment");
        let renorm_scores = renorm_result.scores;

        // WLS test: scores should be finite, alignment should reflect missingness
        for row in 0..renorm_scores.nrows() {
            for col in 0..renorm_scores.ncols() {
                let score = renorm_scores[(row, col)];
                let align = alignment[(row, col)];

                // Score should be finite
                assert!(
                    score.is_finite(),
                    "score should be finite at ({row}, {col})"
                );

                // Alignment should be positive
                assert!(
                    align > 0.0,
                    "alignment should be positive at ({row}, {col})"
                );

                // Samples with missing data should have alignment < 1
                // Sample 0 has variant 1 missing, sample 2 has variant 3 missing
                if row == 0 || row == 2 {
                    assert!(
                        align < 1.0,
                        "alignment should be < 1 for sample {row} with missing data"
                    );
                }
            }
        }

        // Sample 1 has no missing data, alignment should be positive/finite
        for col in 0..renorm_scores.ncols() {
            let align = alignment[(1, col)];
            assert!(
                align > 0.0 && align.is_finite(),
                "alignment should be finite positive"
            );
        }
    }

    #[test]
    fn projection_requires_missing_axis_renormalization() {
        let model = fit_example_model();
        let data = sample_data();

        let mut source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("source");
        let options = ProjectionOptions {
            missing_axis_renormalization: false,
            return_alignment: false,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };

        let err = model
            .projector()
            .project_with_options(&mut source, &options)
            .expect_err("projection without renormalization should fail");

        assert!(matches!(
            err,
            HwePcaError::InvalidInput("Projection requires missing axis renormalization")
        ));
    }

    struct PackedDenseBlockSource {
        dense: Vec<f64>,
        packed: Vec<u8>,
        n_samples: usize,
        n_variants: usize,
        selection: Option<Vec<usize>>,
        match_kinds: Option<Vec<MatchKind>>,
        cursor: usize,
    }

    impl PackedDenseBlockSource {
        fn new(dense: Vec<f64>, n_samples: usize, n_variants: usize) -> Self {
            Self::new_with_selection(dense, n_samples, n_variants, None, None)
        }

        fn new_with_selection(
            dense: Vec<f64>,
            n_samples: usize,
            n_variants: usize,
            selection: Option<Vec<usize>>,
            match_kinds: Option<Vec<MatchKind>>,
        ) -> Self {
            let bytes_per_variant = (n_samples + 3) / 4;
            let mut packed = vec![0u8; bytes_per_variant * n_variants];
            for variant in 0..n_variants {
                for sample in 0..n_samples {
                    let value = dense[variant * n_samples + sample];
                    let code = if value.is_nan() {
                        1u8
                    } else if value == 0.0 {
                        0u8
                    } else if value == 1.0 {
                        2u8
                    } else if value == 2.0 {
                        3u8
                    } else {
                        panic!("packed test source only supports 0/1/2/NaN");
                    };
                    let byte_offset = variant * bytes_per_variant + (sample / 4);
                    let shift = (sample % 4) * 2;
                    packed[byte_offset] |= code << shift;
                }
            }
            Self {
                dense,
                packed,
                n_samples,
                n_variants,
                selection,
                match_kinds,
                cursor: 0,
            }
        }
    }

    impl VariantBlockSource for PackedDenseBlockSource {
        type Error = Infallible;

        fn n_samples(&self) -> usize {
            self.n_samples
        }

        fn n_variants(&self) -> usize {
            self.selection
                .as_ref()
                .map(|selection| selection.len())
                .unwrap_or(self.n_variants)
        }

        fn reset(&mut self) -> Result<(), Self::Error> {
            self.cursor = 0;
            Ok(())
        }

        fn next_block_into(
            &mut self,
            max_variants: usize,
            storage: &mut [f64],
        ) -> Result<usize, Self::Error> {
            if max_variants == 0 {
                return Ok(0);
            }
            let remaining = self.n_variants.saturating_sub(self.cursor);
            if let Some(selection) = self.selection.as_ref() {
                let remaining = selection.len().saturating_sub(self.cursor);
                if remaining == 0 {
                    return Ok(0);
                }
                let filled = remaining.min(max_variants);
                let kinds = self.match_kinds.as_deref();
                for local in 0..filled {
                    let variant = selection[self.cursor + local];
                    let start = variant * self.n_samples;
                    let end = start + self.n_samples;
                    let dest = &mut storage[local * self.n_samples..(local + 1) * self.n_samples];
                    dest.copy_from_slice(&self.dense[start..end]);
                    if let Some(kinds) = kinds
                        && kinds[self.cursor + local] == MatchKind::Swap
                    {
                        for value in dest.iter_mut() {
                            if !value.is_nan() {
                                *value = 2.0 - *value;
                            }
                        }
                    }
                }
                self.cursor += filled;
                return Ok(filled);
            }

            if remaining == 0 {
                return Ok(0);
            }
            let filled = remaining.min(max_variants);
            let len = filled * self.n_samples;
            let start = self.cursor * self.n_samples;
            let end = start + len;
            storage[..len].copy_from_slice(&self.dense[start..end]);
            self.cursor += filled;
            Ok(filled)
        }

        fn hard_call_packed(&mut self) -> Option<super::super::fit::HardCallPacked<'_>> {
            let bytes_per_variant = (self.n_samples + 3) / 4;
            match self.selection.as_deref() {
                Some(selection) => Some(super::super::fit::HardCallPacked::new_selected(
                    &self.packed,
                    bytes_per_variant,
                    self.n_variants,
                    selection,
                    self.match_kinds.as_deref(),
                )),
                None => Some(super::super::fit::HardCallPacked::new(
                    &self.packed,
                    bytes_per_variant,
                    self.n_variants,
                )),
            }
        }
    }

    #[test]
    fn packed_hardcall_path_matches_dense_projection() {
        let model = fit_example_model();
        let mut data = sample_data();
        set_variant_to_nan(&mut data, 1);

        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };

        let mut dense_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("dense");
        let dense = model
            .projector()
            .project_with_options(&mut dense_source, &options)
            .expect("dense projection");

        let mut packed_source = PackedDenseBlockSource::new(data, N_SAMPLES, N_VARIANTS);
        let packed = model
            .projector()
            .project_with_options(&mut packed_source, &options)
            .expect("packed projection");

        assert_mats_close(&dense.scores, &packed.scores, 1e-10);
        let dense_alignment = dense.alignment.expect("dense alignment");
        let packed_alignment = packed.alignment.expect("packed alignment");
        assert_mats_close(&dense_alignment, &packed_alignment, 1e-10);
    }

    #[test]
    fn packed_hardcall_path_handles_ordered_selection_and_swaps() {
        let model = fit_example_model();
        let data = sample_data();
        let selection = vec![2usize, 0usize, 3usize, 1usize];
        let match_kinds = vec![
            MatchKind::Exact,
            MatchKind::Swap,
            MatchKind::Exact,
            MatchKind::Exact,
        ];

        let mut selected_dense = Vec::with_capacity(selection.len() * N_SAMPLES);
        for (out_idx, &variant) in selection.iter().enumerate() {
            for sample in 0..N_SAMPLES {
                let mut value = data[variant * N_SAMPLES + sample];
                if match_kinds[out_idx] == MatchKind::Swap && !value.is_nan() {
                    value = 2.0 - value;
                }
                selected_dense.push(value);
            }
        }

        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };

        let mut dense_source =
            DenseBlockSource::new(&selected_dense, N_SAMPLES, selection.len()).expect("dense");
        let dense = model
            .projector()
            .project_with_options(&mut dense_source, &options)
            .expect("dense projection");

        let mut packed_source = PackedDenseBlockSource::new_with_selection(
            data,
            N_SAMPLES,
            N_VARIANTS,
            Some(selection),
            Some(match_kinds),
        );
        let packed = model
            .projector()
            .project_with_options(&mut packed_source, &options)
            .expect("packed projection");

        assert_mats_close(&dense.scores, &packed.scores, 1e-10);
        let dense_alignment = dense.alignment.expect("dense alignment");
        let packed_alignment = packed.alignment.expect("packed alignment");
        assert_mats_close(&dense_alignment, &packed_alignment, 1e-10);
    }

    #[test]
    fn single_sample_projection_is_stable_with_missingness() {
        let model = fit_example_model();
        let mut data = sample_data();
        set_sample_to_nan(&mut data, 0);

        let one_sample_variants: Vec<f64> = (0..N_VARIANTS)
            .map(|variant| data[variant * N_SAMPLES])
            .collect();
        let mut source = DenseBlockSource::new(&one_sample_variants, 1, N_VARIANTS)
            .expect("single sample source");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let result = model
            .projector()
            .project_with_options(&mut source, &options)
            .expect("single sample projection");

        assert_eq!(result.scores.nrows(), 1);
        assert_eq!(result.scores.ncols(), model.components());
        for col in 0..result.scores.ncols() {
            assert!(
                result.scores[(0, col)].is_finite(),
                "single-sample score should be finite at component {col}"
            );
        }
        let alignment = result.alignment.expect("alignment");
        for col in 0..alignment.ncols() {
            assert!(
                alignment[(0, col)].is_finite(),
                "single-sample alignment should be finite at component {col}"
            );
        }
    }
    struct ChunkedBlockSource<'a> {
        data: &'a [f64],
        dims: (usize, usize),
        cursor: usize,
        chunk: usize,
    }

    impl<'a> ChunkedBlockSource<'a> {
        fn new(data: &'a [f64], n_samples: usize, n_variants: usize, chunk: usize) -> Self {
            Self {
                data,
                dims: (n_samples, n_variants),
                cursor: 0,
                chunk: chunk.max(1),
            }
        }
    }

    impl<'a> VariantBlockSource for ChunkedBlockSource<'a> {
        type Error = Infallible;

        fn n_samples(&self) -> usize {
            self.dims.0
        }

        fn n_variants(&self) -> usize {
            self.dims.1
        }

        fn reset(&mut self) -> Result<(), Self::Error> {
            self.cursor = 0;
            Ok(())
        }

        fn next_block_into(
            &mut self,
            max_variants: usize,
            storage: &mut [f64],
        ) -> Result<usize, Self::Error> {
            if max_variants == 0 {
                return Ok(0);
            }
            let remaining = self.n_variants().saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(0);
            }
            let ncols = remaining.min(self.chunk).min(max_variants);
            let nrows = self.n_samples();
            let len = nrows * ncols;
            let start = self.cursor * nrows;
            let end = start + len;
            storage[..len].copy_from_slice(&self.data[start..end]);
            self.cursor += ncols;
            Ok(ncols)
        }
    }

    #[test]
    fn block_boundary_missingness_is_stable() {
        let model = fit_example_model();
        let mut data = sample_data();
        data[1 * N_SAMPLES + 0] = f64::NAN;
        data[2 * N_SAMPLES + 2] = f64::NAN;

        let mut dense_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("dense");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let dense_result = model
            .projector()
            .project_with_options(&mut dense_source, &options)
            .expect("dense projection");

        let mut chunked_source = ChunkedBlockSource::new(&data, N_SAMPLES, N_VARIANTS, 2);
        let chunked_result = model
            .projector()
            .project_with_options(&mut chunked_source, &options)
            .expect("chunked projection");

        assert_mats_close(&dense_result.scores, &chunked_result.scores, TOLERANCE);

        let dense_alignment = dense_result.alignment.expect("dense alignment");
        let chunked_alignment = chunked_result.alignment.expect("chunked alignment");
        assert_mats_close(&dense_alignment, &chunked_alignment, TOLERANCE);
    }
}
