use cudarc::driver::DriverError;
use cudarc::nvrtc::{result as nvrtc_result, sys as nvrtc_sys};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs;
use std::path::Path;

fn cuda_library_family(basename: &str) -> Option<&'static str> {
    let stem = basename.split(".so").next()?;
    match stem {
        "libcuda" => Some("libcuda"),
        "libcudart" => Some("libcudart"),
        "libcublas" => Some("libcublas"),
        "libcublasLt" => Some("libcublasLt"),
        "libcusparse" => Some("libcusparse"),
        "libcusolver" => Some("libcusolver"),
        "libnvJitLink" => Some("libnvJitLink"),
        "libnvrtc" => Some("libnvrtc"),
        _ => None,
    }
}

/// CUDA libraries currently mapped into the process, grouped by SONAME family.
pub(crate) fn cuda_library_mappings() -> BTreeMap<&'static str, BTreeSet<String>> {
    let mut by_family = BTreeMap::new();
    let Ok(maps) = fs::read_to_string("/proc/self/maps") else {
        return by_family;
    };
    for line in maps.lines() {
        let Some(path) = line.split_whitespace().last() else {
            continue;
        };
        if !path.starts_with('/') {
            continue;
        }
        let name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if let Some(family) = cuda_library_family(name) {
            by_family
                .entry(family)
                .or_insert_with(BTreeSet::new)
                .insert(path.to_string());
        }
    }
    by_family
}

/// Refuse to initialize CUDA when two physical libraries share one SONAME.
pub(crate) fn detect_cuda_library_conflicts() -> Result<(), String> {
    let conflicts: Vec<(&'static str, Vec<String>)> = cuda_library_mappings()
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .map(|(family, paths)| (family, paths.into_iter().collect()))
        .collect();
    if conflicts.is_empty() {
        return Ok(());
    }
    let mut message = String::from(
        "CUDA library conflict: multiple distinct files share a SONAME and coexist in this \
         process. glibc dlopen deduplicates by (device, inode) not by SONAME, so all of the \
         following are simultaneously mapped. cuBLAS handle state would split across them and \
         the next cublasDestroy_v2 would abort with 'double free or corruption (!prev)'.",
    );
    for (family, paths) in &conflicts {
        message.push_str(&format!("\n  {family}:"));
        for path in paths {
            message.push_str(&format!("\n    {path}"));
        }
    }
    message.push_str(
        "\nKeep exactly one CUDA toolkit reachable to the loader: either the system toolkit \
         (usually /usr/local/cuda*) or the pip nvidia-*-cu12 wheels, not both. Skipping cuBLAS \
         handle creation to avoid crashing later.",
    );
    Err(message)
}

pub(crate) fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

pub(crate) fn should_retry_with_cubin(load_error: DriverError) -> bool {
    matches!(
        load_error.0,
        cudarc::driver::sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION
            | cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_PTX
    )
}

pub(crate) fn compile_cubin_for_device(
    kernel_source: &str,
    compute_capability_major: i32,
    compute_capability_minor: i32,
) -> Result<Vec<u8>, String> {
    let architecture =
        format!("--gpu-architecture=sm_{compute_capability_major}{compute_capability_minor}");
    let source = CString::new(kernel_source)
        .map_err(|_| "CUDA kernel source contained interior NUL".to_string())?;
    let program = nvrtc_result::create_program(source.as_c_str(), None)
        .map_err(|error| format!("NVRTC create_program failed for CUBIN fallback: {error:?}"))?;

    let compile_result =
        unsafe { nvrtc_result::compile_program(program, &[architecture.as_str()]) };
    if let Err(error) = compile_result {
        let log = nvrtc_program_log(program).unwrap_or_else(|| "<no NVRTC log available>".into());
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        return Err(format!(
            "NVRTC CUBIN fallback compile failed ({error:?}) with {architecture}: {log}"
        ));
    }

    let mut cubin_size = 0usize;
    let size_result = unsafe { nvrtc_sys::nvrtcGetCUBINSize(program, &mut cubin_size as *mut _) };
    if let Err(error) = size_result.result() {
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        return Err(format!("NVRTC CUBIN fallback get size failed: {error:?}"));
    }

    let mut cubin_raw: Vec<std::ffi::c_char> = vec![0; cubin_size];
    let cubin_result = unsafe { nvrtc_sys::nvrtcGetCUBIN(program, cubin_raw.as_mut_ptr()) };
    if let Err(error) = cubin_result.result() {
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        return Err(format!("NVRTC CUBIN fallback get data failed: {error:?}"));
    }

    if let Err(error) = unsafe { nvrtc_result::destroy_program(program) } {
        return Err(format!(
            "NVRTC CUBIN fallback destroy_program failed: {error:?}"
        ));
    }

    Ok(cubin_raw.into_iter().map(|byte| byte as u8).collect())
}

fn nvrtc_program_log(program: nvrtc_sys::nvrtcProgram) -> Option<String> {
    let raw = unsafe { nvrtc_result::get_program_log(program) }.ok()?;
    let mut bytes: Vec<u8> = raw.into_iter().map(|byte| byte as u8).collect();
    if let Some(position) = bytes.iter().position(|&byte| byte == 0) {
        bytes.truncate(position);
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}
