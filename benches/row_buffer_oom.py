"""Exercise a production row-buffer growth failure using the warm MSI build."""
import subprocess
from build_cached_probe import build, root

source = (root / 'src/score/io.rs').read_text()
start = source.index('fn prepare_pooled_buffer(')
end = source.index('\n}\n', start) + 3
function = source[start:end]
harness = r'''
use gnomon::pipeline_error::PipelineError;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, Ordering};
static FAIL_NEXT: AtomicBool = AtomicBool::new(false);
struct FailOneAllocation;
unsafe impl GlobalAlloc for FailOneAllocation {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FAIL_NEXT.swap(false, Ordering::SeqCst) {
            std::ptr::null_mut()
        } else {
            unsafe { System.alloc(layout) }
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if FAIL_NEXT.swap(false, Ordering::SeqCst) {
            std::ptr::null_mut()
        } else {
            unsafe { System.realloc(ptr, layout, size) }
        }
    }
}
#[global_allocator]
static ALLOCATOR: FailOneAllocation = FailOneAllocation;
fn main() {
    let buffer = vec![1u8; 8];
    let requested = buffer.capacity() + 1;
    FAIL_NEXT.store(true, Ordering::SeqCst);
    let result = prepare_pooled_buffer(buffer, requested);
    assert!(result.is_err(), "growth must report allocation failure");
    assert!(!FAIL_NEXT.load(Ordering::SeqCst), "failure was exercised");
    let result = prepare_pooled_buffer(vec![1u8; 8], requested).unwrap();
    assert_eq!(result, vec![0u8; requested]);
    println!("row-buffer growth: injected allocation failure returned an error; normal growth passed");
}
'''
probe = root / 'row_buffer_oom.rs'
probe.write_text(harness + function)
build(probe, 'row-buffer-oom', ['-C', 'panic=abort'])
subprocess.run([str(root / 'row-buffer-oom')], check=True, timeout=5)
