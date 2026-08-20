use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_gnomon")]
fn gnomon_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    gnomon::python_ext::register_python_module(module)
}
