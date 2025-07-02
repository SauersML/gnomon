// Simple test to verify bfgs crate works with ndarray
use bfgs;
use ndarray::Array1;

fn main() {
    // Test the simple bfgs API
    let x0 = Array1::from_vec(vec![8.888, 1.234]);
    let f = |x: &Array1<f64>| x.dot(x);
    let g = |x: &Array1<f64>| 2.0 * x;
    
    match bfgs::bfgs(x0, f, g) {
        Ok(x_min) => println!("BFGS optimization successful: {:?}", x_min),
        Err(e) => println!("BFGS optimization failed: {:?}", e),
    }
}
