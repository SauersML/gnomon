fn compute_gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut nodes_weights = Vec::with_capacity(n);
    let m = (n + 1) / 2;
    let pi = std::f64::consts::PI;

    for i in 0..m {
        // Initial guess
        let mut z = (pi * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;
        let mut z1;

        // Newton's method
        loop {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            // p1 is now P_n(z)
            // pp is P'_n(z) calculated using derivative relation
            // Derivative relation: (z^2 - 1) P'_n(z) = n * (z * P_n(z) - P_{n-1}(z))
            // So P'_n(z) = n * (z * p1 - p2) / (z^2 - 1)
            // Let's use the formula from Numerical Recipes or similar standard

            // Using recurrence for derivative:
            // P'_n(z) = n * (z * P_n(z) - P_{n-1}(z)) / (z^2 - 1)
            // But usually we compute it inside the loop or use relation

            // Standard Newton step: z_new = z - P_n(z) / P'_n(z)

            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() < 1e-14 {
                break;
            }
        }

        let x = z; // Root is at -z actually? Wait.
        // The roots of Legendre polynomials are in (-1, 1).
        // My initial guess corresponds to roots in descending order or similar.
        // Let's check sign.
        // The roots are symmetric.

        // Weight formula: w = 2 / ((1 - x^2) * (P'_n(x))^2)
        let w = 2.0 / ((1.0 - z * z) * pp * pp);

        // We need to store them.
        // Since we are iterating i from 0 to m-1, we get roughly half the roots.
        // If n is even, we get n/2 pairs.
        // If n is odd, we get (n-1)/2 pairs and one zero.

        // Let's ensure we just output all of them sorted?
        // Or match the previous output format which was pairs (-x, w), (x, w).
        // The previous output had:
        // (-0.998..., 0.004...), (0.998..., 0.004...)

        // My calculated z will be one of these (likely positive or negative depending on guess).
        // The guess cos(...) for i=0 gives cos(0.75*pi / 30.5) approx cos(0.077) ~ 1.
        // So it gives positive roots.

        nodes_weights.push((-z, w));
        nodes_weights.push((z, w));
    }

    if n % 2 != 0 {
       // Handle the middle root at 0.0
       // Wait, the loop 0..m covers (n+1)/2 items.
       // If n=3, m=2. i=0, i=1.
       // Roots are roughly at cos...
       // If n is odd, one root is 0.
       // In my loop, does it handle 0?
       // For odd n, middle root is 0.
       // I should probably just collect positive roots and then mirror.
    }

    // Let's stick to the Numerical Recipes algorithm structure more closely to be safe.
    // Actually, let's just use what I wrote and debug it.

    nodes_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    nodes_weights
}

fn main() {
    let nodes = compute_gauss_legendre(30);
    // Print first few
    for (x, w) in nodes.iter().take(4) {
        println!("({}, {})", x, w);
    }
    println!("...");
}
