# GetDoc Report - Wed, 27 Aug 2025 14:01:33 -0500

## Compiler Output (Errors and Warnings)

### Diagnostics for: default features

```text
ERROR: E0782: error[E0782]: expected a type, found a trait
   --> score/../calibrate/estimate.rs:801:18
    |
801 |         Cholesky(ndarray_linalg::cholesky::Cholesky<f64, ndarray::Ix2>),
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
help: you might be missing a type parameter
    |
799 ~     enum RobustSolver<T: ndarray_linalg::cholesky::Cholesky<f64, ndarray::Ix2>> {
800 |         // Store the factor to avoid re-factorizing per solve
801 ~         Cholesky(T),
    |

  **Explanation (E0782)**:
  > Trait objects must include the `dyn` keyword.
  > 
  > Erroneous code example:
  > 
  > ```edition2021,compile_fail,E0782
  > trait Foo {}
  > fn test(arg: Box<Foo>) {} // error!
  > ```
  > 
  > Trait objects are a way to call methods on types that are not known until
  > runtime but conform to some trait.
  > 
  > Trait objects should be formed with `Box<dyn Foo>`, but in the code above
  > `dyn` is left off.
  > 
  > This makes it harder to see that `arg` is a trait object and not a
  > simply a heap allocated type called `Foo`.
  > 
  > To fix this issue, add `dyn` before the trait name.
  > 
  > ```edition2021
  > trait Foo {}
  > fn test(arg: Box<dyn Foo>) {} // ok!
  > ```
  > 
  > This used to be allowed before edition 2021, but is now an error.
    (Diagnostic primary location: score/../calibrate/estimate.rs:801)
ERROR: E0061: error[E0061]: this function takes 2 arguments but 1 argument was supplied
    --> score/../calibrate/estimate.rs:1139:49
     |
1139 |                     let factor = if let Ok(f) = FaerLlt::new(h_faer.as_ref()) {
     |                                                 ^^^^^^^^^^^^----------------- argument #2 of type `Side` is missing
     |
note: associated function defined here
    --> /home/user/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.22.6/src/linalg/solvers.rs:587:9
     |
587  |     pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Result<Self, LltError> {
     |            ^^^
help: provide the argument
     |
1139 |                     let factor = if let Ok(f) = FaerLlt::new(h_faer.as_ref(), /* Side */) {
     |                                                                             ++++++++++++

  **Explanation (E0061)**:
  > An invalid number of arguments was passed when calling a function.
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0061
  > fn f(u: i32) {}
  > 
  > f(); // error!
  > ```
  > 
  > The number of arguments passed to a function must match the number of arguments
  > specified in the function signature.
  > 
  > For example, a function like:
  > 
  > ```
  > fn f(a: u16, b: &str) {}
  > ```
  > 
  > Must always be called with exactly two arguments, e.g., `f(2, "test")`.
  > 
  > Note that Rust does not have a notion of optional function arguments or
  > variadic functions (except for its C-FFI).
    (Diagnostic primary location: score/../calibrate/estimate.rs:1139)
ERROR: E0061: error[E0061]: this function takes 2 arguments but 1 argument was supplied
    --> score/../calibrate/estimate.rs:1141:43
     |
1141 |                     } else if let Ok(f) = FaerLdlt::new(h_faer.as_ref()) {
     |                                           ^^^^^^^^^^^^^----------------- argument #2 of type `Side` is missing
     |
note: associated function defined here
    --> /home/user/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.22.6/src/linalg/solvers.rs:624:9
     |
624  |     pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Result<Self, LdltError> {
     |            ^^^
help: provide the argument
     |
1141 |                     } else if let Ok(f) = FaerLdlt::new(h_faer.as_ref(), /* Side */) {
     |                                                                        ++++++++++++

  **Explanation (E0061)**:
  > An invalid number of arguments was passed when calling a function.
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0061
  > fn f(u: i32) {}
  > 
  > f(); // error!
  > ```
  > 
  > The number of arguments passed to a function must match the number of arguments
  > specified in the function signature.
  > 
  > For example, a function like:
  > 
  > ```
  > fn f(a: u16, b: &str) {}
  > ```
  > 
  > Must always be called with exactly two arguments, e.g., `f(2, "test")`.
  > 
  > Note that Rust does not have a notion of optional function arguments or
  > variadic functions (except for its C-FFI).
    (Diagnostic primary location: score/../calibrate/estimate.rs:1141)
ERROR: E0277: error[E0277]: cannot subtract `f64` from `usize`
    --> score/../calibrate/estimate.rs:1171:39
     |
1171 |                     let phi = dp / (n - edf).max(1e-8);
     |                                       ^ no implementation for `usize - f64`
     |
     = help: the trait `std::ops::Sub<f64>` is not implemented for `usize`
     = help: the following other types implement trait `std::ops::Sub<Rhs>`:
               `&usize` implements `std::ops::Sub<&Complex<usize>>`
               `&usize` implements `std::ops::Sub<&num_bigint::bigint::BigInt>`
               `&usize` implements `std::ops::Sub<&num_bigint::biguint::BigUint>`
               `&usize` implements `std::ops::Sub<Complex<usize>>`
               `&usize` implements `std::ops::Sub<num_bigint::bigint::BigInt>`
               `&usize` implements `std::ops::Sub<num_bigint::biguint::BigUint>`
               `&usize` implements `std::ops::Sub<usize>`
               `&usize` implements `std::ops::Sub`
             and 11 others

  **Explanation (E0277)**:
  > You tried to use a type which doesn't implement some trait in a place which
  > expected that trait.
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0277
  > // here we declare the Foo trait with a bar method
  > trait Foo {
  >     fn bar(&self);
  > }
  > 
  > // we now declare a function which takes an object implementing the Foo trait
  > fn some_func<T: Foo>(foo: T) {
  >     foo.bar();
  > }
  > 
  > fn main() {
  >     // we now call the method with the i32 type, which doesn't implement
  >     // the Foo trait
  >     some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied
  > }
  > ```
  > 
  > In order to fix this error, verify that the type you're using does implement
  > the trait. Example:
  > 
  > ```
  > trait Foo {
  >     fn bar(&self);
  > }
  > 
  > // we implement the trait on the i32 type
  > impl Foo for i32 {
  >     fn bar(&self) {}
  > }
  > 
  > fn some_func<T: Foo>(foo: T) {
  >     foo.bar(); // we can now use this method since i32 implements the
  >                // Foo trait
  > }
  > 
  > fn main() {
  >     some_func(5i32); // ok!
  > }
  > ```
  > 
  > Or in a generic context, an erroneous code example would look like:
  > 
  > ```compile_fail,E0277
  > fn some_func<T>(foo: T) {
  >     println!("{:?}", foo); // error: the trait `core::fmt::Debug` is not
  >                            //        implemented for the type `T`
  > }
  > 
  > fn main() {
  >     // We now call the method with the i32 type,
  >     // which *does* implement the Debug trait.
  >     some_func(5i32);
  > }
  > ```
  > 
  > Note that the error here is in the definition of the generic function. Although
  > we only call it with a parameter that does implement `Debug`, the compiler
  > still rejects the function. It must work with all possible input types. In
  > order to make this example compile, we need to restrict the generic type we're
  > accepting:
  > 
  > ```
  > use std::fmt;
  > 
  > // Restrict the input type to types that implement Debug.
  > fn some_func<T: fmt::Debug>(foo: T) {
  >     println!("{:?}", foo);
  > }
  > 
  > fn main() {
  >     // Calling the method is still fine, as i32 implements Debug.
  >     some_func(5i32);
  > 
  >     // This would fail to compile now:
  >     // struct WithoutDebug;
  >     // some_func(WithoutDebug);
  > }
  > ```
  > 
  > Rust only looks at the signature of the called function, as such it must
  > already specify all requirements that will be used for every type parameter.
    (Diagnostic primary location: score/../calibrate/estimate.rs:1171)
ERROR: E0277: error[E0277]: cannot subtract `f64` from `usize`
    --> score/../calibrate/estimate.rs:1173:26
     |
1173 |                     if n - edf < 1.0 {
     |                          ^ no implementation for `usize - f64`
     |
     = help: the trait `std::ops::Sub<f64>` is not implemented for `usize`
     = help: the following other types implement trait `std::ops::Sub<Rhs>`:
               `&usize` implements `std::ops::Sub<&Complex<usize>>`
               `&usize` implements `std::ops::Sub<&num_bigint::bigint::BigInt>`
               `&usize` implements `std::ops::Sub<&num_bigint::biguint::BigUint>`
               `&usize` implements `std::ops::Sub<Complex<usize>>`
               `&usize` implements `std::ops::Sub<num_bigint::bigint::BigInt>`
               `&usize` implements `std::ops::Sub<num_bigint::biguint::BigUint>`
               `&usize` implements `std::ops::Sub<usize>`
               `&usize` implements `std::ops::Sub`
             and 11 others

  **Explanation (E0277)**:
  > You tried to use a type which doesn't implement some trait in a place which
  > expected that trait.
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0277
  > // here we declare the Foo trait with a bar method
  > trait Foo {
  >     fn bar(&self);
  > }
  > 
  > // we now declare a function which takes an object implementing the Foo trait
  > fn some_func<T: Foo>(foo: T) {
  >     foo.bar();
  > }
  > 
  > fn main() {
  >     // we now call the method with the i32 type, which doesn't implement
  >     // the Foo trait
  >     some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied
  > }
  > ```
  > 
  > In order to fix this error, verify that the type you're using does implement
  > the trait. Example:
  > 
  > ```
  > trait Foo {
  >     fn bar(&self);
  > }
  > 
  > // we implement the trait on the i32 type
  > impl Foo for i32 {
  >     fn bar(&self) {}
  > }
  > 
  > fn some_func<T: Foo>(foo: T) {
  >     foo.bar(); // we can now use this method since i32 implements the
  >                // Foo trait
  > }
  > 
  > fn main() {
  >     some_func(5i32); // ok!
  > }
  > ```
  > 
  > Or in a generic context, an erroneous code example would look like:
  > 
  > ```compile_fail,E0277
  > fn some_func<T>(foo: T) {
  >     println!("{:?}", foo); // error: the trait `core::fmt::Debug` is not
  >                            //        implemented for the type `T`
  > }
  > 
  > fn main() {
  >     // We now call the method with the i32 type,
  >     // which *does* implement the Debug trait.
  >     some_func(5i32);
  > }
  > ```
  > 
  > Note that the error here is in the definition of the generic function. Although
  > we only call it with a parameter that does implement `Debug`, the compiler
  > still rejects the function. It must work with all possible input types. In
  > order to make this example compile, we need to restrict the generic type we're
  > accepting:
  > 
  > ```
  > use std::fmt;
  > 
  > // Restrict the input type to types that implement Debug.
  > fn some_func<T: fmt::Debug>(foo: T) {
  >     println!("{:?}", foo);
  > }
  > 
  > fn main() {
  >     // Calling the method is still fine, as i32 implements Debug.
  >     some_func(5i32);
  > 
  >     // This would fail to compile now:
  >     // struct WithoutDebug;
  >     // some_func(WithoutDebug);
  > }
  > ```
  > 
  > Rust only looks at the signature of the called function, as such it must
  > already specify all requirements that will be used for every type parameter.
    (Diagnostic primary location: score/../calibrate/estimate.rs:1173)
ERROR: E0277: error[E0277]: cannot subtract `f64` from `usize`
    --> score/../calibrate/estimate.rs:1212:31
     |
1212 |                         + ((n - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();
     |                               ^ no implementation for `usize - f64`
     |
     = help: the trait `std::ops::Sub<f64>` is not implemented for `usize`
     = help: the following other types implement trait `std::ops::Sub<Rhs>`:
               `&usize` implements `std::ops::Sub<&Complex<usize>>`
               `&usize` implements `std::ops::Sub<&num_bigint::bigint::BigInt>`
               `&usize` implements `std::ops::Sub<&num_bigint::biguint::BigUint>`
               `&usize` implements `std::ops::Sub<Complex<usize>>`
               `&usize` implements `std::ops::Sub<num_bigint::bigint::BigInt>`
               `&usize` implements `std::ops::Sub<num_bigint::biguint::BigUint>`
               `&usize` implements `std::ops::Sub<usize>`
               `&usize` implements `std::ops::Sub`
             and 11 others

  **Explanation (E0277)**:
  > You tried to use a type which doesn't implement some trait in a place which
  > expected that trait.
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0277
  > // here we declare the Foo trait with a bar method
  > trait Foo {
  >     fn bar(&self);
  > }
  > 
  > // we now declare a function which takes an object implementing the Foo trait
  > fn some_func<T: Foo>(foo: T) {
  >     foo.bar();
  > }
  > 
  > fn main() {
  >     // we now call the method with the i32 type, which doesn't implement
  >     // the Foo trait
  >     some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied
  > }
  > ```
  > 
  > In order to fix this error, verify that the type you're using does implement
  > the trait. Example:
  > 
  > ```
  > trait Foo {
  >     fn bar(&self);
  > }
  > 
  > // we implement the trait on the i32 type
  > impl Foo for i32 {
  >     fn bar(&self) {}
  > }
  > 
  > fn some_func<T: Foo>(foo: T) {
  >     foo.bar(); // we can now use this method since i32 implements the
  >                // Foo trait
  > }
  > 
  > fn main() {
  >     some_func(5i32); // ok!
  > }
  > ```
  > 
  > Or in a generic context, an erroneous code example would look like:
  > 
  > ```compile_fail,E0277
  > fn some_func<T>(foo: T) {
  >     println!("{:?}", foo); // error: the trait `core::fmt::Debug` is not
  >                            //        implemented for the type `T`
  > }
  > 
  > fn main() {
  >     // We now call the method with the i32 type,
  >     // which *does* implement the Debug trait.
  >     some_func(5i32);
  > }
  > ```
  > 
  > Note that the error here is in the definition of the generic function. Although
  > we only call it with a parameter that does implement `Debug`, the compiler
  > still rejects the function. It must work with all possible input types. In
  > order to make this example compile, we need to restrict the generic type we're
  > accepting:
  > 
  > ```
  > use std::fmt;
  > 
  > // Restrict the input type to types that implement Debug.
  > fn some_func<T: fmt::Debug>(foo: T) {
  >     println!("{:?}", foo);
  > }
  > 
  > fn main() {
  >     // Calling the method is still fine, as i32 implements Debug.
  >     some_func(5i32);
  > 
  >     // This would fail to compile now:
  >     // struct WithoutDebug;
  >     // some_func(WithoutDebug);
  > }
  > ```
  > 
  > Rust only looks at the signature of the called function, as such it must
  > already specify all requirements that will be used for every type parameter.
    (Diagnostic primary location: score/../calibrate/estimate.rs:1212)
ERROR: E0599: error[E0599]: no method named `read` found for struct `Mat<faer::mat::Ref<'_, f64>>` in the current scope
    --> score/../calibrate/estimate.rs:1124:42
     |
1124 | ...                   acc += a.read(i, j) * b.read(i, j);
     |                                ^^^^ method not found in `Mat<faer::mat::Ref<'_, f64>>`

  **Explanation (E0599)**:
  > This error occurs when a method is used on a type which doesn't implement it:
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0599
  > struct Mouth;
  > 
  > let x = Mouth;
  > x.chocolate(); // error: no method named `chocolate` found for type `Mouth`
  >                //        in the current scope
  > ```
  > 
  > In this case, you need to implement the `chocolate` method to fix the error:
  > 
  > ```
  > struct Mouth;
  > 
  > impl Mouth {
  >     fn chocolate(&self) { // We implement the `chocolate` method here.
  >         println!("Hmmm! I love chocolate!");
  >     }
  > }
  > 
  > let x = Mouth;
  > x.chocolate(); // ok!
  > ```
    (Diagnostic primary location: score/../calibrate/estimate.rs:1124)
ERROR: E0599: error[E0599]: no method named `read` found for struct `Mat<faer::mat::Ref<'_, f64>>` in the current scope
    --> score/../calibrate/estimate.rs:1124:57
     |
1124 | ...                   acc += a.read(i, j) * b.read(i, j);
     |                                               ^^^^ method not found in `Mat<faer::mat::Ref<'_, f64>>`

  **Explanation (E0599)**:
  > This error occurs when a method is used on a type which doesn't implement it:
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0599
  > struct Mouth;
  > 
  > let x = Mouth;
  > x.chocolate(); // error: no method named `chocolate` found for type `Mouth`
  >                //        in the current scope
  > ```
  > 
  > In this case, you need to implement the `chocolate` method to fix the error:
  > 
  > ```
  > struct Mouth;
  > 
  > impl Mouth {
  >     fn chocolate(&self) { // We implement the `chocolate` method here.
  >         println!("Hmmm! I love chocolate!");
  >     }
  > }
  > 
  > let x = Mouth;
  > x.chocolate(); // ok!
  > ```
    (Diagnostic primary location: score/../calibrate/estimate.rs:1124)
ERROR: E0061: error[E0061]: this function takes 2 arguments but 1 argument was supplied
    --> score/../calibrate/estimate.rs:1676:51
     |
1676 |                     let factor_g = if let Ok(f) = FaerLlt::new(h_faer_g.as_ref()) {
     |                                                   ^^^^^^^^^^^^------------------- argument #2 of type `Side` is missing
     |
note: associated function defined here
    --> /home/user/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.22.6/src/linalg/solvers.rs:587:9
     |
587  |     pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Result<Self, LltError> {
     |            ^^^
help: provide the argument
     |
1676 |                     let factor_g = if let Ok(f) = FaerLlt::new(h_faer_g.as_ref(), /* Side */) {
     |                                                                                 ++++++++++++

  **Explanation (E0061)**:
  > An invalid number of arguments was passed when calling a function.
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0061
  > fn f(u: i32) {}
  > 
  > f(); // error!
  > ```
  > 
  > The number of arguments passed to a function must match the number of arguments
  > specified in the function signature.
  > 
  > For example, a function like:
  > 
  > ```
  > fn f(a: u16, b: &str) {}
  > ```
  > 
  > Must always be called with exactly two arguments, e.g., `f(2, "test")`.
  > 
  > Note that Rust does not have a notion of optional function arguments or
  > variadic functions (except for its C-FFI).
    (Diagnostic primary location: score/../calibrate/estimate.rs:1676)
ERROR: E0061: error[E0061]: this function takes 2 arguments but 1 argument was supplied
    --> score/../calibrate/estimate.rs:1678:43
     |
1678 |                     } else if let Ok(f) = FaerLdlt::new(h_faer_g.as_ref()) {
     |                                           ^^^^^^^^^^^^^------------------- argument #2 of type `Side` is missing
     |
note: associated function defined here
    --> /home/user/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.22.6/src/linalg/solvers.rs:624:9
     |
624  |     pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Result<Self, LdltError> {
     |            ^^^
help: provide the argument
     |
1678 |                     } else if let Ok(f) = FaerLdlt::new(h_faer_g.as_ref(), /* Side */) {
     |                                                                          ++++++++++++

  **Explanation (E0061)**:
  > An invalid number of arguments was passed when calling a function.
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0061
  > fn f(u: i32) {}
  > 
  > f(); // error!
  > ```
  > 
  > The number of arguments passed to a function must match the number of arguments
  > specified in the function signature.
  > 
  > For example, a function like:
  > 
  > ```
  > fn f(a: u16, b: &str) {}
  > ```
  > 
  > Must always be called with exactly two arguments, e.g., `f(2, "test")`.
  > 
  > Note that Rust does not have a notion of optional function arguments or
  > variadic functions (except for its C-FFI).
    (Diagnostic primary location: score/../calibrate/estimate.rs:1678)
ERROR: E0599: error[E0599]: no method named `read` found for struct `Mat<faer::mat::Own<f64>>` in the current scope
    --> score/../calibrate/estimate.rs:1723:46
     |
1723 | ...                   acc += x.read(ii, jj) * rt.read(ii, jj);
     |                                ^^^^ method not found in `Mat<faer::mat::Own<f64>>`

  **Explanation (E0599)**:
  > This error occurs when a method is used on a type which doesn't implement it:
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0599
  > struct Mouth;
  > 
  > let x = Mouth;
  > x.chocolate(); // error: no method named `chocolate` found for type `Mouth`
  >                //        in the current scope
  > ```
  > 
  > In this case, you need to implement the `chocolate` method to fix the error:
  > 
  > ```
  > struct Mouth;
  > 
  > impl Mouth {
  >     fn chocolate(&self) { // We implement the `chocolate` method here.
  >         println!("Hmmm! I love chocolate!");
  >     }
  > }
  > 
  > let x = Mouth;
  > x.chocolate(); // ok!
  > ```
    (Diagnostic primary location: score/../calibrate/estimate.rs:1723)
ERROR: E0599: error[E0599]: no method named `read` found for struct `Mat<faer::mat::Own<f64>>` in the current scope
    --> score/../calibrate/estimate.rs:1723:64
     |
1723 | ...                   acc += x.read(ii, jj) * rt.read(ii, jj);
     |                                                  ^^^^ method not found in `Mat<faer::mat::Own<f64>>`

  **Explanation (E0599)**:
  > This error occurs when a method is used on a type which doesn't implement it:
  > 
  > Erroneous code example:
  > 
  > ```compile_fail,E0599
  > struct Mouth;
  > 
  > let x = Mouth;
  > x.chocolate(); // error: no method named `chocolate` found for type `Mouth`
  >                //        in the current scope
  > ```
  > 
  > In this case, you need to implement the `chocolate` method to fix the error:
  > 
  > ```
  > struct Mouth;
  > 
  > impl Mouth {
  >     fn chocolate(&self) { // We implement the `chocolate` method here.
  >         println!("Hmmm! I love chocolate!");
  >     }
  > }
  > 
  > let x = Mouth;
  > x.chocolate(); // ok!
  > ```
    (Diagnostic primary location: score/../calibrate/estimate.rs:1723)
```

---
## From File: `/home/user/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.22.6/src/linalg/solvers.rs`

**Referenced by:**
* NOTE (originating at `/home/user/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.22.6/src/linalg/solvers.rs:587` from configuration: `default features`)
* NOTE (originating at `/home/user/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.22.6/src/linalg/solvers.rs:624` from configuration: `default features`)

### Use Statement `linalg :: cholesky :: ldlt :: factor :: LdltError`

```rust
pub use linalg :: cholesky :: ldlt :: factor :: LdltError ;
```

### Use Statement `linalg :: cholesky :: llt :: factor :: LltError`

```rust
pub use linalg :: cholesky :: llt :: factor :: LltError ;
```

### Use Statement `linalg :: evd :: EvdError`

```rust
pub use linalg :: evd :: EvdError ;
```

### Use Statement `linalg :: svd :: SvdError`

```rust
pub use linalg :: svd :: SvdError ;
```

### Trait `ShapeCore`

> shape info of a linear system solver

```rust
pub trait ShapeCore
```

### Trait `SolveCore`

> linear system solver implementation

```rust
pub trait SolveCoreT : ComplexField
```

### Trait `SolveLstsqCore`

> least squares linear system solver implementation

```rust
pub trait SolveLstsqCoreT : ComplexField
```

### Trait `DenseSolveCore`

> dense linear system solver

```rust
pub trait DenseSolveCoreT : ComplexField
```

### Trait Impl Block `impl S : ? Sized + ShapeCore ShapeCore for & S`

```rust
impl S : ? Sized + ShapeCore ShapeCore for & S
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField , S : ? Sized + SolveCore < T > SolveCore < T > for & S`

```rust
impl T : ComplexField , S : ? Sized + SolveCore < T > SolveCore < T > for & S
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField , S : ? Sized + SolveLstsqCore < T > SolveLstsqCore < T > for & S`

```rust
impl T : ComplexField , S : ? Sized + SolveLstsqCore < T > SolveLstsqCore < T > for & S
```

#### Impl Method `solve_lstsq_in_place_with_conj`

```rust
fn solve_lstsq_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField , S : ? Sized + DenseSolveCore < T > DenseSolveCore < T > for & S`

```rust
impl T : ComplexField , S : ? Sized + DenseSolveCore < T > DenseSolveCore < T > for & S
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait `Solve`

> [`SolveCore`] extension trait

```rust
pub trait SolveT : ComplexField
```

### Inherent Impl Block `mat::generic::Mat<Inner>`

```rust
impl C : Conjugate , Inner : for < 'short > Reborrow < 'short , Target = mat :: Ref < 'short , C > > mat :: generic :: Mat < Inner >
```

#### Impl Method `partial_piv_lu`

> returns the $LU$ decomposition of `self` with partial (row) pivoting

```rust
pub fn partial_piv_lu (& self) -> PartialPivLu < C :: Canonical >;
```

#### Impl Method `full_piv_lu`

> returns the $LU$ decomposition of `self` with full pivoting

```rust
pub fn full_piv_lu (& self) -> FullPivLu < C :: Canonical >;
```

#### Impl Method `qr`

> returns the $QR$ decomposition of `self`

```rust
pub fn qr (& self) -> Qr < C :: Canonical >;
```

#### Impl Method `col_piv_qr`

> returns the $QR$ decomposition of `self` with column pivoting

```rust
pub fn col_piv_qr (& self) -> ColPivQr < C :: Canonical >;
```

#### Impl Method `svd`

> returns the svd of `self`
>
> singular values are nonnegative and sorted in nonincreasing order

```rust
pub fn svd (& self) -> Result < Svd < C :: Canonical > , SvdError >;
```

#### Impl Method `thin_svd`

> returns the thin svd of `self`
>
> singular values are nonnegative and sorted in nonincreasing order

```rust
pub fn thin_svd (& self) -> Result < Svd < C :: Canonical > , SvdError >;
```

#### Impl Method `llt`

> returns the $L L^\top$ decomposition of `self`

```rust
pub fn llt (& self , side : Side) -> Result < Llt < C :: Canonical > , LltError >;
```

#### Impl Method `ldlt`

> returns the $L D L^\top$ decomposition of `self`

```rust
pub fn ldlt (& self , side : Side) -> Result < Ldlt < C :: Canonical > , LdltError >;
```

#### Impl Method `lblt`

> returns the $LBL^\top$ decomposition of `self`

```rust
pub fn lblt (& self , side : Side) -> Lblt < C :: Canonical >;
```

#### Impl Method `self_adjoint_eigen`

> returns the eigendecomposition of `self`, assuming it is self-adjoint
>
> eigenvalues sorted in nondecreasing order

```rust
pub fn self_adjoint_eigen (& self , side : Side) -> Result < SelfAdjointEigen < C :: Canonical > , EvdError >;
```

#### Impl Method `self_adjoint_eigenvalues`

> returns the eigenvalues of `self`, assuming it is self-adjoint
>
> eigenvalues sorted in nondecreasing order

```rust
pub fn self_adjoint_eigenvalues (& self , side : Side) -> Result < Vec < Real < C > > , EvdError >;
```

#### Impl Method `singular_values`

> returns the singular values of `self`
>
> singular values are nonnegative and sorted in nonincreasing order

```rust
pub fn singular_values (& self) -> Result < Vec < Real < C > > , SvdError >;
```

### Inherent Impl Block `MatRef<'_,C>`

```rust
impl C : Conjugate MatRef < '_ , C >
```

#### Impl Method `eigen_imp`

```rust
fn eigen_imp (& self) -> Result < Eigen < Real < C > > , EvdError >;
```

#### Impl Method `eigenvalues_imp`

```rust
fn eigenvalues_imp (& self) -> Result < Vec < Complex < Real < C > > > , EvdError >;
```

### Inherent Impl Block `mat::generic::Mat<Inner>`

```rust
impl T : Conjugate , Inner : for < 'short > Reborrow < 'short , Target = mat :: Ref < 'short , T > > mat :: generic :: Mat < Inner >
```

#### Impl Method `eigen`

> returns the eigendecomposition of `self`

```rust
pub fn eigen (& self) -> Result < Eigen < Real < T > > , EvdError >;
```

#### Impl Method `eigenvalues`

> returns the eigenvalues of `self`

```rust
pub fn eigenvalues (& self) -> Result < Vec < Complex < Real < T > > > , EvdError >;
```

### Trait `SolveLstsq`

> [`SolveLstsqCore`] extension trait

```rust
pub trait SolveLstsqT : ComplexField
```

### Trait `DenseSolve`

> [`DenseSolveCore`] extension trait

```rust
pub trait DenseSolveT : ComplexField
```

### Trait Impl Block `impl T : ComplexField , S : ? Sized + SolveCore < T > Solve < T > for S`

```rust
impl T : ComplexField , S : ? Sized + SolveCore < T > Solve < T > for S
```

### Trait Impl Block `impl T : ComplexField , S : ? Sized + SolveLstsqCore < T > SolveLstsq < T > for S`

```rust
impl T : ComplexField , S : ? Sized + SolveLstsqCore < T > SolveLstsq < T > for S
```

### Trait Impl Block `impl T : ComplexField , S : ? Sized + DenseSolveCore < T > DenseSolve < T > for S`

```rust
impl T : ComplexField , S : ? Sized + DenseSolveCore < T > DenseSolve < T > for S
```

### Struct `Llt`

> $L L^\top$ decomposition

```rust
pub struct Llt< T >
```

### Struct `Ldlt`

> $L D L^\top$ decomposition

```rust
pub struct Ldlt< T >
```

### Struct `Lblt`

> $LBL^\top$ decomposition

```rust
pub struct Lblt< T >
```

### Struct `PartialPivLu`

> $LU$ decomposition with partial (row) pivoting

```rust
pub struct PartialPivLu< T >
```

### Struct `FullPivLu`

> $LU$ decomposition with full pivoting

```rust
pub struct FullPivLu< T >
```

### Struct `Qr`

> $QR$ decomposition

```rust
pub struct Qr< T >
```

### Struct `ColPivQr`

> $QR$ decomposition with column pivoting

```rust
pub struct ColPivQr< T >
```

### Struct `Svd`

> svd decomposition (either full or thin)

```rust
pub struct Svd< T >
```

### Struct `SelfAdjointEigen`

> self-adjoint eigendecomposition

```rust
pub struct SelfAdjointEigen< T >
```

### Struct `Eigen`

> eigendecomposition

```rust
pub struct Eigen< T >
```

### Inherent Impl Block `Llt<T>`

```rust
impl T : ComplexField Llt < T >
```

#### Impl Method `new`

> returns the $L L^\top$ decomposition of $A$

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C > , side : Side) -> Result < Self , LltError >;
```

#### Impl Method `new_imp`

```rust
fn new_imp (mut L : Mat < T >) -> Result < Self , LltError >;
```

#### Impl Method `L`

> returns the $L$ factor

```rust
pub fn L (& self) -> MatRef < '_ , T >;
```

### Inherent Impl Block `Ldlt<T>`

```rust
impl T : ComplexField Ldlt < T >
```

#### Impl Method `new`

> returns the $L D L^\top$ decomposition of $A$

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C > , side : Side) -> Result < Self , LdltError >;
```

#### Impl Method `new_imp`

```rust
fn new_imp (mut L : Mat < T >) -> Result < Self , LdltError >;
```

#### Impl Method `L`

> returns the $L$ factor

```rust
pub fn L (& self) -> MatRef < '_ , T >;
```

#### Impl Method `D`

> returns the $D$ factor

```rust
pub fn D (& self) -> DiagRef < '_ , T >;
```

### Inherent Impl Block `Lblt<T>`

```rust
impl T : ComplexField Lblt < T >
```

#### Impl Method `new`

> returns the $LBL^\top$ decomposition of $A$

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C > , side : Side) -> Self;
```

#### Impl Method `new_imp`

```rust
fn new_imp (mut L : Mat < T >) -> Self;
```

#### Impl Method `L`

> returns the $L$ factor

```rust
pub fn L (& self) -> MatRef < '_ , T >;
```

#### Impl Method `B_diag`

> returns the diagonal of the $B$ factor

```rust
pub fn B_diag (& self) -> DiagRef < '_ , T >;
```

#### Impl Method `B_subdiag`

> returns the subdiagonal of the $B$ factor

```rust
pub fn B_subdiag (& self) -> DiagRef < '_ , T >;
```

#### Impl Method `P`

> returns the pivoting permutation $P$

```rust
pub fn P (& self) -> PermRef < '_ , usize >;
```

### Function `split_LU`

```rust
fn split_LU < T : ComplexField > (LU : Mat < T >) -> (Mat < T > , Mat < T >)
```

### Inherent Impl Block `PartialPivLu<T>`

```rust
impl T : ComplexField PartialPivLu < T >
```

#### Impl Method `new`

> returns the $LU$ decomposition of $A$ with partial pivoting

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C >) -> Self;
```

#### Impl Method `new_imp`

```rust
fn new_imp (mut LU : Mat < T >) -> Self;
```

#### Impl Method `L`

> returns the $L$ factor

```rust
pub fn L (& self) -> MatRef < '_ , T >;
```

#### Impl Method `U`

> returns the $U$ factor

```rust
pub fn U (& self) -> MatRef < '_ , T >;
```

#### Impl Method `P`

> returns the row pivoting permutation $P$

```rust
pub fn P (& self) -> PermRef < '_ , usize >;
```

### Inherent Impl Block `FullPivLu<T>`

```rust
impl T : ComplexField FullPivLu < T >
```

#### Impl Method `new`

> returns the $LU$ decomposition of $A$ with full pivoting

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C >) -> Self;
```

#### Impl Method `new_imp`

```rust
fn new_imp (mut LU : Mat < T >) -> Self;
```

#### Impl Method `L`

> returns the factor $L$

```rust
pub fn L (& self) -> MatRef < '_ , T >;
```

#### Impl Method `U`

> returns the factor $U$

```rust
pub fn U (& self) -> MatRef < '_ , T >;
```

#### Impl Method `P`

> returns the row pivoting permutation $P$

```rust
pub fn P (& self) -> PermRef < '_ , usize >;
```

#### Impl Method `Q`

> returns the column pivoting permutation $P$

```rust
pub fn Q (& self) -> PermRef < '_ , usize >;
```

### Inherent Impl Block `Qr<T>`

```rust
impl T : ComplexField Qr < T >
```

#### Impl Method `new`

> returns the $QR$ decomposition of $A$

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C >) -> Self;
```

#### Impl Method `new_imp`

```rust
fn new_imp (mut QR : Mat < T >) -> Self;
```

#### Impl Method `Q_basis`

> returns the householder basis of $Q$

```rust
pub fn Q_basis (& self) -> MatRef < '_ , T >;
```

#### Impl Method `Q_coeff`

> returns the householder coefficients of $Q$

```rust
pub fn Q_coeff (& self) -> MatRef < '_ , T >;
```

#### Impl Method `R`

> returns the factor $R$

```rust
pub fn R (& self) -> MatRef < '_ , T >;
```

#### Impl Method `thin_R`

> returns the upper trapezoidal part of $R$

```rust
pub fn thin_R (& self) -> MatRef < '_ , T >;
```

#### Impl Method `compute_Q`

> computes the factor $Q$

```rust
pub fn compute_Q (& self) -> Mat < T >;
```

#### Impl Method `compute_thin_Q`

> computes the first $\min(\text{nrows}, \text{ncols})$ columns of the factor $Q$

```rust
pub fn compute_thin_Q (& self) -> Mat < T >;
```

### Inherent Impl Block `ColPivQr<T>`

```rust
impl T : ComplexField ColPivQr < T >
```

#### Impl Method `new`

> returns the $QR$ decomposition of $A$ with column pivoting

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C >) -> Self;
```

#### Impl Method `new_imp`

```rust
fn new_imp (mut QR : Mat < T >) -> Self;
```

#### Impl Method `Q_basis`

> returns the householder basis of $Q$

```rust
pub fn Q_basis (& self) -> MatRef < '_ , T >;
```

#### Impl Method `Q_coeff`

> returns the householder coefficients of $Q$

```rust
pub fn Q_coeff (& self) -> MatRef < '_ , T >;
```

#### Impl Method `R`

> returns the factor $R$

```rust
pub fn R (& self) -> MatRef < '_ , T >;
```

#### Impl Method `thin_R`

> returns the upper trapezoidal part of $R$

```rust
pub fn thin_R (& self) -> MatRef < '_ , T >;
```

#### Impl Method `compute_Q`

> computes the factor $Q$

```rust
pub fn compute_Q (& self) -> Mat < T >;
```

#### Impl Method `compute_thin_Q`

> computes the first $\min(\text{nrows}, \text{ncols})$ columns of the factor $Q$

```rust
pub fn compute_thin_Q (& self) -> Mat < T >;
```

#### Impl Method `P`

> returns the column pivoting permutation $P$

```rust
pub fn P (& self) -> PermRef < '_ , usize >;
```

### Inherent Impl Block `Svd<T>`

```rust
impl T : ComplexField Svd < T >
```

#### Impl Method `new`

> returns the svd of $A$

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C >) -> Result < Self , SvdError >;
```

#### Impl Method `new_thin`

> returns the thin svd of $A$

```rust
pub fn new_thin < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C >) -> Result < Self , SvdError >;
```

#### Impl Method `new_imp`

```rust
fn new_imp (A : MatRef < '_ , T > , conj : Conj , thin : bool) -> Result < Self , SvdError >;
```

#### Impl Method `U`

> returns the factor $U$

```rust
pub fn U (& self) -> MatRef < '_ , T >;
```

#### Impl Method `V`

> returns the factor $V$

```rust
pub fn V (& self) -> MatRef < '_ , T >;
```

#### Impl Method `S`

> returns the factor $S$

```rust
pub fn S (& self) -> DiagRef < '_ , T >;
```

#### Impl Method `pseudoinverse`

> returns the pseudoinverse of the original matrix $A$.

```rust
pub fn pseudoinverse (& self) -> Mat < T >;
```

### Inherent Impl Block `SelfAdjointEigen<T>`

```rust
impl T : ComplexField SelfAdjointEigen < T >
```

#### Impl Method `new`

> returns the eigendecomposition of $A$, assuming it is self-adjoint

```rust
pub fn new < C : Conjugate < Canonical = T > > (A : MatRef < '_ , C > , side : Side) -> Result < Self , EvdError >;
```

#### Impl Method `new_imp`

```rust
fn new_imp (A : MatRef < '_ , T > , conj : Conj) -> Result < Self , EvdError >;
```

#### Impl Method `U`

> returns the factor $U$

```rust
pub fn U (& self) -> MatRef < '_ , T >;
```

#### Impl Method `S`

> returns the factor $S$

```rust
pub fn S (& self) -> DiagRef < '_ , T >;
```

### Inherent Impl Block `Eigen<T>`

```rust
impl T : RealField Eigen < T >
```

#### Impl Method `new`

> returns the eigendecomposition of $A$

```rust
pub fn new < C : Conjugate < Canonical = Complex < T > > > (A : MatRef < '_ , C >) -> Result < Self , EvdError >;
```

#### Impl Method `new_from_real`

> returns the eigendecomposition of $A$

```rust
pub fn new_from_real (A : MatRef < '_ , T >) -> Result < Self , EvdError >;
```

#### Impl Method `new_imp`

```rust
fn new_imp (A : MatRef < '_ , Complex < T > > , conj : Conj) -> Result < Self , EvdError >;
```

#### Impl Method `U`

> returns the factor $U$

```rust
pub fn U (& self) -> MatRef < '_ , Complex < T > >;
```

#### Impl Method `S`

> returns the factor $S$

```rust
pub fn S (& self) -> DiagRef < '_ , Complex < T > >;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for Llt < T >`

```rust
impl T : ComplexField ShapeCore for Llt < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for Ldlt < T >`

```rust
impl T : ComplexField ShapeCore for Ldlt < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for Lblt < T >`

```rust
impl T : ComplexField ShapeCore for Lblt < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for PartialPivLu < T >`

```rust
impl T : ComplexField ShapeCore for PartialPivLu < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for FullPivLu < T >`

```rust
impl T : ComplexField ShapeCore for FullPivLu < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for Qr < T >`

```rust
impl T : ComplexField ShapeCore for Qr < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for ColPivQr < T >`

```rust
impl T : ComplexField ShapeCore for ColPivQr < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for Svd < T >`

```rust
impl T : ComplexField ShapeCore for Svd < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField ShapeCore for SelfAdjointEigen < T >`

```rust
impl T : ComplexField ShapeCore for SelfAdjointEigen < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : RealField ShapeCore for Eigen < T >`

```rust
impl T : RealField ShapeCore for Eigen < T >
```

#### Impl Method `nrows`

```rust
fn nrows (& self) -> usize;
```

#### Impl Method `ncols`

```rust
fn ncols (& self) -> usize;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for Llt < T >`

```rust
impl T : ComplexField SolveCore < T > for Llt < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Function `make_self_adjoint`

```rust
fn make_self_adjoint < T : ComplexField > (mut A : MatMut < '_ , T >)
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for Llt < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for Llt < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for Ldlt < T >`

```rust
impl T : ComplexField SolveCore < T > for Ldlt < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for Ldlt < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for Ldlt < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for Lblt < T >`

```rust
impl T : ComplexField SolveCore < T > for Lblt < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for Lblt < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for Lblt < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for PartialPivLu < T >`

```rust
impl T : ComplexField SolveCore < T > for PartialPivLu < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for PartialPivLu < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for PartialPivLu < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for FullPivLu < T >`

```rust
impl T : ComplexField SolveCore < T > for FullPivLu < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for FullPivLu < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for FullPivLu < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for Qr < T >`

```rust
impl T : ComplexField SolveCore < T > for Qr < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField SolveLstsqCore < T > for Qr < T >`

```rust
impl T : ComplexField SolveLstsqCore < T > for Qr < T >
```

#### Impl Method `solve_lstsq_in_place_with_conj`

```rust
fn solve_lstsq_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for Qr < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for Qr < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for ColPivQr < T >`

```rust
impl T : ComplexField SolveCore < T > for ColPivQr < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField SolveLstsqCore < T > for ColPivQr < T >`

```rust
impl T : ComplexField SolveLstsqCore < T > for ColPivQr < T >
```

#### Impl Method `solve_lstsq_in_place_with_conj`

```rust
fn solve_lstsq_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for ColPivQr < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for ColPivQr < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for Svd < T >`

```rust
impl T : ComplexField SolveCore < T > for Svd < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField SolveLstsqCore < T > for Svd < T >`

```rust
impl T : ComplexField SolveLstsqCore < T > for Svd < T >
```

#### Impl Method `solve_lstsq_in_place_with_conj`

```rust
fn solve_lstsq_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for Svd < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for Svd < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Trait Impl Block `impl T : ComplexField SolveCore < T > for SelfAdjointEigen < T >`

```rust
impl T : ComplexField SolveCore < T > for SelfAdjointEigen < T >
```

#### Impl Method `solve_in_place_with_conj`

```rust
fn solve_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

#### Impl Method `solve_transpose_in_place_with_conj`

```rust
fn solve_transpose_in_place_with_conj (& self , conj : Conj , rhs : MatMut < '_ , T >);
```

### Trait Impl Block `impl T : ComplexField DenseSolveCore < T > for SelfAdjointEigen < T >`

```rust
impl T : ComplexField DenseSolveCore < T > for SelfAdjointEigen < T >
```

#### Impl Method `reconstruct`

```rust
fn reconstruct (& self) -> Mat < T >;
```

#### Impl Method `inverse`

```rust
fn inverse (& self) -> Mat < T >;
```

### Module `tests`

```rust
mod tests { /* ... */ }
```

