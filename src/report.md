# GetDoc Report - Wed, 25 Jun 2025 22:41:00 -0500

## Compiler Output (Errors and Warnings)

### Diagnostics for: default features

```text
ERROR: error: character constant must be escaped: `\t`
   --> src/prepare.rs:511:29
    |
511 |                     .split('    ')
    |                             ^^^^
    |
help: escape the character
    |
511 |                     .split('\t')
    |                             ++
    (Diagnostic primary location: src/prepare.rs:511)
ERROR: error: character constant must be escaped: `\t`
   --> src/prepare.rs:537:45
    |
537 |             let mut parts = line.splitn(4, '    ');
    |                                             ^^^^
    |
help: escape the character
    |
537 |             let mut parts = line.splitn(4, '\t');
    |                                             ++
    (Diagnostic primary location: src/prepare.rs:537)
ERROR: error: character constant must be escaped: `\t`
   --> src/prepare.rs:549:25
    |
549 |                 .split('    ')
    |                         ^^^^
    |
help: escape the character
    |
549 |                 .split('\t')
    |                         ++
    (Diagnostic primary location: src/prepare.rs:549)
ERROR: error: character constant must be escaped: `\t`
  --> src/reformat.rs:99:46
   |
99 |     let cols: Vec<&str> = header_line.split('    ').collect();
   |                                              ^^^^
   |
help: escape the character
   |
99 |     let cols: Vec<&str> = header_line.split('\t').collect();
   |                                              ++
    (Diagnostic primary location: src/reformat.rs:99)
ERROR: error: character constant must be escaped: `\t`
   --> src/reformat.rs:143:45
    |
143 |             let key = parse_key(line.split('    ').next().unwrap_or(""))?;
    |                                             ^^^^
    |
help: escape the character
    |
143 |             let key = parse_key(line.split('\t').next().unwrap_or(""))?;
    |                                             ++
    (Diagnostic primary location: src/reformat.rs:143)
ERROR: error: character constant must be escaped: `\t`
   --> src/reformat.rs:244:44
    |
244 |         let fields: Vec<&str> = row.split('    ').collect();
    |                                            ^^^^
    |
help: escape the character
    |
244 |         let fields: Vec<&str> = row.split('\t').collect();
    |                                            ++
    (Diagnostic primary location: src/reformat.rs:244)
ERROR: E0412: error[E0412]: cannot find type `Reverse` in this scope
   --> src/prepare.rs:532:32
    |
532 |         heap: &mut BinaryHeap<(Reverse<VariantKey>, usize)>,
    |                                ^^^^^^^ not found in this scope
    |
help: consider importing this struct
    |
11  + use std::cmp::Reverse;
    |

  **Explanation (E0412)**:
  > A used type name is not in scope.
  > 
  > Erroneous code examples:
  > 
  > ```compile_fail,E0412
  > impl Something {} // error: type name `Something` is not in scope
  > 
  > // or:
  > 
  > trait Foo {
  >     fn bar(N); // error: type name `N` is not in scope
  > }
  > 
  > // or:
  > 
  > fn foo(x: T) {} // type name `T` is not in scope
  > ```
  > 
  > To fix this error, please verify you didn't misspell the type name, you did
  > declare it or imported it into the scope. Examples:
  > 
  > ```
  > struct Something;
  > 
  > impl Something {} // ok!
  > 
  > // or:
  > 
  > trait Foo {
  >     type N;
  > 
  >     fn bar(_: Self::N); // ok!
  > }
  > 
  > // or:
  > 
  > fn foo<T>(x: T) {} // ok!
  > ```
  > 
  > Another case that causes this error is when a type is imported into a parent
  > module. To fix this, you can follow the suggestion and use File directly or
  > `use super::File;` which will import the types from the parent namespace. An
  > example that causes this error is below:
  > 
  > ```compile_fail,E0412
  > use std::fs::File;
  > 
  > mod foo {
  >     fn some_function(f: File) {}
  > }
  > ```
  > 
  > ```
  > use std::fs::File;
  > 
  > mod foo {
  >     // either
  >     use super::File;
  >     // or
  >     // use std::fs::File;
  >     fn foo(f: File) {}
  > }
  > # fn main() {} // don't insert it for us; that'll break imports
  > ```
    (Diagnostic primary location: src/prepare.rs:532)
ERROR: E0425: error[E0425]: cannot find function, tuple struct or tuple variant `Reverse` in this scope
   --> src/prepare.rs:555:28
    |
555 |                 heap.push((Reverse(key), file_idx));
    |                            ^^^^^^^ not found in this scope
    |
help: consider importing this tuple struct
    |
11  + use std::cmp::Reverse;
    |

  **Explanation (E0425)**:
  > An unresolved name was used.
  > 
  > Erroneous code examples:
  > 
  > ```compile_fail,E0425
  > something_that_doesnt_exist::foo;
  > // error: unresolved name `something_that_doesnt_exist::foo`
  > 
  > // or:
  > 
  > trait Foo {
  >     fn bar() {
  >         Self; // error: unresolved name `Self`
  >     }
  > }
  > 
  > // or:
  > 
  > let x = unknown_variable;  // error: unresolved name `unknown_variable`
  > ```
  > 
  > Please verify that the name wasn't misspelled and ensure that the
  > identifier being referred to is valid for the given situation. Example:
  > 
  > ```
  > enum something_that_does_exist {
  >     Foo,
  > }
  > ```
  > 
  > Or:
  > 
  > ```
  > mod something_that_does_exist {
  >     pub static foo : i32 = 0i32;
  > }
  > 
  > something_that_does_exist::foo; // ok!
  > ```
  > 
  > Or:
  > 
  > ```
  > let unknown_variable = 12u32;
  > let x = unknown_variable; // ok!
  > ```
  > 
  > If the item is not defined in the current module, it must be imported using a
  > `use` statement, like so:
  > 
  > ```
  > # mod foo { pub fn bar() {} }
  > # fn main() {
  > use foo::bar;
  > bar();
  > # }
  > ```
  > 
  > If the item you are importing is not defined in some super-module of the
  > current module, then it must also be declared as public (e.g., `pub fn`).
    (Diagnostic primary location: src/prepare.rs:555)
ERROR: E0425: error[E0425]: cannot find function, tuple struct or tuple variant `Reverse` in this scope
   --> src/prepare.rs:583:29
    |
583 |             self.heap.push((Reverse(*current_key), file_idx));
    |                             ^^^^^^^ not found in this scope
    |
help: consider importing this tuple struct
    |
11  + use std::cmp::Reverse;
    |

  **Explanation (E0425)**:
  > An unresolved name was used.
  > 
  > Erroneous code examples:
  > 
  > ```compile_fail,E0425
  > something_that_doesnt_exist::foo;
  > // error: unresolved name `something_that_doesnt_exist::foo`
  > 
  > // or:
  > 
  > trait Foo {
  >     fn bar() {
  >         Self; // error: unresolved name `Self`
  >     }
  > }
  > 
  > // or:
  > 
  > let x = unknown_variable;  // error: unresolved name `unknown_variable`
  > ```
  > 
  > Please verify that the name wasn't misspelled and ensure that the
  > identifier being referred to is valid for the given situation. Example:
  > 
  > ```
  > enum something_that_does_exist {
  >     Foo,
  > }
  > ```
  > 
  > Or:
  > 
  > ```
  > mod something_that_does_exist {
  >     pub static foo : i32 = 0i32;
  > }
  > 
  > something_that_does_exist::foo; // ok!
  > ```
  > 
  > Or:
  > 
  > ```
  > let unknown_variable = 12u32;
  > let x = unknown_variable; // ok!
  > ```
  > 
  > If the item is not defined in the current module, it must be imported using a
  > `use` statement, like so:
  > 
  > ```
  > # mod foo { pub fn bar() {} }
  > # fn main() {
  > use foo::bar;
  > bar();
  > # }
  > ```
  > 
  > If the item you are importing is not defined in some super-module of the
  > current module, then it must also be declared as public (e.g., `pub fn`).
    (Diagnostic primary location: src/prepare.rs:583)
WARNING: unused_imports: warning: unused import: `std::iter::Peekable`
  --> src/prepare.rs:24:5
   |
24 | use std::iter::Peekable;
   |     ^^^^^^^^^^^^^^^^^^^
   |
   = note: `#[warn(unused_imports)]` on by default
    (Diagnostic primary location: src/prepare.rs:24)
WARNING: unused_imports: warning: unused import: `PathBuf`
  --> src/reformat.rs:13:23
   |
13 | use std::path::{Path, PathBuf};
   |                       ^^^^^^^
    (Diagnostic primary location: src/reformat.rs:13)
WARNING: unused_variables: warning: unused variable: `reconciled_idx`
   --> src/prepare.rs:295:10
    |
295 |     for (reconciled_idx, scores) in variant_to_scores_map.iter().enumerate() {
    |          ^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_reconciled_idx`
    |
    = note: `#[warn(unused_variables)]` on by default
    (Diagnostic primary location: src/prepare.rs:295)
```

No third-party crate information extracted (or no third-party files were implicated).
