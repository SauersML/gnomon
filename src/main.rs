// ========================================================================================
//
//                         THE STRATEGIC ORCHESTRATOR
//
// ========================================================================================
//
// This module is the central nervous system and strategic planner of the application.
// Its primary responsibility is to perform all "thinking" and "planning" upfront,
// creating an immutable "world view" so that the subsequent parallel computation can
// proceed as a pure, "unthinking" execution of that plan.
//
// 1.  **Context Construction:** The `main` function's most critical task is to
//     orchestrate the creation of the `PgsContext`. It calls into `io.rs` to ingest
//     ALL metadata—every score file, every SNP in the model, every person in the
//     cohort—and synthesizes it into a single, comprehensive, read-only data
//     structure. This includes the creation of the master interleaved weight matrix.
//
// 2.  **Immutable Universe:** Once the `PgsContext` is built, it is treated as an
//     immutable constant for the remainder of the program's lifetime. It is passed
//     by reference (`&PgsContext`) to all other components. This is the key to safe,
//     scalable parallelism. By guaranteeing that the "world" does not change, we
//     eliminate the need for locks, mutexes, or any other form of costly
//     synchronization in the compute stages.
//
// 3.  **Engine Dispatch:** This module contains the top-level logic that inspects the
//     user's request and dispatches control to the appropriate engine. It is a simple,
//     high-level routing decision: based on the number of target individuals, it
//     invokes either the `batch::calculate_for_batch` function or the `single.rs`
//     logic (which is out of scope for this plan). It does not perform any
//     calculations itself.
//
// The philosophy here is a strict separation of planning from execution. This module
// builds the perfect blueprint; the other modules execute it flawlessly.
