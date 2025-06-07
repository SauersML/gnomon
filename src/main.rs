// ========================================================================================
//
//                      THE STRATEGIC ORCHESTRATOR
//
// ========================================================================================
//
// This module is the central nervous system and active conductor of the application.
// Its primary responsibility is to prepare all data and then orchestrate a
// high-performance, multi-threaded pipeline, ensuring that the I/O and computational
// components of the system are operating simultaneously.
//
// 1.  **Pre-computation Setup:** The `main` function's first critical task is to
//     prepare the data. It **pre-loads the entire weight matrix into memory** and
//     transforms it into a CPU-kernel-friendly interleaved layout. This establishes the
//     application's primary memory footprint up-front. It then creates the
//     `PgsContext`, which holds all metadata and a reference to this in-memory weight
//     matrix.
//
// 2.  **Pipeline Construction and Conduction:** The philosophy of separating planning
//     from execution is evolved into a concurrent, producer-consumer model
//     **communicating via message-passing channels** to ensure thread safety and avoid
//     locks.
//       a. It creates a **fixed pool of large, reusable pivot buffers** (e.g., two
//          buffers).
//       b. It constructs two **bounded channels**: one to send filled buffers to the
//          consumer, and one to return empty buffers to the producer. It pre-populates
//          the "free" channel with the buffers from the pool.
//       c. It spawns a dedicated **I/O/Pivot Thread**, whose logic is defined in
//          `batch.rs`.
//       d. The **Main Thread** itself becomes the **Compute Consumer**. It enters a loop,
//          **receiving** a filled buffer from the channel.
//
// 3.  **Engine Dispatch:** Once the main thread receives a data-ready buffer, it
//     dispatches it to the `batch.rs` compute engine. When computation is complete, the
//     main thread **sends** the now-empty buffer back to the I/O thread **via the 'free'
//     channel**.
//
// This architecture ensures that while the CPU cores are busy calculating scores on
// one chunk of data, the I/O subsystem is already reading and preparing the next,
// eliminating idle time and maximizing hardware utilization.
