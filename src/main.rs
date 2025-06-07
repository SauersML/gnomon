// ========================================================================================
//
//                              THE STRATEGIC ORCHESTRATOR
//
// ========================================================================================
//
// This module is the central nervous system and active conductor of the application.
// Its primary responsibility is to configure the hardware-aware resource limits and
// then orchestrate a high-performance, multi-threaded pipeline, ensuring that the I/O
// and computational components of the system are operating simultaneously.
//
// 1.  **Resource-Aware Configuration:** The `main` function's first critical task is to
//     determine the operational parameters. It inspects the environment (e.g., available
//     memory) and the workload (e.g., number of target individuals) to dynamically
//     calculate robust chunk sizes. This prevents out-of-memory failures and ensures
//     the application is tailored to the machine it is running on. It then creates the
//     `PgsContext`, which holds all *metadata* for the run—importantly, it does **not**
//     load large data matrices.
//
// 2.  **Pipeline Construction and Conduction:** The philosophy of separating planning
//     from execution is evolved into a concurrent, producer-consumer model. This module
//     is the master conductor:
//         a. It allocates two large, shared **pivot buffers** that will be passed
//            between threads.
//         b. It spawns a dedicated **I/O/Pivot Thread**, whose logic is defined in
//            `batch.rs`. This thread's sole purpose is to read genotype data from
//            disk and pivot it into the shared buffers.
//         c. The **Main Thread** itself becomes the **Compute Consumer**. It enters a
//            loop, waiting to receive a buffer filled with pivoted genotype data from
//            the I/O thread.
//
// 3.  **Engine Dispatch:** Once the main thread receives a data-ready buffer, it
//     dispatches it to the `batch.rs` compute engine, which in turn unleashes the
//     `rayon` thread pool on the data. When computation is complete, the main thread
//     returns the now-free buffer to the I/O thread to be filled with the next chunk
//     of data.
//
// This architecture ensures that while the CPU cores are busy calculating scores on
// one chunk of data, the I/O subsystem is already reading and preparing the next,
// eliminating idle time and maximizing hardware utilization.
