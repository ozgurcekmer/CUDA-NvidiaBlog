# How to Overlap Data Transfers in CUDA/C++
## Accomplished
- The original NVIDIA blog is [here](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/).
- A personal laptop, ***Hagi***, with NVIDIA GeForce RTX 2070 with Max-Q Design is used for the simulations (with Windows & Microsoft Visual Studio solution).
- Codes are developed in object oriented fashion.
- Pinned vectors are created by modifying ***std::vector***s.
- The problem size was set to ~4M (1 << 22) at the beginning.
- 4 CUDA streams are used
- The following flag is used while compiling:
```
--default-stream per-thread
```
- Nsight-systems is used for profiling.
- The following timeline was obtained for the sequential GPU solver.

<img src="images/ProblemSize4M__Seq.png" alt="Sequential problem size 4M" width="600"/>

- Function call for ***gpuSequential*** takes **1.775 ms**, which is comparable to the function execution itself. A bigger problem size is selected to make the runtime for the function calls negligible and focus on the function and data transfer overlaps.
- Problem size is set to **~33.6M (1 << 25)**.
- Timeline for the sequential solver is below:

<img src="images/ProblemSize33M__Seq.png" alt="Sequential problem size 33M" width="600"/>

- The results for runtime are tabulated below:

| Solver | Total Runtime (ms) | 
| --- | ---: | 
| CPU* | 218.375 | 
| GPU Sequential | 70.615 |
| GPU Version 1 | 70.344 | 
| GPU Version 2 | 70.252 | 

* ****CPU*** is a CPU solver using OpenMP threads
- No overlap could be achieved in the first try.
- Here are the timelines for GPU solver versions 1 and 2, respectively:

<img src="images/Version1-Initial.png" alt="Hagi_InitialVersion_1" width="600"/>

<img src="images/Version2-Initial.png" alt="Hagi_InitialVersion_2" width="600"/>

## To do
- Try to resolve the problem with the result for the current hardware.
- Build CUDA Samples and run deviceQuery.
- Try another system (Setonix and one more NVIDIA card)