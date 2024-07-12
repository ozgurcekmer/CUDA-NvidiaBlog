# How to Overlap Data Transfers in CUDA/C++
## Problem
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

## Solution & Optimisation
### 1. Lack of kernel overlap
- The reason that the kernels don't overlap is the fact that each stream already saturates the GPU in terms of computation. Problem size is big, and our streams already use enough threads and then the kernels don't overlap. 
- In addition, since a kernel execution by a single stream already saturates the GPU computation, using multistreams wouldn't give us a speedup even if they overlap. 
- Let's check this and try to see a kernel overlap. 
- As the first job, let's check the properties of the device that we are using by building and running ***deviceQuery*** on **Hagi**. Here is the results:

<img src="images/DeviceQuery.png" alt="Device Query" width="600"/>

- The items we need to focus for now are the followings:
```
1. Concurrent copy and kernel execution         : Yes with 6 copy engines
2. Maximum number of threads per multiprocessor : 1024
```

- From Item 1, we can conclude that the problem is not the hardware, since concurrency is supported, and there are 6 copy engines available.
- Let's start to modify our GPU solvers using [flexible kernels with grid-stride loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) in the light of Item 2.

### 2. Lack of data transfer overlap
- On the other hand, the lack of overlapping in data transfer is a problem.

- Try to resolve the problem with the result for the current hardware.
- Build CUDA Samples and run deviceQuery.
- Try another system (Setonix and one more NVIDIA card)