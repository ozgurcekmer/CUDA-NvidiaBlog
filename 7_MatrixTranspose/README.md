# An Efficient Matrix Transpose
- The original post by [Mark Harris](https://developer.nvidia.com/blog/author/mharris/) is [here](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/#:~:text=In%20this%20post%20I%20will%20show%20some%20of%20the%20performance).
- I am sharing my experience with code development and optimization here.
- A personal laptop, ***Hagi***, with ***NVIDIA GeForce RTX 2070 with Max-Q Design*** and ***INTEL CORE i7 10th GEN*** is used for the simulations (with Windows & Microsoft Visual Studio).
## Code for analysis
- The code performs a matrix transpose.
- Matrix to be transposed has a size of **4096 x 4096**. 
- Four solvers have been developed:
  - ***cpu***: A CPU solver with OpenMP threads
  - ***gpuSolver1***: A naive GPU solver
  - ***gpuSolver2***: A coalesced GPU solver with shared memory
  - ***gpuSolver3***: A coalesced GPU solver with shared memory - bank conflicts prevented
## Results
- The bandwidth results as computed by **nsight-compute** are as follows:

| Solver | Kernel runtime (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: |
| cpu (12 threads) | 110.6 | |
| gpuSolver1 | 2.36 | 59.3 |
| gpuSolver2 | 2.28 | 58.7 |
| gpuSolver3 | 2.33 | 61.2 |

- ***Add some comments here***