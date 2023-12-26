# An Even Easier Introduction to CUDA
- The related NVIDIA blog is [here](https://developer.nvidia.com/blog/even-easier-introduction-cuda/).
- An object-oriented approach has been implemented within the CUDA/C++ code.
- A personal laptop with NVIDIA GeForce RTX 2070 with Max-Q Design is used for the simulations.

## Runtimes:
- The performance results are as follows:

| Solver | Kernel Runtime (ms) | Bandwidth (GB/s) |
| --- | --- | --- |
| CPU | 164.75 | N/A 
| CUDA (1, 1) | 8341.16 | N/A
| CUDA (256, 1) | 111.94 | 2.42
| CUDA (256, 256) | 1.16 | 171.64
| CUDA (1152, 256)* | 1.06 | 188.56 
| CUDA (1152, 1024)* | 1.10 | 182.18
| CUDA (16384, 1024) | 1.72 | 117.40

* \* 1152 is the number of SMs in my GPU times 32

## Roofline analysis:

<img src="images/Roofline.png" alt="Roofline" width="600"/>