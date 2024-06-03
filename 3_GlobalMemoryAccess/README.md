# How to Access Global Memory Efficiently in CUDA C/C++ Kernels

- Efficiently access device memory
    - ***global memory*** in particular, from within kernels
- There are several kinds of memory on a CUDA device, each with different:
    - scope
    - lifetime
    - caching behaviour
- **Global** refers to the scope
- ***Global memory*** resides in device DRAM
    - ***Global memory*** can be accessed and modified from both the host and the device
    - data transfer between the host and device
    - data input to & output from kernels
- ***Global memory***:
    - can be declared in global scope using **\_\_device\_\_**
        - ```
            __device__ int globalArray[256];
          ```
    - or, dynamically allocated using **cudaMalloc()** and assigned to a regular C pointer
        - ```
            int *myDevMem = 0;
            result = cudaMalloc(&myDevMem, 256*sizeof(int));
            ```
- ***Global memory*** allocations can persist for the lifetime of the application
- Depending on the ***compute capability*** of the device, ***global memory*** may or may not be cached on the chip
- ***Threads*** are grouped into ***thread blocks***, which are assigned to ***multiprocessors*** on the device
- During execution, there is a finer grouping of ***threads*** into ***warps***
- ***Multiprocessors*** on the GPU execute instructions for each warp in SIMD fashion
- The ***warp size*** (effectively the ***SIMD width***) of all CUDA capable GPUs is ***32 threads***

## Global Memory Coalescing
- Grouping of threads into warps is not only relevant to computation, but also to ***global memory access***
- The device ***coalesces*** global memory loads & stores issued by threads of a warp into as few transactions as possible to minimise ***DRAM bandwidth***
- Conditions under which coalescing occurs across CUDA device architectures
    - misaligned accesses to the input array (offset in the example below)
    - strided accesses to the input array

```
template <class T>
__global__
void offset (T* a, int s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + s;
    a[i] += 1;
}

template <class T>
__global__
void stride (T* a, int s)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * s;
    a[i] += 1;
}
```

### Misaligned Access
- Arrays allocated in device memory are aligned to ***256-byte memory segments*** by the CUDA driver
- The device can access global memory via 32-, 64-, or 128-byte transactions that are aligned to their size
- Devices of compute capability 2.0 (e.g. Tesla C2050) have an L1 cache in each multiprocessor with a 128-byte line size
    - The device coalesces accesses by threads in a warp into as few cache lines as possible, resulting in negligible effect of alignment on throughput for sequential memory accesses across threads 

### Strided Memory Access
- For large strides, the effective bandwidth is poor regardless of the architecture
    - When concurrent threads simultaneously access memory addresses that are very far apart physical memory, then there is no chance for the hardware to combine the accesses
- Use ***shared memory*** when applicable
    - Extract a 2D tile of a multidimensional array from global memory in a coalesced fashion into shared memory, and then have contiguous threads stride through the shared memory tile




