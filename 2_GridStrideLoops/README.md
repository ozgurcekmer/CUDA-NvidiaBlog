# CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops
## An example case: SAXPY
### CPU code
```
void saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; ++i)
    {
        y[i] = a * x[i] + y[i];
    }
}
```

### GPU common code
```
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}
```
- ***Monolithic kernel:*** 
    - It assumes a single grid of threads to process the entire array in 1 pass 
    - If there is 1<<20 elements: 
        ```
        saxpy <<<4096, 256>>> (1<<20, 2.0, x, y);
        ```
### GPU code with a ***grid-stride loop***
```
__global__
void saxpy(int n, float a, float *x, float *y)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; 
            i < n;
            i += blockDim.x * gridDim.x)
    {
        y[i] = a * x[i] + y[i];
    }
}
```
- ***blockDim.x * gridDim.x:*** total number of threads in the grid
- If there are 1280 threads in the grid, ***thread 0*** will compute elements 0, 1280, 2560, ...
- By using a loop with a stride equal to the grid size, we ensure that all addressing within warps is unit-stride, so we get ***maximum memory coalescing*** just as in the monolithic version.
- When launched with a grid large enough to cover all iterations of the loop, the instruction cost will be the same as the if statement in the monolithic kernel:
    - The loop increment will only be evaluated when the loop condition is true.

### Benefits to use a grid-stride loop
#### 1. Scalability & thread reuse
- Applicable to any problem size
- Can limit the number of thread blocks to tune performance
- It's often useful to launch a number of blocks that is a multiple of the number of multiprocessors on the device
    - (to balance utilisation)
    ```
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    saxpy<<< 32 * numSMs, 256 >>> (1<<20, 2.0, x, y);
    ```
- Limiting the number of blocks causes threads to get reused
    - Thread reuse amortises thread creation & destruction cost along with any other processing the kernel might do before or after the loop (such as thread-private or shared data initialisation)

#### 2. Debugging
- Easily can be switched to serial processing
```
saxpy <<< 1, 1 >>> (1<<20, 2.0, x, y);
```

#### 3. Portability & Readability





