# CUDA Pro Tip: Increase Performance with Vectorised Memory Access
- How to use vector loads & stores in CUDA to increase bandwidth utilisation while decreasing the number of executed instructions
## Example: A Simple Memory Copy Kernel
```
__global__
void device_copy_scalar_kernel(int* d_in, int* d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        d_out[i] = d_in[i];
    }
}

void device_copy_scalar(int* d_in, int* d_out, int N)
{
    int threads = 128;
    int blocks = min((N + threads - 1) / threads, MAX_BLOCKS);
    device_copy_scalar_kernel <<< blocks, threads >>> (d_in, d_out, N);
}
```
- Use ***cuobjdump*** tool: 
```
cuobjdump -sass executable
```
- The **SASS** for the body of the scalar copy kernel:
```
/*0058*/ IMAD R6.CC, R0, R9, c[0x0][0x140]
/*0060*/ IMAD.HI.X R7, R0, R9, c[0x0][0x144]
/*0068*/ IMAD R4.CC, R0, R9, c[0x0][0x148]
/*0070*/ LD.E R2, [R6]
/*0078*/ IMAD.HI.X R5, R0, R9, c[0x0][0x14c]
/*0090*/ ST.E [R4], R2
```
- A total of 6 instructions associated with the copy operation
    - 4 IMAD instructions compute the load & store addresses
    - LD.E & ST.E load & store 32 bits from those addresses

- Performance of this operation can be improved by using the vectorised load & store instructions
    - LD.E.{64, 128}
    - ST.E.{64, 128}
    - The above two load & store data in 64- or 128-bit widths
- Using vectorised loads 
    - reduces the total number of instructions