# CUDA Pro Tip: Increase Performance with Vectorised Memory Access
## Results

### Nvidia - Geforce RTX 2070 with Max Q Design (352.06 GB/s max bandwidth)
| Solver | Kernel Runtime (ms) | Bandwidth (GB/s) |
| :--- | ---: | ---: |
| cudaMemcpy | 0.88 | 152.82 
| float | 1.34 | 199.76
| float2 | 2.34 | 114.68
| float4 | 1.20 | 222.89
* ***cudaMemcpyDeviceToDevice*** metrics were extracted using ***Nsight-Systems***, whereas the others were obtained by ***Nsight-Compute***. 
### AMD


