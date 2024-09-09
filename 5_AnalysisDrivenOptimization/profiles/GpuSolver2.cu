#include "../include/GpuSolver2.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSolver2(T* __restrict__ v, T* __restrict__ A, T* __restrict__ y)
{
    __shared__ T S[L];
    size_t tID = threadIdx.x; // thread ID
    size_t bID = blockIdx.x;  // block ID

    T vAvg = 0;

    // Vector average
    for (auto iM = 0; iM < M; ++iM)
    {
        vAvg += v[bID * M * L + tID * M + iM];
    }
    vAvg /= M;

    // Matrix - Vector multiplication
    for (auto iL = 0; iL < L; ++iL)
    {
        S[tID] = A[iL * L + tID] * vAvg;

        for (auto s = blockDim.x / 2; s > 0; s /= 2)
        {
            __syncthreads();
            if (tID < s)
            {
                S[tID] += S[tID + s];
            }
        }
        if (tID == 0)
        {
            y[iL * N + bID] = S[0];
        }
    }
}

template<typename T>
void GpuSolver2<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dV, BYTES_V);
    gpuMalloc(&dA, BYTES_A);
    gpuMalloc(&dY, BYTES_Y);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSolver2<T>::copyH2D()
{
    gpuMemcpy(dV, this->v.data(), BYTES_V, gpuMemcpyHostToDevice);
    gpuMemcpy(dA, this->A.data(), BYTES_A, gpuMemcpyHostToDevice);
    //gpuMemcpy(dY, this->y.data(), BYTES_Y, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSolver2<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, BYTES_Y, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSolver2<T>::~GpuSolver2()
{
    gpuFree(dV);
    gpuFree(dA);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSolver2<T>::solver()
{
    deviceAllocations();
    copyH2D();
    size_t gridSize = N;
#ifdef KERNELTIME
    auto t0 = omp_get_wtime();
    gpuSolver2<T> << < gridSize, BLOCK_SIZE >> > (dV, dA, dY);
    gpuCheckErrors("gpu kernel launch failure"); 
    gpuDeviceSynchronize();
    auto t1 = omp_get_wtime();
    cout << "Kernel runtime: " << (t1 - t0) * 1000.0 << " ms." << endl;
#else
    gpuSolver2<T> << < gridSize, BLOCK_SIZE >> > (dV, dA, dY);
    gpuCheckErrors("gpu kernel launch failure");
#endif
    copyD2H();
}

template void GpuSolver2<float>::solver();
template void GpuSolver2<double>::solver();
template void GpuSolver2<float>::deviceAllocations();
template void GpuSolver2<double>::deviceAllocations();
template void GpuSolver2<float>::copyH2D();
template void GpuSolver2<double>::copyH2D();
template GpuSolver2<float>::~GpuSolver2();
template GpuSolver2<double>::~GpuSolver2();
