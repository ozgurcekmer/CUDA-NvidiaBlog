#include "../include/GpuSolver1.h"

#ifdef KERNELTIME
#include <omp.h>
#endif

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSolver1(T* __restrict__ v, T* __restrict__ A, T* __restrict__ y)
{
    size_t idx = threadIdx.x;
    __shared__ T S[L];

    for (auto iN = 0; iN < N; ++iN)
    {
        T vAvg = 0;

        // Vector average
        for (auto iM = 0; iM < M; ++iM)
        {
            vAvg += v[iN * M * L + idx * M + iM];
        }
        vAvg /= M;
        
        // Matrix - Vector multiplication
        for (auto iL = 0; iL < L; ++iL)
        {
            S[idx] = A[iL * L + idx] * vAvg;

            for (auto s = blockDim.x / 2; s > 0; s /= 2)
            {
                __syncthreads();
                if (idx < s)
                {
                    S[idx] += S[idx + s];
                }
            }
            if (idx == 0)
            {
                y[iL * N + iN] = S[0];
            }
        }
    }
}

template<typename T>
void GpuSolver1<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dV, BYTES_V);
    gpuMalloc(&dA, BYTES_A);
    gpuMalloc(&dY, BYTES_Y);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSolver1<T>::copyH2D()
{
    gpuMemcpy(dV, this->v.data(), BYTES_V, gpuMemcpyHostToDevice);
    gpuMemcpy(dA, this->A.data(), BYTES_A, gpuMemcpyHostToDevice);
    //gpuMemcpy(dY, this->y.data(), BYTES_Y, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSolver1<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, BYTES_Y, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSolver1<T>::~GpuSolver1()
{
    gpuFree(dV);
    gpuFree(dA);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSolver1<T>::solver()
{
    deviceAllocations();
    copyH2D();
#ifdef KERNELTIME
    auto t0 = omp_get_wtime();
    gpuSolver1<T> << < GRID_SIZE, BLOCK_SIZE >> > (dV, dA, dY);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    auto t1 = omp_get_wtime();
    cout << "Kernel runtime: " << (t1 - t0) * 1000.0 << " ms." << endl;
#else
    gpuSolver1<T> << < GRID_SIZE, BLOCK_SIZE >> > (dV, dA, dY);
    gpuCheckErrors("gpu kernel launch failure");
#endif
    copyD2H();
}

template void GpuSolver1<float>::solver();
template void GpuSolver1<double>::solver();
template void GpuSolver1<float>::deviceAllocations();
template void GpuSolver1<double>::deviceAllocations();
template void GpuSolver1<float>::copyH2D();
template void GpuSolver1<double>::copyH2D();
template GpuSolver1<float>::~GpuSolver1();
template GpuSolver1<double>::~GpuSolver1();
