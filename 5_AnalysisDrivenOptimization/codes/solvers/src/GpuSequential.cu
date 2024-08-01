#include "../include/GpuSequential.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSequential(T* __restrict__ v, T* __restrict__ A, T* __restrict__ y)
{
    __shared__ T sMem[L];

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Iterate over N data sets
    for (auto iN = 0; iN < N; ++iN)
    {
        T v1 = 0;

        // Vector average
        for (auto iM = 0; iM < M; ++iM)
        {
            v1 += v[iN * M * L + idx * M + iM];
        }
        v1 /= static_cast<T>(M);

        // Matrix - Vector multiplication
        for (auto iL = 0; iL < L; ++iL)
        {
            __syncthreads();
            sMem[threadIdx.x] = v1 * A[iL * L + idx];
            for (int s = blockDim.x / 2; s > 0; s /= 2)
            {
                __syncthreads();
                if (threadIdx.x < s)
                {
                    sMem[threadIdx.x] += sMem[threadIdx.x + s];
                }
            }
            if (threadIdx.x == 0)
            {
                y[iL * N + iN] = sMem[0];
            }
        }


    }
}

template<typename T>
void GpuSequential<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dV, BYTES_V);
    gpuMalloc(&dA, BYTES_A);
    gpuMalloc(&dY, BYTES_Y);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSequential<T>::copyH2D()
{
    gpuMemcpy(dV, this->v.data(), BYTES_V, gpuMemcpyHostToDevice);
    gpuMemcpy(dA, this->A.data(), BYTES_A, gpuMemcpyHostToDevice);
    gpuMemcpy(dY, this->y.data(), BYTES_Y, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSequential<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, BYTES_Y, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSequential<T>::~GpuSequential()
{
    gpuFree(dV);
    gpuFree(dA);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSequential<T>::solver()
{
    deviceAllocations();
    copyH2D();
    gpuSequential<T> << < GRID_SIZE, BLOCK_SIZE >> > (dV, dA, dY);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
}

template void GpuSequential<float>::solver();
template void GpuSequential<double>::solver();
template void GpuSequential<float>::deviceAllocations();
template void GpuSequential<double>::deviceAllocations();
template void GpuSequential<float>::copyH2D();
template void GpuSequential<double>::copyH2D();
template GpuSequential<float>::~GpuSequential();
template GpuSequential<double>::~GpuSequential();
