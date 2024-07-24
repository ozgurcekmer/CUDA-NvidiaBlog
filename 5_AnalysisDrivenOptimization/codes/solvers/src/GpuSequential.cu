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
    for (auto k = 0; k < N; ++k)
    {
        T v1 = 0;

        // Vector average
        for (auto i = 0; i < M; ++i)
        {
            v1 += v[k * M * L + idx * M + i];
        }
        v1 /= static_cast<T>(M);

        // Matrix - Vector multiplication
        for (auto i = 0; i < L; ++i)
        {
            __syncthreads();
            sMem[threadIdx.x] = v1 * A[i * L + idx];
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
                y[i * N + k] = sMem[0];
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
