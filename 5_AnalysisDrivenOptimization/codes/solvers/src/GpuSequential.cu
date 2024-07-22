#include "../include/GpuSequential.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSequential(T* v, T* vAvg, T* A, T* y, const size_t L, const size_t M, const size_t N)
{
    size_t iN = blockIdx.x;
    size_t iL = threadIdx.x;

    // Vector average
    T temp = 0.0;
    for (auto j = 0; j < M; ++j)
    {
        temp += v[iN * M * L + j * L + iL];
    }
    vAvg[iL] = temp / static_cast<T>(M);

    __syncthreads;
    // Matrix - Vector multiplication
    temp = 0.0;
    for (auto j = 0; j < L; ++j)
    {
        temp += A[iL * L + j] * vAvg[j];
    }
    y[iN * L + iL] = temp;
}

template<typename T>
void GpuSequential<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dV, BYTES_V);
    gpuMalloc(&dA, BYTES_A);
    gpuMalloc(&dY, BYTES_Y);
    gpuMalloc(&dVavg, BYTES_Vavg);
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
    gpuFree(dVavg);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSequential<T>::solver()
{
    deviceAllocations();
    copyH2D();
    gpuSequential<T> << < GRID_SIZE, BLOCK_SIZE >> > (dV, dVavg, dA, dY, L, M, N);
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
