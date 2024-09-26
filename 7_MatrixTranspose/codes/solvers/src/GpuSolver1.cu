#include "../include/GpuSolver1.h"

#ifdef KERNELTIME
#include <omp.h>
#endif

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSolver1(T* __restrict__ v, T* __restrict__ y)
{
    int ix = blockIdx.x * TILE_DIM + threadIdx.x;
    int iy = blockIdx.y * TILE_DIM + threadIdx.y;
    //int width = gridDim.x * TILE_DIM;
    int width = N;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        y[ix * width + (iy + j)] = v[(iy + j) * width + ix];
    }
    
}

template<typename T>
void GpuSolver1<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dV, BYTES);
    gpuMalloc(&dY, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSolver1<T>::copyH2D()
{
    gpuMemcpy(dV, this->v.data(), BYTES, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSolver1<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, BYTES, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSolver1<T>::~GpuSolver1()
{
    gpuFree(dV);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSolver1<T>::solver()
{
    
    deviceAllocations();
    
    copyH2D();
    dim3 threads(TILE_DIM, BLOCK_ROWS, 1);
    dim3 blocks(N / TILE_DIM, N / TILE_DIM, 1);
#ifdef KERNELTIME
    auto t0 = omp_get_wtime();
    gpuSolver1<T> << < blocks, threads >> > (dV, dY);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    auto t1 = omp_get_wtime();
    cout << "Kernel runtime: " << (t1 - t0) * 1000.0 << " ms." << endl;
#else
    gpuSolver1<T> << < blocks, threads >> > (dV, dY);
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
