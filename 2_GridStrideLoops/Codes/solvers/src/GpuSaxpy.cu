#include "../include/GpuSaxpy.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void saxpyKernel(T* x, T* y, T A, const int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride)
    {
        y[i] += A * x[i];
    }

}

template<typename T>
void GpuSaxpy<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dX, SIZE);
    gpuMalloc(&dY, SIZE);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSaxpy<T>::copyH2D()
{
    gpuMemcpy(dX, this->x.data(), SIZE, gpuMemcpyHostToDevice);
    gpuMemcpy(dY, this->y.data(), SIZE, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSaxpy<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, SIZE, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSaxpy<T>::~GpuSaxpy()
{
    // Deallocate device vectors
    gpuFree(dX);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSaxpy<T>::saxpy()
{
    deviceAllocations();
    copyH2D();
 
    cout << "Block size: " << BLOCK_SIZE << endl;
    cout << "Grid size : " << GRID_SIZE << endl;

    // gpuFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), gpuFuncCachePreferL1);

    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);
    
    const T A = 2.0;

    saxpyKernel << < GRID_SIZE, BLOCK_SIZE >> > (dX, dY, A, N);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
}

template void GpuSaxpy<float>::saxpy();
template void GpuSaxpy<double>::saxpy();
template void GpuSaxpy<float>::deviceAllocations();
template void GpuSaxpy<double>::deviceAllocations();
template void GpuSaxpy<float>::copyH2D();
template void GpuSaxpy<double>::copyH2D();
template GpuSaxpy<float>::~GpuSaxpy();
template GpuSaxpy<double>::~GpuSaxpy();
