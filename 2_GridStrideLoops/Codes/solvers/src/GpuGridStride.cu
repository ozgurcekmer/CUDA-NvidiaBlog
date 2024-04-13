#include "../include/GpuGridStride.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gridStrideKernel(T* x, T* y, T A, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
    {
        y[i] += A * x[i];
    }

}

template<typename T>
void GpuGridStride<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dX, SIZE);
    gpuMalloc(&dY, SIZE);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuGridStride<T>::copyH2D()
{
    gpuMemcpy(dX, this->x.data(), SIZE, gpuMemcpyHostToDevice);
    gpuMemcpy(dY, this->y.data(), SIZE, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuGridStride<T>::copyD2H()
{
    gpuMemcpy(this->y.data(), dY, SIZE, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuGridStride<T>::~GpuGridStride()
{
    // Deallocate device vectors
    gpuFree(dX);
    gpuFree(dY);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuGridStride<T>::saxpy()
{
    gpuReportDevice();
    deviceAllocations();
    copyH2D();

    int blocksPerSM = 2048 / BLOCK_SIZE;

    int devID;
    int numSMs;
    gpuGetDevice(&devID);    

    blocksPerSM = 32;

    gpuDeviceGetAttribute(&numSMs, gpuDevAttrMultiProcessorCount, devID);
    cout << "There are " << numSMs << " SMs in this device." << endl;
    cout << "Blocks per SM: " << blocksPerSM << endl;

    const int gridSize = blocksPerSM * numSMs;

    cout << "Block size: " << BLOCK_SIZE << endl;
    cout << "Grid size : " << gridSize << endl;

    // gpuFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), gpuFuncCachePreferL1);

    const T A = 2.0;

    gridStrideKernel << < gridSize, BLOCK_SIZE >> > (dX, dY, A, N);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
}

template void GpuGridStride<float>::saxpy();
template void GpuGridStride<double>::saxpy();
template void GpuGridStride<float>::deviceAllocations();
template void GpuGridStride<double>::deviceAllocations();
template void GpuGridStride<float>::copyH2D();
template void GpuGridStride<double>::copyH2D();
template GpuGridStride<float>::~GpuGridStride();
template GpuGridStride<double>::~GpuGridStride();
