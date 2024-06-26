#include "../include/GpuSequential.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSequential(T* a, int offset) 
{
    int idx = offset + blockDim.x * blockIdx.x + threadIdx.x;
    T iX = static_cast<T>(idx);
    T s = sin(iX);
    T c = cos(iX);
    a[idx] = sqrt(s * s + c * c);
}

template<typename T>
void GpuSequential<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, SIZE);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSequential<T>::copyH2D()
{
    gpuMemcpy(dA, this->a.data(), SIZE, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSequential<T>::copyD2H()
{
    gpuMemcpy(this->a.data(), dA, SIZE, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

/*
template<typename T>
void GpuSequential<T>::launchSetup()
{
    auto blocksPerSM = 2048 / BLOCK_SIZE;
    int devID;
    int numSMs;
    gpuGetDevice(&devID);

    gpuDeviceGetAttribute(&numSMs, gpuDevAttrMultiProcessorCount, devID);
    std::cout << "There are " << numSMs << " SMs in this device." << std::endl;
    std::cout << "Blocks per SM: " << blocksPerSM << std::endl;

    gridSize = blocksPerSM * numSMs;
    std::cout << "Grid Size: " << gridSize << std::endl;
    std::cout << "Block Size: " << BLOCK_SIZE << std::endl;
}
*/

template<typename T>
GpuSequential<T>::~GpuSequential()
{
    gpuFree(dA);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSequential<T>::solver()
{
    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);
    deviceAllocations();
    copyH2D();
    //launchSetup();
    int offset = 0;
    gpuSequential<T> << < GRID_SIZE, BLOCK_SIZE >> > (dA, offset);
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
