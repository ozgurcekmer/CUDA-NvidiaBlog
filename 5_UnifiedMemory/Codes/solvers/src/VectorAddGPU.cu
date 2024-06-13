#include "../include/VectorAddGPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuVectorAdd(const T* a, const T* b, T* c, int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int STRIDE = blockDim.x * gridDim.x;

    while (idx < N)
    {
        c[idx] = a[idx] + b[idx];
        idx += STRIDE;
    }

}

template<typename T>
void VectorAddGPU<T>::launchSetup()
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

template <typename T>
void VectorAddGPU<T>::vectorAdd()
{
    launchSetup();
    gpuVectorAdd << < gridSize, BLOCK_SIZE >> > (a.data(), b.data(), c.data(), N);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    gpuCheckErrors("gpu device sync failure");

    //copyD2H();
}

template void VectorAddGPU<float>::vectorAdd();
template void VectorAddGPU<double>::vectorAdd();
template VectorAddGPU<float>::~VectorAddGPU();
template VectorAddGPU<double>::~VectorAddGPU();
