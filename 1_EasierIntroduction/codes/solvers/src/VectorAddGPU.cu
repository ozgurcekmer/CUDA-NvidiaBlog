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
    int devID;
    int numSMs;
    gpuGetDevice(&devID);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, devID);
    int maxThreadsPerSM = properties.maxThreadsPerMultiProcessor;

    gpuDeviceGetAttribute(&numSMs, gpuDevAttrMultiProcessorCount, devID);
    auto blocksPerSM = maxThreadsPerSM / BLOCK_SIZE;
    std::cout << "There are " << numSMs << " SMs in this device." << std::endl;
    std::cout << "Max number of threads per SM: " << maxThreadsPerSM << endl;
    std::cout << "Block Size: " << BLOCK_SIZE << std::endl;
    std::cout << "Blocks per SM (maxThreadsPerSM / BLOCK_SIZE): " << blocksPerSM << std::endl;

    gridSize = blocksPerSM * numSMs;
    std::cout << "Grid Size (BlocksPerSM * numSMs) : " << gridSize << std::endl;
}

template <typename T>
void VectorAddGPU<T>::vectorAdd()
{
    launchSetup();
    gridSize *= 32;
    //gridSize = 1;
    cout << "Launch configuration: (" << gridSize << ", " << BLOCK_SIZE << ")" << endl;
    gpuVectorAdd << < gridSize, BLOCK_SIZE >> > (a.data(), b.data(), c.data(), N);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    gpuCheckErrors("gpu device sync failure");
}

template void VectorAddGPU<float>::vectorAdd();
template void VectorAddGPU<double>::vectorAdd();
template VectorAddGPU<float>::~VectorAddGPU();
template VectorAddGPU<double>::~VectorAddGPU();
