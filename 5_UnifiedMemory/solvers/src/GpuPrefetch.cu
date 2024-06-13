#include "../include/GpuPrefetch.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuVectorAddPrefetch(const T* a, const T* b, T* c, int N)
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
void GpuPrefetch<T>::launchSetup()
{
    auto blocksPerSM = 2048 / BLOCK_SIZE;
    int devID;
    int numSMs;
    
    gpuGetDevice(&devID);
    cout << "DevID = " << devID << endl;
    gpuDeviceGetAttribute(&numSMs, gpuDevAttrMultiProcessorCount, devID);
    
    gpuMemPrefetchAsync(a.data(), SIZE, devID, NULL);
    gpuMemPrefetchAsync(b.data(), SIZE, devID, NULL);
    gpuMemPrefetchAsync(c.data(), SIZE, devID, NULL);
    gpuCheckErrors("gpu prefetch async failure");

    std::cout << "There are " << numSMs << " SMs in this device." << std::endl;
    std::cout << "Blocks per SM: " << blocksPerSM << std::endl;

    gridSize = blocksPerSM * numSMs;
    std::cout << "Grid Size: " << gridSize << std::endl;
    std::cout << "Block Size: " << BLOCK_SIZE << std::endl;
}

template <typename T>
void GpuPrefetch<T>::vectorAdd()
{
 
    launchSetup();
    //gpuVectorAdd << < 1, 1 >> > (a.data(), b.data(), c.data(), N);
    gpuVectorAddPrefetch << < gridSize, BLOCK_SIZE >> > (a.data(), b.data(), c.data(), N);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    gpuCheckErrors("gpu device sync failure"); 
}

template void GpuPrefetch<float>::vectorAdd();
template void GpuPrefetch<double>::vectorAdd();
template GpuPrefetch<float>::~GpuPrefetch();
template GpuPrefetch<double>::~GpuPrefetch();
