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

template<typename T>
void GpuPrefetch<T>::prefetch()
{
    int device = -1;
    gpuGetDevice(&device);
    gpuMemPrefetchAsync(a.data(), SIZE, device, NULL);
    gpuMemPrefetchAsync(b.data(), SIZE * sizeof(T), device, NULL);
    gpuMemPrefetchAsync(c.data(), SIZE * sizeof(T), device, NULL);
}

template <typename T>
void GpuPrefetch<T>::vectorAdd()
{
    launchSetup();
    prefetch();
    gpuVectorAddPrefetch << < gridSize, BLOCK_SIZE >> > (a.data(), b.data(), c.data(), N);
    gpuCheckErrors("gpu kernel launch failure");
    gpuDeviceSynchronize();
    gpuCheckErrors("gpu device sync failure"); 
}

template void GpuPrefetch<float>::vectorAdd();
template void GpuPrefetch<double>::vectorAdd();
template GpuPrefetch<float>::~GpuPrefetch();
template GpuPrefetch<double>::~GpuPrefetch();
