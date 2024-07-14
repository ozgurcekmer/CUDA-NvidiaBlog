#include "../include/GpuSeqMaxOcc.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuSeqMaxOcc(T* a, int offset, int n)
{
    int idx = offset + blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < n)
    {
        T iX = static_cast<T>(idx);
        T s = sin(iX);
        T c = cos(iX);
        a[idx] = sqrt(s * s + c * c);
        idx += stride;
    }
    
}

template<typename T>
void GpuSeqMaxOcc<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, SIZE);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSeqMaxOcc<T>::copyH2D()
{
    gpuMemcpy(dA, this->a.data(), SIZE, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSeqMaxOcc<T>::copyD2H()
{
    gpuMemcpy(this->a.data(), dA, SIZE, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}


template<typename T>
void GpuSeqMaxOcc<T>::launchSetup()
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
GpuSeqMaxOcc<T>::~GpuSeqMaxOcc()
{
    gpuFree(dA);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSeqMaxOcc<T>::solver()
{
    deviceAllocations();

    launchSetup();

    gpuEvent_t startEvent, stopEvent;
    gpuEventCreate(&startEvent);
    gpuEventCreate(&stopEvent);
    //gpuEventCreate(&dummyEvent);
    gpuCheckErrors("event create failure");

    gpuEventRecord(startEvent, 0);
    gpuCheckErrors("event record failure");
    int offset = 0;
    copyH2D();
    gpuSeqMaxOcc<T> << < gridSize, BLOCK_SIZE >> > (dA, offset, N);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
    gpuEventRecord(stopEvent, 0);
    gpuCheckErrors("event record failure");

    gpuEventSynchronize(stopEvent);
    gpuCheckErrors("event sync failure");
    gpuEventElapsedTime(&ms, startEvent, stopEvent);
    gpuCheckErrors("event elapsed time failure");
    cout << "Sequential max occupancy version passed time in ms: " << ms << endl;

    // Cleanup
    gpuEventDestroy(startEvent);
    gpuEventDestroy(stopEvent);


}

template void GpuSeqMaxOcc<float>::solver();
template void GpuSeqMaxOcc<double>::solver();
template void GpuSeqMaxOcc<float>::deviceAllocations();
template void GpuSeqMaxOcc<double>::deviceAllocations();
template void GpuSeqMaxOcc<float>::copyH2D();
template void GpuSeqMaxOcc<double>::copyH2D();
template GpuSeqMaxOcc<float>::~GpuSeqMaxOcc();
template GpuSeqMaxOcc<double>::~GpuSeqMaxOcc();
