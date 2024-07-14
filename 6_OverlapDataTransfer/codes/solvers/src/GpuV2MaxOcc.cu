#include "../include/GpuV2MaxOcc.h"

using std::cout;
using std::endl;
using std::vector;


template <typename T>
__global__
void gpuV2MaxOcc(T* a, int offset, int offsetEnd)
{
    int idx = offset + blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < offsetEnd)
    {
        T iX = static_cast<T>(idx);
        T s = sin(iX);
        T c = cos(iX);
        a[idx] = sqrt(s * s + c * c);
        idx += stride;
    }

}

template<typename T>
void GpuV2MaxOcc<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuV2MaxOcc<T>::copyH2D(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&dA[offset], &a[offset], STREAM_BYTES, gpuMemcpyHostToDevice, stream);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuV2MaxOcc<T>::copyD2H(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&a[offset], &dA[offset], STREAM_BYTES, gpuMemcpyDeviceToHost, stream);
    gpuCheckErrors("gpuMemcpy D2H failure");
}


template<typename T>
void GpuV2MaxOcc<T>::launchSetup()
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
GpuV2MaxOcc<T>::~GpuV2MaxOcc()
{
    gpuFree(dA);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuV2MaxOcc<T>::solver()
{
    deviceAllocations();

    launchSetup();

    // Stream setup
    gpuEvent_t startEvent, stopEvent;
    gpuStream_t stream[N_STREAMS];
    gpuEventCreate(&startEvent);
    gpuEventCreate(&stopEvent);
    gpuCheckErrors("event create failure");

    for (int i = 0; i < N_STREAMS; ++i)
    {
        gpuStreamCreate(&stream[i]);
        gpuCheckErrors("stream create failure");
    }

    // VERSION 2 algorithm
    gpuEventRecord(startEvent, 0);
    gpuCheckErrors("event record failure");
    for (int i = 0; i < N_STREAMS; ++i)
    {
        int offset = i * STREAM_SIZE;
        copyH2D(offset, stream[i]);
    }
    for (int i = 0; i < N_STREAMS; ++i)
    {
        int offset = i * STREAM_SIZE;
        int offsetEnd = offset + STREAM_SIZE;
        gpuV2MaxOcc << < gridSize / N_STREAMS, BLOCK_SIZE, 0, stream[i] >> > (dA, offset, offsetEnd);
    }
    for (int i = 0; i < N_STREAMS; ++i)
    {
        int offset = i * STREAM_SIZE;
        copyD2H(offset, stream[i]);
    }

    gpuEventRecord(stopEvent, 0);
    gpuCheckErrors("event record failure");
    gpuEventSynchronize(stopEvent);
    gpuCheckErrors("event sync failure");
    gpuEventElapsedTime(&ms, startEvent, stopEvent);
    gpuCheckErrors("event elapsed time failure");
    cout << "Version2 passed time in ms: " << ms << endl;

    // Cleanup
    gpuEventDestroy(startEvent);
    gpuEventDestroy(stopEvent);
    for (int i = 0; i < N_STREAMS; ++i)
    {
        gpuStreamDestroy(stream[i]);
        gpuCheckErrors("stream destroy failure");
    }

}

template void GpuV2MaxOcc<float>::solver();
template void GpuV2MaxOcc<double>::solver();
template void GpuV2MaxOcc<float>::deviceAllocations();
template void GpuV2MaxOcc<double>::deviceAllocations();
template void GpuV2MaxOcc<float>::copyH2D(size_t, gpuStream_t);
template void GpuV2MaxOcc<double>::copyH2D(size_t, gpuStream_t);
template void GpuV2MaxOcc<float>::copyD2H(size_t, gpuStream_t);
template void GpuV2MaxOcc<double>::copyD2H(size_t, gpuStream_t);
template GpuV2MaxOcc<float>::~GpuV2MaxOcc();
template GpuV2MaxOcc<double>::~GpuV2MaxOcc();
