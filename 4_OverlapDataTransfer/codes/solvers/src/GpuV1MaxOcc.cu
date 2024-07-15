#include "../include/GpuV1MaxOcc.h"

using std::cout;
using std::endl;
using std::vector;


template <typename T>
__global__
void gpuV1MaxOcc(T* a, int offset, int offsetEnd)
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
void GpuV1MaxOcc<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuV1MaxOcc<T>::copyH2D(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&dA[offset], &a[offset], STREAM_BYTES, gpuMemcpyHostToDevice, stream);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuV1MaxOcc<T>::copyD2H(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&a[offset], &dA[offset], STREAM_BYTES, gpuMemcpyDeviceToHost, stream);
    gpuCheckErrors("gpuMemcpy D2H failure");
}


template<typename T>
void GpuV1MaxOcc<T>::launchSetup()
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
GpuV1MaxOcc<T>::~GpuV1MaxOcc()
{
    gpuFree(dA);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuV1MaxOcc<T>::solver()
{
    deviceAllocations();

    launchSetup();

    // Stream setup
    gpuEvent_t startEvent, stopEvent;
    //gpuEvent_t dummyEvent;
    gpuStream_t stream[N_STREAMS];
    gpuEventCreate(&startEvent);
    gpuEventCreate(&stopEvent);
    //gpuEventCreate(&dummyEvent);
    gpuCheckErrors("event create failure");

    for (int i = 0; i < N_STREAMS; ++i)
    {
        gpuStreamCreate(&stream[i]);
        gpuCheckErrors("stream create failure");
    }

    //cout << "Grid size: " << GRID_SIZE << ", Block size: " << BLOCK_SIZE << endl;

    // VERSION 1 algorithm
    gpuEventRecord(startEvent, 0);
    gpuCheckErrors("event record failure");
    for (int i = 0; i < N_STREAMS; ++i)
    {
        int offset = i * STREAM_SIZE;
        int offsetEnd = offset + STREAM_SIZE;
        //cout << "Stream " << i << " offset: " << offset << endl;
        copyH2D(offset, stream[i]);
        //cout << "Grid size: " << GRID_SIZE/N_STREAMS << ", Block size: " << BLOCK_SIZE << endl;
        gpuV1MaxOcc << < gridSize / N_STREAMS, BLOCK_SIZE, 0, stream[i] >> > (dA, offset, offsetEnd);
        copyD2H(offset, stream[i]);
        gpuStreamQuery(stream[i]);
    }
    //gpuEventRecord(dummyEvent, 0);
    gpuEventRecord(stopEvent, 0);
    gpuCheckErrors("event record failure");
    //gpuEventSynchronize(dummyEvent);
    gpuEventSynchronize(stopEvent);
    gpuCheckErrors("event sync failure");
    gpuEventElapsedTime(&ms, startEvent, stopEvent);
    gpuCheckErrors("event elapsed time failure");
    cout << "Version1 passed time in ms: " << ms << endl;

    // Cleanup
    gpuEventDestroy(startEvent);
    gpuEventDestroy(stopEvent);
    //gpuEventDestroy(dummyEvent);
    for (int i = 0; i < N_STREAMS; ++i)
    {
        gpuStreamDestroy(stream[i]);
        gpuCheckErrors("stream destroy failure");
    }

}

template void GpuV1MaxOcc<float>::solver();
template void GpuV1MaxOcc<double>::solver();
template void GpuV1MaxOcc<float>::deviceAllocations();
template void GpuV1MaxOcc<double>::deviceAllocations();
template void GpuV1MaxOcc<float>::copyH2D(size_t, gpuStream_t);
template void GpuV1MaxOcc<double>::copyH2D(size_t, gpuStream_t);
template void GpuV1MaxOcc<float>::copyD2H(size_t, gpuStream_t);
template void GpuV1MaxOcc<double>::copyD2H(size_t, gpuStream_t);
template GpuV1MaxOcc<float>::~GpuV1MaxOcc();
template GpuV1MaxOcc<double>::~GpuV1MaxOcc();
