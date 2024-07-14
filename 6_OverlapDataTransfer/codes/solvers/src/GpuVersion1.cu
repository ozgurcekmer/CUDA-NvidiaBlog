#include "../include/GpuVersion1.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuVersion1(T* a, int offset)
{
    int idx = offset + blockDim.x * blockIdx.x + threadIdx.x;
    
    T iX = static_cast<T>(idx);
    T s = sin(iX);
    T c = cos(iX);
    a[idx] = sqrt(s * s + c * c);
}

template<typename T>
void GpuVersion1<T>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dA, BYTES);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuVersion1<T>::copyH2D(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&dA[offset], &a[offset], STREAM_BYTES, gpuMemcpyHostToDevice, stream);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuVersion1<T>::copyD2H(size_t offset, gpuStream_t stream)
{
    gpuMemcpyAsync(&a[offset], &dA[offset], STREAM_BYTES, gpuMemcpyDeviceToHost, stream);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuVersion1<T>::~GpuVersion1()
{
    gpuFree(dA);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuVersion1<T>::solver()
{
    deviceAllocations();
    
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
    
    //launchSetup();

    //cout << "Grid size: " << GRID_SIZE << ", Block size: " << BLOCK_SIZE << endl;

    // VERSION 1 algorithm
    gpuEventRecord(startEvent, 0);
    gpuCheckErrors("event record failure");
    for (int i = 0; i < N_STREAMS; ++i)
    {
        int offset = i * STREAM_SIZE;
        //cout << "Stream " << i << " offset: " << offset << endl;
        copyH2D(offset, stream[i]);
        //cout << "Grid size: " << GRID_SIZE/N_STREAMS << ", Block size: " << BLOCK_SIZE << endl;
        gpuVersion1 << < GRID_SIZE/N_STREAMS, BLOCK_SIZE, 0, stream[i] >> > (dA, offset);
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

template void GpuVersion1<float>::solver();
template void GpuVersion1<double>::solver();
template void GpuVersion1<float>::deviceAllocations();
template void GpuVersion1<double>::deviceAllocations();
template void GpuVersion1<float>::copyH2D(size_t, gpuStream_t);
template void GpuVersion1<double>::copyH2D(size_t, gpuStream_t);
template void GpuVersion1<float>::copyD2H(size_t, gpuStream_t);
template void GpuVersion1<double>::copyD2H(size_t, gpuStream_t);
template GpuVersion1<float>::~GpuVersion1();
template GpuVersion1<double>::~GpuVersion1();
