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

template<typename T>
GpuSequential<T>::~GpuSequential()
{
    gpuFree(dA);
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void GpuSequential<T>::solver()
{
    deviceAllocations();

    gpuEvent_t startEvent, stopEvent;
    gpuEventCreate(&startEvent);
    gpuEventCreate(&stopEvent);
    gpuCheckErrors("event create failure");

    gpuEventRecord(startEvent, 0);
    gpuCheckErrors("event record failure");
    copyH2D();

    int offset = 0;
    gpuSequential<T> << < GRID_SIZE, BLOCK_SIZE >> > (dA, offset);
    gpuCheckErrors("gpu kernel launch failure");
    copyD2H();
    gpuEventRecord(stopEvent, 0);
    gpuCheckErrors("event record failure");
    
    gpuEventSynchronize(stopEvent);
    gpuCheckErrors("event sync failure");
    gpuEventElapsedTime(&ms, startEvent, stopEvent);
    gpuCheckErrors("event elapsed time failure");
    cout << "Sequential version passed time in ms: " << ms << endl;
    
    // Cleanup
    gpuEventDestroy(startEvent);
    gpuEventDestroy(stopEvent);

}

template void GpuSequential<float>::solver();
template void GpuSequential<double>::solver();
template void GpuSequential<float>::deviceAllocations();
template void GpuSequential<double>::deviceAllocations();
template void GpuSequential<float>::copyH2D();
template void GpuSequential<double>::copyH2D();
template GpuSequential<float>::~GpuSequential();
template GpuSequential<double>::~GpuSequential();
