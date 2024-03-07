#include "../include/VectorAddGPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
__global__
void gpuVectorAdd(const T* a, const T* b, T* c, int N) 
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
void VectorAddGPU<T>::deviceAllocations()
{
    // Allocate device vectors
    /*
    gpuMalloc(&dA, SIZE);
    gpuMalloc(&dB, SIZE);
    gpuMalloc(&dC, SIZE);
    */
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void VectorAddGPU<T>::copyH2D()
{
    /*
    gpuMemcpy(dA, this->a.data(), SIZE, cudaMemcpyHostToDevice);
    gpuMemcpy(dB, this->b.data(), SIZE, cudaMemcpyHostToDevice);
    */
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void VectorAddGPU<T>::copyD2H()
{
    /*
    gpuMemcpy(this->c.data(), dC, SIZE, gpuMemcpyDeviceToHost);
    */
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
VectorAddGPU<T>::~VectorAddGPU()
{
    // Deallocate device vectors
    /*
    gpuFree(dA);
    gpuFree(dB);
    gpuFree(dC);
    */
    gpuCheckErrors("gpuFree failure");
}

template <typename T>
void VectorAddGPU<T>::vectorAdd()
{
    //deviceAllocations();
    //copyH2D();

    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

    cout << "Block size: " << BLOCK_SIZE << endl;
    cout << "Grid size : " << GRID_SIZE << endl;
    cout << "Number of SMs: " << numSMs << endl;
    cout << "numSMs x 32: " << numSMs*32 << endl;

    // cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), cudaFuncCachePreferL1);

        
    gpuVectorAdd << < GRID_SIZE, BLOCK_SIZE >> > (a.data(), b.data(), c.data(), N);
    cudaDeviceSynchronize();
    gpuCheckErrors("gpu kernel launch failure");

    //copyD2H();
}

template void VectorAddGPU<float>::vectorAdd();
template void VectorAddGPU<double>::vectorAdd();
template void VectorAddGPU<float>::deviceAllocations();
template void VectorAddGPU<double>::deviceAllocations();
template void VectorAddGPU<float>::copyH2D();
template void VectorAddGPU<double>::copyH2D();
template VectorAddGPU<float>::~VectorAddGPU();
template VectorAddGPU<double>::~VectorAddGPU();
