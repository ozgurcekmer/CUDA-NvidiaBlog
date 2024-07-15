#include "../include/CopyDefault.h"
#include <iostream>

/*
__global__
void copyKernel(float* A, float* B, size_t N)
{
	// To be written...
}
*/

template<typename T>
inline void CopyDefault<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE);
	gpuMalloc(&dB, SIZE);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void CopyDefault<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dB, this->B.data(), SIZE, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void CopyDefault<T>::copyD2H()
{
	gpuMemcpy(this->B.data(), this->dB, SIZE, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
CopyDefault<T>::~CopyDefault()
{
	gpuFree(dA);
	gpuFree(dB);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void CopyDefault<T>::solver()
{
	deviceAllocations();
	copyH2D();
	gpuMemcpy(this->dB, this->dA, SIZE, gpuMemcpyDeviceToDevice);
	copyD2H();
}

template void CopyDefault<float>::deviceAllocations();
template void CopyDefault<double>::deviceAllocations();
template void CopyDefault<float>::copyH2D();
template void CopyDefault<double>::copyH2D();
template void CopyDefault<float>::copyD2H();
template void CopyDefault<double>::copyD2H();
template void CopyDefault<float>::solver();
template void CopyDefault<double>::solver();
template CopyDefault<float>::~CopyDefault();
template CopyDefault<double>::~CopyDefault();







