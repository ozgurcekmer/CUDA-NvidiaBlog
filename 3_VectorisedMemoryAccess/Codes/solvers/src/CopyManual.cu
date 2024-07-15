#include "../include/CopyManual.h"
#include <iostream>


__global__
void copyKernel(float* A, float* B, size_t N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t STRIDE = blockDim.x * gridDim.x;

	while (idx < N)
	{
		B[idx] = A[idx];
		idx += STRIDE;
	}


}

template<typename T>
inline void CopyManual<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE);
	gpuMalloc(&dB, SIZE);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void CopyManual<T>::launchSetup()
{
	auto blocksPerSM = 2048 / blockSize;
	int devID;
	int numSMs;
	gpuGetDevice(&devID);

	gpuDeviceGetAttribute(&numSMs, gpuDevAttrMultiProcessorCount, devID);
	std::cout << "There are " << numSMs << " SMs in this device." << std::endl;
	std::cout << "Blocks per SM: " << blocksPerSM << std::endl;

	gridSize = blocksPerSM * numSMs;
	std::cout << "Grid Size: " << gridSize << std::endl;
	std::cout << "Block Size: " << blockSize << std::endl;
}

template<typename T>
void CopyManual<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dB, this->B.data(), SIZE, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void CopyManual<T>::copyD2H()
{
	gpuMemcpy(this->B.data(), this->dB, SIZE, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
CopyManual<T>::~CopyManual()
{
	gpuFree(dA);
	gpuFree(dB);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void CopyManual<T>::solver()
{
	deviceAllocations();
	copyH2D();
	launchSetup();
	copyKernel << <gridSize, blockSize >> > ((float*)dA, (float*)dB, N);
	copyD2H();
}

template void CopyManual<float>::deviceAllocations();
template void CopyManual<double>::deviceAllocations();
template void CopyManual<float>::copyH2D();
template void CopyManual<double>::copyH2D();
template void CopyManual<float>::copyD2H();
template void CopyManual<double>::copyD2H();
template void CopyManual<float>::solver();
template void CopyManual<double>::solver();
template CopyManual<float>::~CopyManual();
template CopyManual<double>::~CopyManual();







