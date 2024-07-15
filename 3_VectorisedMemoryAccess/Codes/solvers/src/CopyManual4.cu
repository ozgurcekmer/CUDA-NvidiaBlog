#include "../include/CopyManual4.h"
#include <iostream>


__global__
void copyKernel4(float* A, float* B, size_t N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t STRIDE = blockDim.x * gridDim.x;

	while (idx < N / 4)
	{
		reinterpret_cast<float4*>(B)[idx] = reinterpret_cast<float4*>(A)[idx];
		idx += STRIDE;
	}

	int remainder = N % 4;
	if (idx == N / 4 && remainder != 0)
	{
		while (remainder)
		{
			int idx = N - remainder--;
			B[idx] = A[idx];
		}
		
	}


}

template<typename T>
inline void CopyManual4<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE);
	gpuMalloc(&dB, SIZE);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void CopyManual4<T>::launchSetup()
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
void CopyManual4<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dB, this->B.data(), SIZE, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void CopyManual4<T>::copyD2H()
{
	gpuMemcpy(this->B.data(), this->dB, SIZE, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
CopyManual4<T>::~CopyManual4()
{
	gpuFree(dA);
	gpuFree(dB);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void CopyManual4<T>::solver()
{
	deviceAllocations();
	copyH2D();
	blockSize /= 4;
	launchSetup();
	copyKernel4 << <gridSize, blockSize >> > ((float*)dA, (float*)dB, N);
	copyD2H();
}

template void CopyManual4<float>::deviceAllocations();
template void CopyManual4<double>::deviceAllocations();
template void CopyManual4<float>::copyH2D();
template void CopyManual4<double>::copyH2D();
template void CopyManual4<float>::copyD2H();
template void CopyManual4<double>::copyD2H();
template void CopyManual4<float>::solver();
template void CopyManual4<double>::solver();
template CopyManual4<float>::~CopyManual4();
template CopyManual4<double>::~CopyManual4();







