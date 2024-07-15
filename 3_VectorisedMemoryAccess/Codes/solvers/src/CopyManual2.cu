#include "../include/CopyManual2.h"
#include <iostream>


__global__
void copyKernel2(float* A, float* B, size_t N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t STRIDE = blockDim.x * gridDim.x;

	while (idx < N / 2)
	{
		reinterpret_cast<float2*>(B)[idx] = reinterpret_cast<float2*>(A)[idx];
		idx += STRIDE;
	}

	if (idx == N / 2 && N % 2 == 1)
	{
		B[N - 1] = A[N - 1];
	}


}

template<typename T>
inline void CopyManual2<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE);
	gpuMalloc(&dB, SIZE);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void CopyManual2<T>::launchSetup()
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
void CopyManual2<T>::copyH2D()
{
	gpuMemcpy(this->dA, this->A.data(), SIZE, gpuMemcpyHostToDevice);
	gpuMemcpy(this->dB, this->B.data(), SIZE, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void CopyManual2<T>::copyD2H()
{
	gpuMemcpy(this->B.data(), this->dB, SIZE, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
CopyManual2<T>::~CopyManual2()
{
	gpuFree(dA);
	gpuFree(dB);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void CopyManual2<T>::solver()
{
	deviceAllocations();
	copyH2D();
	blockSize /= 2;
	launchSetup();
	copyKernel2 << <gridSize, blockSize >> > ((float*)dA, (float*)dB, N);
	copyD2H();
}

template void CopyManual2<float>::deviceAllocations();
template void CopyManual2<double>::deviceAllocations();
template void CopyManual2<float>::copyH2D();
template void CopyManual2<double>::copyH2D();
template void CopyManual2<float>::copyD2H();
template void CopyManual2<double>::copyD2H();
template void CopyManual2<float>::solver();
template void CopyManual2<double>::solver();
template CopyManual2<float>::~CopyManual2();
template CopyManual2<double>::~CopyManual2();







