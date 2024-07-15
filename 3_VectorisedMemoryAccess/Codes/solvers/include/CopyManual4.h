#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

template <typename T>
class CopyManual4 : public ISolver<T>
{
private:
	T* dA;
	T* dB;

	const size_t SIZE = N * sizeof(T);

	size_t blockSize = BLOCK_SIZE;
	size_t gridSize;

	void deviceAllocations();
	void copyH2D();
	void copyD2H();
	void launchSetup();

public:
	CopyManual4(const std::vector<T>& A,
		std::vector<T>& B) : ISolver<T>(A, B) {}

	virtual ~CopyManual4();
	void solver() override;
};