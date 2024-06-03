#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

template <typename T>
class CopyDefault : public ISolver<T>
{
private:
	T* dA;
	T* dB;

	const size_t SIZE =  N * sizeof(T);

	void deviceAllocations();
	void copyH2D();
	void copyD2H();

public:
	CopyDefault(const std::vector<T>& A,
		std::vector<T>& B) : ISolver<T>(A, B) {}

	virtual ~CopyDefault();
	void solver() override;
};