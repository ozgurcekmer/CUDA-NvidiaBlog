#pragma once

#include <memory>
#include <string>

#include "../interface/IVectorAdd.h"
#include "../include/VectorAddGPU.h"
#include "../include/VectorAddCPU.h"
#include "../include/GpuPrefetch.h"
#include "../../utilities/include/vectors/ManagedVector.h"

template <typename T>
class VectorAddFactory
{
private:
	const Vector::managedVector<T>& a;
	const Vector::managedVector<T>& b;
	Vector::managedVector<T>& c;

	std::shared_ptr<IVectorAdd<T>> solverSelect;

public:
	VectorAddFactory(const Vector::managedVector<T>& a,
		const Vector::managedVector<T>& b,
		Vector::managedVector<T>& c) : a{ a }, b{ b }, c{ c } {}
	
	std::shared_ptr<IVectorAdd<T>> getSolver(std::string solverType)
	{
		if (solverType == "gpu")
		{
			solverSelect = std::make_shared<VectorAddGPU<T>>(a, b, c);
		}
		else if (solverType == "cpu")
		{
			solverSelect = std::make_shared<VectorAddCPU<T>>(a, b, c);
		}
		else if (solverType == "gpuPrefetch")
		{
			solverSelect = std::make_shared<GpuPrefetch<T>>(a, b, c);
		}
		return solverSelect;
	}

};
