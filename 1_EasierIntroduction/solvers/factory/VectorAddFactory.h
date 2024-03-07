#pragma once

#include <memory>
#include <string>

#include "../interface/IVectorAdd.h"
#include "../include/VectorAddCPU.h"
#include "../include/VectorAddGPU.h"

template <typename T>
class VectorAddFactory
{
private:
	const Vectors::managedVector<T>& a;
	const Vectors::managedVector<T>& b;
	Vectors::managedVector<T>& c;

	std::shared_ptr<IVectorAdd<T>> solverSelect;

public:
	VectorAddFactory(const Vectors::managedVector<T>& a,
		const Vectors::managedVector<T>& b,
		Vectors::managedVector<T>& c) : a{ a }, b{ b }, c{ c } {}
	
	std::shared_ptr<IVectorAdd<T>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<VectorAddCPU<T>>(a, b, c);
		}
		else if (solverType == "gpu")
		{
			solverSelect = std::make_shared<VectorAddGPU<T>>(a, b, c);
		}
		return solverSelect;
	}

};
