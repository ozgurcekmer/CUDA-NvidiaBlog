#pragma once

#include <memory>
#include <string>

#include "../interface/ISaxpy.h"
#include "../include/CpuSaxpy.h"
#include "../include/GpuGridStride.h"
#include "../include/GpuSaxpy.h"

template <typename T>
class SaxpyFactory
{
private:
	const std::vector<T>& x;
	std::vector<T>& y;

	std::shared_ptr<ISaxpy<T>> solverSelect;

public:
	SaxpyFactory(const std::vector<T>& x,
		std::vector<T>& y) : x{ x }, y{ y } {}
	
	std::shared_ptr<ISaxpy<T>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSaxpy<T>>(x, y);
		}
		else if (solverType == "gpuCommon")
		{
			solverSelect = std::make_shared<GpuSaxpy<T>>(x, y);
		}
		else if (solverType == "gpuGridStride")
		{
			solverSelect = std::make_shared<GpuGridStride<T>>(x, y);
		}
		return solverSelect;
	}

};
