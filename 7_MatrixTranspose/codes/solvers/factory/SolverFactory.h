#pragma once

#include <memory>
#include <string>

#include "../interface/ISolver.h"
#include "../include/GpuSolver1.h"
#include "../include/GpuSolver2.h"
#include "../include/GpuSolver3.h"
#include "../include/CpuSolver.h"
#include "../../utilities/include/vectors/PinnedVector.h"

template <typename T>
class SolverFactory
{
private:
	std::vector<T>& v;
	std::vector<T>& y;
	
	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(std::vector<T>& v, std::vector<T>& y) : v{ v }, y{ y } {}
	
	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "gpuSolver1")
		{
			solverSelect = std::make_shared<GpuSolver1<T>>(v, y);
		}
		else if (solverType == "gpuSolver2")
		{
			solverSelect = std::make_shared<GpuSolver2<T>>(v, y);
		}
		else if (solverType == "gpuSolver3")
		{
			solverSelect = std::make_shared<GpuSolver3<T>>(v, y);
		}
		else if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(v, y);
		}
		return solverSelect;
	}

};
