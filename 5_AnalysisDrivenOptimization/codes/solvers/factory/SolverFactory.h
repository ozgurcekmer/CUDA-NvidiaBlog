#pragma once

#include <memory>
#include <string>

#include "../interface/ISolver.h"
#include "../include/GpuSequential.h"
#include "../include/GpuSolver1.h"
#include "../include/GpuSolver2.h"
#include "../include/CpuOriginal.h"
#include "../include/CpuSolver.h"
#include "../../utilities/include/vectors/PinnedVector.h"

template <typename T>
class SolverFactory
{
private:
	std::vector<T>& v;
	std::vector<T>& A;
	std::vector<T>& y;
	
	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(std::vector<T>& v, std::vector<T>& A, std::vector<T>& y) : v{ v }, A{ A }, y{ y } {}
	
	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "gpuSequential")
		{
			solverSelect = std::make_shared<GpuSequential<T>>(v, A, y);
		}
		else if (solverType == "gpuSolver1")
		{
			solverSelect = std::make_shared<GpuSolver1<T>>(v, A, y);
		}
		else if (solverType == "gpuSolver2")
		{
			solverSelect = std::make_shared<GpuSolver2<T>>(v, A, y);
		}
		else if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(v, A, y);
		}
		else if (solverType == "cpuOriginal")
		{
			solverSelect = std::make_shared<CpuOriginal<T>>(v, A, y);
		}
		return solverSelect;
	}

};
