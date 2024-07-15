#pragma once

#include <memory>
#include <string>

#include "../interface/ISolver.h"
#include "../include/GpuSequential.h"
#include "../include/GpuSeqMaxOcc.h"
#include "../include/GpuVersion1.h"
#include "../include/GpuVersion2.h"
#include "../include/GpuV1MaxOcc.h"
#include "../include/GpuV2MaxOcc.h"
#include "../include/CpuSolver.h"
#include "../../utilities/include/vectors/PinnedVector.h"

template <typename T>
class SolverFactory
{
private:
	Vector::pinnedVector<T>& a;

	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(Vector::pinnedVector<T>& a) : a{ a } {}
	
	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "gpuSequential")
		{
			solverSelect = std::make_shared<GpuSequential<T>>(a);
		}
		else if (solverType == "gpuSeqMaxOcc")
		{
			solverSelect = std::make_shared<GpuSeqMaxOcc<T>>(a);
		}
		else if (solverType == "gpuVersion1")
		{
			solverSelect = std::make_shared<GpuVersion1<T>>(a);
		}
		else if (solverType == "gpuVersion2")
		{
			solverSelect = std::make_shared<GpuVersion2<T>>(a);
		}
		else if (solverType == "gpuV1MaxOcc")
		{
			solverSelect = std::make_shared<GpuV1MaxOcc<T>>(a);
		}
		else if (solverType == "gpuV2MaxOcc")
		{
			solverSelect = std::make_shared<GpuV2MaxOcc<T>>(a);
		}
		else if (solverType == "cpu")
		{
			solverSelect = std::make_shared<CpuSolver<T>>(a);
		}
		
		return solverSelect;
	}

};
