#pragma once

#include <memory>

#include "../interface/ISolver.h"
#include "../include/CopyDefault.h"
#include "../include/CopyManual.h"
#include "../include/CopyManual2.h"
#include "../include/CopyManual4.h"

template <typename T>
class SolverFactory
{
private:
	const std::vector<T>& A;
	std::vector<T>& B;
	
	std::shared_ptr<ISolver<T>> solverSelect;

public:
	SolverFactory(const std::vector<T>& A,
		std::vector<T>& B) : A{ A }, B{ B } {}

	std::shared_ptr<ISolver<T>> getSolver(std::string solverType)
	{
		if (solverType == "gpuDefault")
		{
			solverSelect = std::make_shared<CopyDefault<T>>(A, B);
		}
		else if (solverType == "gpuManual")
		{
			solverSelect = std::make_shared<CopyManual<T>>(A, B);
		}
		else if (solverType == "gpuManual2")
		{
			solverSelect = std::make_shared<CopyManual2<T>>(A, B);
		}
		else if (solverType == "gpuManual4")
		{
			solverSelect = std::make_shared<CopyManual4<T>>(A, B);
		}

		return solverSelect;
	}
};