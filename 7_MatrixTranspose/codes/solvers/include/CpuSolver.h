#pragma once

#include "../interface/ISolver.h"

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class CpuSolver : public ISolver<T>
{
private:

public:
    CpuSolver(std::vector<T>& v, std::vector<T>& y) : ISolver<T>(v, y) {}

    virtual ~CpuSolver() {}
    void solver() override;
};