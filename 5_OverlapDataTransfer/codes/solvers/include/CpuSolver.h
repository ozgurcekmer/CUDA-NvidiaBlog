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
    CpuSolver(Vector::pinnedVector<T>& a) : ISolver<T>(a) {}

    virtual ~CpuSolver() {}

    void solver() override;
};
