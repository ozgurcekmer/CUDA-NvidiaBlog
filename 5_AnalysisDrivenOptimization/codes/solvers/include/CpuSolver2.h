#pragma once

#include "../interface/ISolver.h"

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class CpuSolver2 : public ISolver<T>
{
private:
    
public:
    CpuSolver2(std::vector<T>& v, std::vector<T>& A, std::vector<T>& y) : ISolver<T>(v, A, y) {}

    virtual ~CpuSolver2() {}
    void solver() override;
};
