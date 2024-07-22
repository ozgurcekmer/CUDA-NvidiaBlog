#pragma once

#include "../interface/ISolver.h"

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class CpuSolver3 : public ISolver<T>
{
private:
    
public:
    CpuSolver3(std::vector<T>& v, std::vector<T>& A, std::vector<T>& y) : ISolver<T>(v, A, y) {}

    virtual ~CpuSolver3() {}
    void solver() override;
};
