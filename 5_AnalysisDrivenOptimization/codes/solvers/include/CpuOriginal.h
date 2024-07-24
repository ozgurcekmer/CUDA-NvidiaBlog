#pragma once

#include "../interface/ISolver.h"

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class CpuOriginal : public ISolver<T>
{
private:

public:
    CpuOriginal(std::vector<T>& v, std::vector<T>& A, std::vector<T>& y) : ISolver<T>(v, A, y) {}

    virtual ~CpuOriginal() {}
    void solver() override;
};
