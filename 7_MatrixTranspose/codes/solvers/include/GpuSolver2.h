#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuSolver2 : public ISolver<T>
{
private:
    T* dV;
    T* dY;

    const size_t BYTES = N * N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuSolver2(std::vector<T>& v, std::vector<T>& y) : ISolver<T>(v, y) {}

    virtual ~GpuSolver2();
    void solver() override;
};
