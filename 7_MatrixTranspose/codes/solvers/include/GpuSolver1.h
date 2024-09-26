#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuSolver1 : public ISolver<T>
{
private:
    T* dV;
    T* dY;

    const size_t BYTES = N * N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuSolver1(std::vector<T>& v, std::vector<T>& y) : ISolver<T>(v, y) {}

    virtual ~GpuSolver1();
    void solver() override;
};
