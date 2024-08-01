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
    T* dA;
    T* dY;

    const size_t BYTES_V = L * M * N * sizeof(T);
    const size_t BYTES_A = L * L * sizeof(T);
    const size_t BYTES_Y = L * N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuSolver1(std::vector<T>& v, std::vector<T>& A, std::vector<T>& y) : ISolver<T>(v, A, y) {}

    virtual ~GpuSolver1();
    void solver() override;
};
