#pragma once

#include "../interface/ISaxpy.h"

#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuGridStride : public ISaxpy<T>
{
private:
    T* dX;
    T* dY;

    const size_t SIZE = N * sizeof(T);

    void deviceAllocations();
    void copyH2D();
    void copyD2H();

public:
    GpuGridStride(const std::vector<T>& x,
        std::vector<T>& y) : ISaxpy<T>(x, y) {}

    virtual ~GpuGridStride();

    void saxpy() override;
};
