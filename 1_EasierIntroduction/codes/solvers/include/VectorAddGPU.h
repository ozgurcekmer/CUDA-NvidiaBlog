#pragma once

#include "../interface/IVectorAdd.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class VectorAddGPU : public IVectorAdd<T>
{
private:

    size_t gridSize;
    const size_t SIZE = N * sizeof(T);
    void launchSetup();

public:
    VectorAddGPU(const Vector::managedVector<T>& a,
        const Vector::managedVector<T>& b,
        Vector::managedVector<T>& c) : IVectorAdd<T>(a, b, c) {}

    virtual ~VectorAddGPU() {}

    void vectorAdd() override;
};
