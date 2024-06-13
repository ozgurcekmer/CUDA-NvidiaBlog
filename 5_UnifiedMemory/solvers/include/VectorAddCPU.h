#pragma once

#include "../interface/IVectorAdd.h"

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class VectorAddCPU : public IVectorAdd<T>
{
private:

    
public:
    VectorAddCPU(const Vector::managedVector<T>& a,
        const Vector::managedVector<T>& b,
        Vector::managedVector<T>& c) : IVectorAdd<T>(a, b, c) {}

    virtual ~VectorAddCPU() {}

    void vectorAdd() override;
};
