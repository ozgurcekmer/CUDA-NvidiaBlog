#pragma once

#include "../interface/IVectorAdd.h"

#include <vector>

template <typename T>
class VectorAddCPU : public IVectorAdd<T>
{
private:
    
public:
    VectorAddCPU(const Vectors::managedVector<T>& a,
        const Vectors::managedVector<T>& b,
        Vectors::managedVector<T>& c) : IVectorAdd<T>(a, b, c) {}
    
    virtual ~VectorAddCPU() {}

    void vectorAdd() override;
};
