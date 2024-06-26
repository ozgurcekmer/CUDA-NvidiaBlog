#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuSequential : public ISolver<T>
{
private:
    T* dA;

    const size_t SIZE = N * sizeof(T);
 //   size_t gridSize;
    
    void deviceAllocations();
    void copyH2D();
    void copyD2H();
 //   void launchSetup();

public:
    GpuSequential(Vector::pinnedVector<T>& a) : ISolver<T>(a) {}

    virtual ~GpuSequential();

    void solver() override;
};
