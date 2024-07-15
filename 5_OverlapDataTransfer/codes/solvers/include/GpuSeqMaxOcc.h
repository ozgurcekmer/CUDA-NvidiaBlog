#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuSeqMaxOcc : public ISolver<T>
{
private:
    T* dA;

    const size_t SIZE = N * sizeof(T);
    float ms = 0.0;
    size_t gridSize;

    void deviceAllocations();
    void copyH2D();
    void copyD2H();
    void launchSetup();

public:
    GpuSeqMaxOcc(Vector::pinnedVector<T>& a) : ISolver<T>(a) {}

    virtual ~GpuSeqMaxOcc();

    void solver() override;
};
