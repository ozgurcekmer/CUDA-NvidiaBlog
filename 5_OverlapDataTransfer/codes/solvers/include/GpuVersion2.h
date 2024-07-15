#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuVersion2 : public ISolver<T>
{
private:
    T* dA;

    const size_t BYTES = N * sizeof(T);
    const size_t STREAM_SIZE = N / N_STREAMS;
    const size_t STREAM_BYTES = BYTES / N_STREAMS;
    float ms = 0.0;

    void deviceAllocations();
    void copyH2D(size_t offset, gpuStream_t stream);
    void copyD2H(size_t offset, gpuStream_t stream);
    //   void launchSetup();

public:
    GpuVersion2(Vector::pinnedVector<T>& a) : ISolver<T>(a) {}

    virtual ~GpuVersion2();

    void solver() override;
};
