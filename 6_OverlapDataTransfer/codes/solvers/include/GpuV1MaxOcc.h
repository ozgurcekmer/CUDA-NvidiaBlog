#pragma once

#include "../interface/ISolver.h"
#include "../../utilities/include/GpuCommon.h"

#include <vector>
#include <iostream>

template <typename T>
class GpuV1MaxOcc : public ISolver<T>
{
private:
    T* dA;

    const size_t BYTES = N * sizeof(T);
    const size_t STREAM_SIZE = N / N_STREAMS;
    const size_t STREAM_BYTES = BYTES / N_STREAMS;
    float ms = 0.0;
    size_t gridSize;

    void deviceAllocations();
    void copyH2D(size_t offset, gpuStream_t stream);
    void copyD2H(size_t offset, gpuStream_t stream);
    void launchSetup();

public:
    GpuV1MaxOcc(Vector::pinnedVector<T>& a) : ISolver<T>(a) {}

    virtual ~GpuV1MaxOcc();

    void solver() override;
};
