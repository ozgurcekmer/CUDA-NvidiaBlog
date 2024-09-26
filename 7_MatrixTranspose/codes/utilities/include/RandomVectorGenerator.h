#pragma once

#include <vector>
#include <random>
#include <omp.h>
#include "vectors/PinnedVector.h"

template <typename T>
class RandomVectorGenerator
{
public:
    void randomVector(std::vector<T>& v) const;
    void randomVector(Vector::pinnedVector<T>&v) const;
};
