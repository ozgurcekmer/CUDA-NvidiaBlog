#pragma once

#include <vector>
#include <random>

#include "vectors/ManagedVector.h"

template <typename T>
class RandomVectorGenerator
{
public:
    void randomVector(std::vector<T>& v) const;
    void randomVector(Vector::managedVector<T>& v) const;
};
