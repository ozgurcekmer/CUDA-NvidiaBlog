#pragma once

#include <vector>
#include <random>
#include "ManagedVector.h"

template <typename T>
class RandomVectorGenerator
{
public:
    void randomVector(Vectors::managedVector<T>& v) const;
};
