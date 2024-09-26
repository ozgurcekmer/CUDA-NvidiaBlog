#pragma once

#include <vector>
#include <complex>
#include <iostream>

#include "vectors/ManagedVector.h"
#include "vectors/PinnedVector.h"

template <typename T>
class MaxError
{
public:
    void maxError(const std::vector<T>& v1, const std::vector<T>& v2) const;
    void maxError(const Vector::managedVector<T>& v1, const Vector::managedVector<T>& v2) const;
    void maxError(const Vector::pinnedVector<T>& v1, const Vector::pinnedVector<T>& v2) const;
};
