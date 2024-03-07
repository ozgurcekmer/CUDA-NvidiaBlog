#pragma once

#include <vector>
#include <complex>
#include <iostream>
#include "ManagedVector.h"

template <typename T>
class MaxError
{
public:
    void maxError(const Vectors::managedVector<T>& v1, const Vectors::managedVector<T>& v2) const;
};
