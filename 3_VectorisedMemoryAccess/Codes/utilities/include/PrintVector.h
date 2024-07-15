#pragma once

#include <vector>
#include <iostream>
#include <complex>

#include "../../Parameters.h"

template <typename T>
class PrintVector
{
public:
    void printVector(const std::vector<T>& v) const;
};

