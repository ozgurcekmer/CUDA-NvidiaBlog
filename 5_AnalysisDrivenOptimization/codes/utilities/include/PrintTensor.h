#pragma once

#include <vector>
#include <iostream>
#include <complex>

#include "../../Parameters.h"

template <typename T>
class PrintTensor
{
public:
    void printTensor(const std::vector<T>& v, const size_t N, const size_t M, const size_t L) const;
};

