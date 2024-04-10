// Solver interface 
#pragma once

#include "../../Parameters.h"

#include <vector>
#include <iostream>

template <typename T>
class ISaxpy
{
protected:
    const std::vector<T>& x;
    std::vector<T>& y;
        
public:
    ISaxpy(const std::vector<T>& x,
    std::vector<T>& y) : x{ x }, y{ y } {}
    virtual ~ISaxpy() {}
    virtual void saxpy() = 0;
};

