// Solver interface 
#pragma once

#include "../../Parameters.h"
#include "../../utilities/include/vectors/PinnedVector.h"

#include <vector>
#include <iostream>

template <typename T>
class ISolver
{
protected:
    Vector::pinnedVector<T>& a;
        
public:
    ISolver(Vector::pinnedVector<T>& a) : a{ a } {}
    virtual ~ISolver() {}
    virtual void solver() = 0;
};

