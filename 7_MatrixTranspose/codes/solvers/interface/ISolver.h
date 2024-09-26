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
    std::vector<T>& v;
    std::vector<T>& y;
        
public:
    ISolver(std::vector<T>& v, std::vector<T>& y) : v{ v }, y { y } {}
    virtual ~ISolver() {}
    virtual void solver() = 0;
};

