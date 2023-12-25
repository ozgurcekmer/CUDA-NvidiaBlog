// Solver interface 
#pragma once

#include "../../Parameters.h"

#include <vector>
#include <iostream>

template <typename T>
class IVectorAdd
{
protected:
    const Vectors::managedVector<T>& a;
    const Vectors::managedVector<T>& b;
    Vectors::managedVector<T>& c;
        
public:
    IVectorAdd(const Vectors::managedVector<T>& a,
    const Vectors::managedVector<T>& b,
        Vectors::managedVector<T>& c) : a{ a }, b{ b }, c{ c } {}
    virtual ~IVectorAdd() {}
    virtual void vectorAdd() = 0;
};

