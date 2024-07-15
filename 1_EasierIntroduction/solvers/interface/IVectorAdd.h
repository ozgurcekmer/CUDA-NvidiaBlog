// Solver interface 
#pragma once

#include "../../Parameters.h"
#include "../../utilities/include/vectors/ManagedVector.h"

#include <vector>
#include <iostream>

template <typename T>
class IVectorAdd
{
protected:
    const Vector::managedVector<T>& a;
    const Vector::managedVector<T>& b;
    Vector::managedVector<T>& c;
        
public:
    IVectorAdd(const Vector::managedVector<T>& a,
    const Vector::managedVector<T>& b,
        Vector::managedVector<T>& c) : a{ a }, b{ b }, c{ c } {}
    virtual ~IVectorAdd() {}
    virtual void vectorAdd() = 0;
};

