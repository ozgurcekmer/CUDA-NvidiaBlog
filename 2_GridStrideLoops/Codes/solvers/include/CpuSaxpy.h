#pragma once

#include "../interface/ISaxpy.h"

#include <vector>

template <typename T>
class CpuSaxpy : public ISaxpy<T>
{
private:
    
public:
    CpuSaxpy(const std::vector<T>& x,
        std::vector<T>& y) : ISaxpy<T>(x, y) {}
    
    virtual ~CpuSaxpy() {}

    void saxpy() override;
};
