#pragma once

#include <vector>
#include <complex>

#include "../GpuCommon.h"

namespace Vector
{
    template <typename T>
    class managedAlloc
    {
    public:
        using value_type = T;
        using pointer = value_type*;
        using size_type = std::size_t;

        managedAlloc() noexcept = default;

        template <typename U>
        managedAlloc(managedAlloc<U> const&) noexcept {}

        auto allocate(size_type n, const void* = 0) -> value_type*
        {
            value_type* tmp;
            auto error = gpuMallocManaged((void**)&tmp, n * sizeof(T));
            if (error != gpuSuccess)
            {
                throw std::runtime_error
                {
                    gpuGetErrorString(error)
                };
            }
            return tmp;
        }

        auto deallocate(pointer p, size_type n) -> void
        {
            if (p)
            {
                auto error = gpuFree(p);
                if (error != gpuSuccess)
                {
                    throw std::runtime_error
                    {
                        gpuGetErrorString(error)
                    };
                }
            }
            
        }
    };

    template <typename T, typename U>
    auto operator==(managedAlloc<T> const&, managedAlloc<U> const&) -> bool
    {
        return true;
    }

    template <typename T, typename U>
    auto operator!=(managedAlloc<T> const&, managedAlloc<U> const&) -> bool
    {
        return false;
    }

    template <typename T>
    using managedVector = std::vector<T, managedAlloc<T>>;


}