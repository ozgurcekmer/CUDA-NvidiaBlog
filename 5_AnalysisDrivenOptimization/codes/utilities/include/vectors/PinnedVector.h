#pragma once

#include <vector>
#include <complex>

#include "../GpuCommon.h"

namespace Vector
{
    template <typename T>
    class pinnedAlloc
    {
    public:
        using value_type = T;
        using pointer = value_type*;
        using size_type = std::size_t;

        pinnedAlloc() noexcept = default;

        template <typename U>
        pinnedAlloc(pinnedAlloc<U> const&) noexcept {}

        auto allocate(size_type n, const void* = 0) -> value_type*
        {
            value_type* tmp;
            auto error = gpuMallocHost((void**)&tmp, n * sizeof(T));
            //auto error = gpuHostAlloc((void**)&tmp, n * sizeof(T), gpuHostAllocDefault);
            //auto error = gpuHostAlloc((void**)&tmp, n * sizeof(T), gpuHostAllocMapped);
            //auto error = gpuHostAlloc((void**)&tmp, n * sizeof(T), gpuHostAllocPortable);
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
                auto error = gpuFreeHost(p);
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
    auto operator==(pinnedAlloc<T> const&, pinnedAlloc<U> const&) -> bool
    {
        return true;
    }

    template <typename T, typename U>
    auto operator!=(pinnedAlloc<T> const&, pinnedAlloc<U> const&) -> bool
    {
        return false;
    }

    template <typename T>
    using pinnedVector = std::vector<T, pinnedAlloc<T>>;


}