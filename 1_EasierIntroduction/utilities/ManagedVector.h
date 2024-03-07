#pragma once

#include <vector>
#include <complex>

#include "GpuCommon.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace Vectors
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
            auto error = cudaMallocManaged((void**)&tmp, n * sizeof(T));
            if (error != cudaSuccess)
            {
                throw std::runtime_error
                {
                    cudaGetErrorString(error)
                };
            }
            return tmp;
        }

        auto deallocate(pointer p, size_type n) -> void
        {
            if (p)
            {
                auto error = cudaFree(p);
                if (error != cudaSuccess)
                {
                    throw std::runtime_error
                    {
                        cudaGetErrorString(error)
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