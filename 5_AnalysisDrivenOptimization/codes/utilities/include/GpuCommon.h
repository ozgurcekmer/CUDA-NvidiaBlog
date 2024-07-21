#pragma once

#include <iostream>

#ifdef USEHIP
#define __GPU_API__ "HIP"
#define __GPU_TO_SECONDS__ 1.0/1000.0
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#elif defined(USECUDA)
#define __GPU_API__ "CUDA"
#define __GPU_TO_SECONDS__ 1.0/1000.0
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#endif

#if defined(USEHIP)
#define gpuStreamQuery hipStreamQuery
#define gpuMalloc hipMalloc
#define gpuHostMalloc hipMallocHost
#define gpuMallocHost hipHostMalloc
#define gpuMallocManaged hipMallocManaged
#define gpuMemset hipMemset
#define gpuFree hipFree
#define gpuFreeHost hipHostFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define gpuGetErrorString hipGetErrorString
#define gpuError_t hipError_t
#define gpuErr hipErr
#define gpuGetLastError hipGetLastError
#define gpuSuccess hipSuccess
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceGetPCIBusId hipDeviceGetPCIBusId
#define gpuMemGetInfo hipMemGetInfo
#define gpuMemPrefetchAsync hipMemPrefetchAsync
#define gpuDeviceReset hipDeviceReset
#define gpuLaunchKernel(...) hipLaunchKernelGGL(__VA_ARGS__)
#define GPU_VISIBLE_DEVICES "ROCR_VISIBLE_DEVICES"
#define fullMask 0xFFFFFFFFFFFFFFFFU

#elif defined(USECUDA)
//#define gpuStream_t ((cudaStream_t)0x2)
#define gpuStreamQuery cudaStreamQuery
#define gpuMalloc cudaMalloc
#define gpuMallocHost cudaMallocHost
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuFreeHost cudaFreeHost
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuError_t cudaError_t
#define gpuErr cudaErr
#define gpuMemset cudaMemset
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuSuccess cudaSuccess
#define gpuDeviceProp_t cudaDeviceProp
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceGetPCIBusId cudaDeviceGetPCIBusId
#define gpuMemPrefetchAsync cudaMemPrefetchAsync
#define gpuMemGetInfo cudaMemGetInfo
#define gpuDeviceReset cudaDeviceReset
#define gpuLaunchKernel(...) cudaLaunchKernel(__VA_ARGS__)
#define GPU_VISIBLE_DEVICES "CUDA_VISIBLE_DEVICES"
#define fullMask 0xFFFFFFFFU
#endif

 // macro for checking errors in HIP API calls
#define gpuErrorCheck(call)                                                                 \
do{                                                                                         \
    gpuError_t __gpuErr = call;                                                               \
    if(__gpuErr != gpuSuccess){                                                               \
        std::cerr<<__GPU_API__<<" Fatal error : "<<gpuGetErrorString(__gpuErr) \
        <<" - "<<__FILE__<<":"<<__LINE__<<std::endl; \
        std::cerr<<" *** FAILED - ABORTING "<<std::endl; \
        exit(1);                                                                            \
    }                                                                                       \
}while(0)

#define gpuCheckErrors(msg) \
    do { \
        gpuError_t __gpuErr = gpuGetLastError(); \
        if (__gpuErr != gpuSuccess) { \
            std::cerr<<__GPU_API__<<" Fatal error: "<<msg<<" ("<<gpuGetErrorString(__gpuErr) \
            <<" at "<<__FILE__<<":"<<__LINE__<<")"<<std::endl; \
            std::cerr<<" *** FAILED - ABORTING "<<std::endl; \
            exit(1); \
        } \
    } while (0)

#define gpuReportDevice() \
    { \
    int device; \
    gpuDeviceProp_t devprop; \
    gpuGetDevice(&device); \
    gpuGetDeviceProperties(&devprop, device); \
    std::cout << "[@" << __func__ << " L" << __LINE__ << "] :" << "Using " << __GPU_API__ << " Device " << device << ": " << devprop.name << std::endl; \
    } \

