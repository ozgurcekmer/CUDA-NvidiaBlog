#pragma once

#include <string>

// single/double precision
typedef float Real;
//typedef double Real;

// Solver selection
static const std::string refSolverName = "gpuManual2";
static const std::string testSolverName = "gpuManual4";
/*
	SOLVERS:
    CPU Solvers:
    - NONE -

    GPU Solvers:
    - gpuDefault: default gpu memcpy device to device
    - gpuManual: manual copy from device to device
    - gpuManual2: manual copy from device to device using float2
    - gpuManual4: manual copy from device to device using float4
    
    WARNING: All GPU solvers need to have the letters "gpu"
    (in this order & lower case) in their names
*/

static bool refGPU = false;
static bool testGPU = false;

/* DSIZE:
    1<<25  : 34M
    1<<24  : 17M
    1<<23  :  8M
    1<<22  :  4M
    1<<21  :  2M
    1<<20  :  1M
    1<<14  : 16k
*/

// Vector dimension
const size_t N = 1<<25; 
//const size_t N = 72 * 1024;
//const size_t N = 220 * 1024;

//const int BLOCK_SIZE = 256;
const int BLOCK_SIZE = 1024;
const int GRID_SIZE = static_cast<int>(ceil(static_cast<float>(N) / BLOCK_SIZE));