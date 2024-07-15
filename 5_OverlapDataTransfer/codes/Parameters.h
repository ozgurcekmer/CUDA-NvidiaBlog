#pragma once

#include <complex>
#include <cmath>
#include <vector>

typedef float Real;

/* VECTOR SIZE:
    1<<25  : 34M
    1<<24  : 17M
    1<<23  :  8M
    1<<22  :  4M
    1<<21  :  2M
    1<<20  :  1M
    1<<14  : 16k
*/

// Vector dimension
const size_t N = 1 << 25;
//const size_t N = 72 * 1024;

// Kernel launch parameters
static const size_t BLOCK_SIZE = 256;
static const size_t GRID_SIZE = static_cast<size_t>(std::ceil(static_cast<float>(N) / BLOCK_SIZE));

// CUDA streams parameters
static const int N_STREAMS = 4;

// Solver selection
static const std::string refSolverName = "gpuVersion2";
static const std::string testSolverName = "gpuV2MaxOcc";
/*
    SOLVERS:
    CPU Solvers:
    - cpu: a CPU solver using OpenMP threads
  
    GPU Solvers:
    - gpuSequential: Classic data transfer
    - gpuSeqMaxOcc: Using a grid-stride loop - targeting max thread occupancy
    - gpuVersion1: Loop over all the operations for each chunk of data
    - gpuVersion2: Batch similar operations together
    - gpuV1MaxOcc: Modified Version 1 - Using a grid-stride loop - targeting max thread occupancy
    - gpuV2MaxOcc: Modified Version 2 - Using a grid-stride loop - targeting max thread occupancy

    WARNING: All GPU solvers need to have the letters "gpu"
    (in this order & lower case) in their names
*/

static bool refGPU = false;
static bool testGPU = false;
