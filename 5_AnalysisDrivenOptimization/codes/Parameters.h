#pragma once

#include <complex>
#include <cmath>
#include <vector>


/* VECTOR SIZE:
    1<<26  : 68M
    1<<25  : 34M
    1<<24  : 17M
    1<<23  :  8M
    1<<22  :  4M
    1<<21  :  2M
    1<<20  :  1M
    1<<14  : 16k
*/

// Vector dimension
const size_t L = 1024;
//const size_t L = 256;
//const size_t L = 5;

// Number of vectors in a set
const size_t M = 1024;
//const size_t M = 32;
//const size_t M = 5;

// Number of vector sets
const size_t N = 1024;
//const size_t N = 32;
//const size_t N = 5;

/*
// CUDA streams parameters
static const int NUM_STREAMS = 16;
const int CHUNKS = 64;
const size_t N = 1024 * 1024 * CHUNKS;
*/

// Kernel launch parameters
static const size_t BLOCK_SIZE = L;
//static const size_t GRID_SIZE = static_cast<size_t>(std::ceil(static_cast<float>(N) / BLOCK_SIZE));
static const size_t GRID_SIZE = 1;

// Other parameters
typedef float Real;

// Solver selection
//static const std::string refSolverName = "cpuOriginal";
static const std::string refSolverName = "cpu";
static const std::string testSolverName = "gpuSolver2";
/*
    SOLVERS:
    CPU Solvers:
    - cpu: a serial CPU solver
    - cpu2: a CPU solver using OpenMP threads
    - cpu3: uses a 1D average vector

    GPU Solvers:
    - gpuSequential: Classic data transfer

    WARNING: All GPU solvers need to have the letters "gpu"
    (in this order & lower case) in their names
*/

static bool refGPU = false;
static bool testGPU = false;