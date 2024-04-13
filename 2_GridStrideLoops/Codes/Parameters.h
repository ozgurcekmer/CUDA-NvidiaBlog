#pragma once

/* N:
    1<<24  : 17M
    1<<20  :  1M
    1<<14  : 16k
*/

#include <complex>

typedef float Real;
//static const Real A = 2.0;

static const int N = 1<<26 ; //(17 M)

static const int BLOCK_SIZE = 1024;
//static const size_t GRID_SIZE = 256;
static const int GRID_SIZE = N / BLOCK_SIZE;

// Solver selection
static const std::string refSolverName = "gpuCommon";
static const std::string testSolverName = "gpuGridStride";

/*
    SOLVERS:
    CPU Solvers:
    - cpu 

    GPU Solvers:
    - gpuCommon
    - gpuGridStride

    WARNING: All GPU solvers need to have the letters "gpu"
    (in this order & lower case) in their names
*/

static bool refGPU = false;
static bool testGPU = false;