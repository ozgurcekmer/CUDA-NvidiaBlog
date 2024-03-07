#pragma once

/*
    N:
    1<<24: 17M
    1<<20:  1M
*/

typedef float Real;

static const int N = 1<<24;

// Solver selection
static const std::string refSolverName = "cpu";
static const std::string testSolverName = "gpu";

/*
SOLVERS:
cpu 
gpu
gridStride
*/