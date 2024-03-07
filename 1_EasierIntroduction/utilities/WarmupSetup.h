#pragma once

#include "../Parameters.h"

static bool refGPU = false;
static bool testGPU = false;

void warmupSetup()
{
    if (refSolverName == "gpu" 
        || refSolverName == "gpuOld")
    {
        refGPU = true;
    }
    if (testSolverName == "gpu" 
        || testSolverName == "gpuOld")
    {
        testGPU = true;
    }
}
