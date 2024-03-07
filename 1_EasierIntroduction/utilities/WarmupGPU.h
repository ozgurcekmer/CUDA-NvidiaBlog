#pragma once

// Utilities
#include "MaxError.h"

// CUDA libs
#include "GpuCommon.h"

// Standard libs
#include <vector>
#include <iostream>

class WarmupGPU
{
private:
	const size_t N = 1 << 26;

public:
	void warmup() const;
};
