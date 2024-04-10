#include "../include/CpuSaxpy.h"

template <typename T>
void CpuSaxpy<T>::saxpy()
{
	const T A = 2.0;
	for (auto i = 0; i < N; ++i)
	{
		y[i] += A * x[i];
	}
}

template void CpuSaxpy<float>::saxpy();
template void CpuSaxpy<double>::saxpy();