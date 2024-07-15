#include "../include/CpuSolver.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void CpuSolver<T>::solver()
{
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP threads." << endl;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (auto i = 0; i < N; ++i)
        {
            T x = static_cast<T>(i);
            T s = sin(x);
            T c = cos(x);
            a[i] += sqrt(s*s + c*c);
        }
    }
}

template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();