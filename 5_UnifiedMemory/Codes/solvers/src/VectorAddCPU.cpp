#include "../include/VectorAddCPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void VectorAddCPU<T>::vectorAdd()
{
    //int tID;
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP threads." << endl;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (auto i = 0; i < N; ++i)
        {
            c[i] = a[i] + b[i];
        }
    }
}

template void VectorAddCPU<float>::vectorAdd();
template void VectorAddCPU<double>::vectorAdd();