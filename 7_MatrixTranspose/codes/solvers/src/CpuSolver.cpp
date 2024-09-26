#include "../include/CpuSolver.h"
#include "../../utilities/include/PrintTensor.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void CpuSolver<T>::solver()
{
    //PrintTensor<T> printTensor;
    int nThreads = omp_get_max_threads();
    //int nThreads = 1;
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP thread(s)." << endl;
#pragma omp parallel
    {
        //vector<T> vAvg;
#pragma omp for schedule(static) 
//#pragma omp parallel for schedule(static) 
        for (auto i = 0; i < N; ++i)
        {
            for (auto j = 0; j < N; ++j)
            {
                y[j * N + i] = v[i * N + j];
            }
        }
    }
    //printTensor.printTensor(this->y, N, 1, L);
}

template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();