#include "../include/CpuOriginal.h"
#include "../../utilities/include/PrintTensor.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void CpuOriginal<T>::solver()
{
    //PrintTensor<T> printTensor;
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP threads." << endl;
    #pragma omp parallel for schedule(static) 
    for (auto k = 0; k < N; ++k)    
    {
        vector<T> v1(L);

        // Vector average    
        for (auto i = 0; i < M; ++i)    
        {
            for (auto j = 0; j < L; ++j)
            {
                v1[j] += this->v[k * M * L + j * M + i];
            }
        }
        for (auto j = 0; j < L; ++j)
        {
            v1[j] /= M;
        }

        // Matrix - Vector multiplication
        for (auto i = 0; i < L; ++i)
        {
            for (auto j = 0; j < L; ++j)
            {
                this->y[i * N + k] += this->A[i * L + j] * v1[j];
            }
                 
        }
        //printTensor.printTensor(v1, 1, 1, L);
    }
    
    //printTensor.printTensor(this->y, N, 1, L);
}

template void CpuOriginal<float>::solver();
template void CpuOriginal<double>::solver();