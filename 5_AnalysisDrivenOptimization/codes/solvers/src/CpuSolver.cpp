#include "../include/CpuSolver.h"
#include "../../utilities/include/PrintTensor.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void CpuSolver<T>::solver()
{
    vector<T> vAvg(N * L, 0.0);
    PrintTensor<T> printTensor;
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP threads." << endl;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (auto i = 0; i < N; ++i)
        {
            // Vector average
            for (auto k = 0; k < L; ++k)
            {
                T temp = 0.0;
                for (auto j = 0; j < M; ++j)
                {
                    temp += v[i * (M * L) + j * L + k];
                }
                vAvg[i * L + k] = temp / static_cast<T>(M);
            }
            
            // Matrix - Vector multiplication
            for (auto k = 0; k < L; ++k)
            {
                T temp = 0.0;
                for (auto j = 0; j < L; ++j)
                {
                    temp += A[k * L + j] * vAvg[i * L + j];
                }
                y[i * L + k] = temp;
            }
        }
    }
    printTensor.printTensor(vAvg, N, 1, L);
    printTensor.printTensor(y, N, 1, L);


}

template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();