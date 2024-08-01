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
        for (auto iN = 0; iN < N; ++iN)
        {
            vector<T> vAvg;

            // Vector average
            for (auto iL = 0; iL < L; ++iL)
            {
                T temp = 0.0;
                for (auto iM = 0; iM < M; ++iM)
                {
                    temp += this->v[iN * M * L + iL * M + iM];
                }
                vAvg.push_back(temp / static_cast<T>(M));
            }

            // Matrix - Vector multiplication
            for (auto iL = 0; iL < L; ++iL)
            {
                T temp = 0.0;
                for (auto jL = 0; jL < L; ++jL)
                {
                    temp += this->A[iL * L + jL] * vAvg[jL];
                }
                this->y[iL * N + iN] = temp;
            }
            //printTensor.printTensor(vAvg, 1, 1, L);
        }
    }
    //printTensor.printTensor(this->y, N, 1, L);
}

template void CpuSolver<float>::solver();
template void CpuSolver<double>::solver();