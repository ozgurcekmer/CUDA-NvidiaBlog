#include "../include/CpuSolver3.h"
#include "../../utilities/include/PrintTensor.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void CpuSolver3<T>::solver()
{
    PrintTensor<T> printTensor;
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP threads." << endl;
    #pragma omp parallel
    {
        //vector<T> vAvg;
        #pragma omp for schedule(static) 
        for (auto i = 0; i < N; ++i)
        {
            vector<T> vAvg;
            
            // Vector average
            for (auto k = 0; k < L; ++k)
            {
                T temp = 0.0;
                for (auto j = 0; j < M; ++j)
                {
                    temp += this->v[i * (M * L) + j * L + k];
                }
                vAvg.push_back(temp / static_cast<T>(M));
            }
            
            // Matrix - Vector multiplication
            for (auto k = 0; k < L; ++k)
            {
                T temp = 0.0;
                for (auto j = 0; j < L; ++j)
                {
                    temp += this->A[k * L + j] * vAvg[j];
                }
                this->y[i * L + k] = temp;
            }
            
            //vAvg.clear();
        }
    }
    //printTensor.printTensor(vAvg, N, 1, L);
    //printTensor.printTensor(this->y, N, 1, L);
}

template void CpuSolver3<float>::solver();
template void CpuSolver3<double>::solver();