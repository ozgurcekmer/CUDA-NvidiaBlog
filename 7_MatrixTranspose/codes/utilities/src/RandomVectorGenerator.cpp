#include "../include/RandomVectorGenerator.h"
#include <iostream>

using std::cout;
using std::endl;

template <typename T>
void RandomVectorGenerator<T>::randomVector(std::vector<T>& v) const
{
    std::random_device rd;
    std::mt19937 randEng(rd());

    //randEng.seed(0);

    std::uniform_real_distribution<T> uniNum{0.0, 1.0};

    int nThreads = omp_get_max_threads();
    //int nThreads = 1;
    omp_set_num_threads(nThreads);
    cout << "Working with " << nThreads << " OpenMP thread(s)." << endl;
#pragma omp parallel
    {
        #pragma omp for schedule(static) 
        for (auto i = 0; i < v.size(); ++i)
        {
            v[i] = uniNum(randEng);
        }
    }
}

template<typename T>
void RandomVectorGenerator<T>::randomVector(Vector::pinnedVector<T>& v) const
{
    std::random_device rd;
    std::mt19937 randEng(rd());

    std::uniform_real_distribution<T> uniNum{ 0.0, 1.0 };

    for (auto& i : v)
    {
        i = uniNum(randEng);
    }
}

template void RandomVectorGenerator<float>::randomVector(std::vector<float>& v) const;
template void RandomVectorGenerator<double>::randomVector(std::vector<double>& v) const;
template void RandomVectorGenerator<float>::randomVector(Vector::pinnedVector<float>& v) const;
template void RandomVectorGenerator<double>::randomVector(Vector::pinnedVector<double>& v) const;
