#include "../include/RandomVectorGenerator.h"

template <typename T>
void RandomVectorGenerator<T>::randomVector(std::vector<T>& v) const
{
    std::random_device rd;
    std::mt19937 randEng(rd());

    //randEng.seed(0);

    std::uniform_real_distribution<T> uniNum{0.0, 1.0};

    for (auto& i : v)
    {
        i = uniNum(randEng);
        if (i < 0.5)
        {
            i = 1.0;
        }
        else
        {
            i = 2.0;
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
