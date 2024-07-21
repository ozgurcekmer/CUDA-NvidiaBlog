#include "../include/PrintTensor.h"

using std::cout;
using std::endl;
using std::complex;
using std::vector;

template <typename T>
void PrintTensor<T>::printTensor(const std::vector<T>& v, const size_t N, const size_t M, const size_t L) const
{
    cout << "**************************************" << endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            for (int k = 0; k < L; ++k)
            {
                cout << v[i * (M * L) + j * L + k] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }
    cout << "\n**************************************" << endl;
}

template void PrintTensor<float>::printTensor(const std::vector<float>& v, const size_t N, const size_t M, const size_t L) const;
template void PrintTensor<double>::printTensor(const std::vector<double>& v, const size_t N, const size_t M, const size_t L) const;
template void PrintTensor<int>::printTensor(const std::vector<int>& v, const size_t N, const size_t M, const size_t L) const;