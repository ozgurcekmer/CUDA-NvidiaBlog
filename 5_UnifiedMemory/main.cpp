// Standard Libraries
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

// Parameters
#include "Parameters.h"

// Utilities
#include "utilities/include/MaxError.h"
#include "utilities/include/WarmupGPU.h"
#include "utilities/include/RandomVectorGenerator.h"

// Solver factory
#include "solvers/factory/VectorAddFactory.h"
#include "solvers/interface/IVectorAdd.h"

using std::cout;
using std::endl;
using std::vector;
using std::left;
using std::setprecision;
using std::setw;
using std::fixed;

int main()
{
    cout << "Nvidia Blog: Unified Memory for CUDA Beginners" << endl;
    cout << "Vector Size: " << N << endl;

    // Maximum error evaluator
    MaxError<Real> maximumError;

    //RandomVectorGenerator<Real> randomVector;
    
    Vector::managedVector<Real> a(N, 1.0);
    Vector::managedVector<Real> b(N, 2.0);
    Vector::managedVector<Real> cRef(N, 0.0);
    Vector::managedVector<Real> cTest(N, 0.0);
    
    /*
    randomVector.randomVector(a);
    randomVector.randomVector(b);
    */

    WarmupGPU warmupGPU;
    warmupGPU.setup(refGPU, testGPU);
   
    // Reference solver
    cout << "\nSolver: " << refSolverName << endl;
    VectorAddFactory<Real> refSolverFactory(a, b, cRef);
    std::shared_ptr<IVectorAdd<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
    if (refGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for reference solver: " << refSolverName << endl;
    }
    auto tInit = omp_get_wtime();
    refSolver->vectorAdd();
    auto tFin = omp_get_wtime();
    auto runtimeRef = (tFin - tInit) * 1000.0; // in ms

    // Test gridder
    cout << "\nSolver: " << testSolverName << endl;
    VectorAddFactory<Real> testSolverFactory(a, b, cTest);
    std::shared_ptr<IVectorAdd<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
    if ((!refGPU) && testGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for test solver: " << testSolverName << endl;
    }
    tInit = omp_get_wtime();
    testSolver->vectorAdd();
    tFin = omp_get_wtime();
    auto runtimeTest = (tFin - tInit) * 1000.0; // in ms
    
    cout << "\nVerifying the code" << endl;
    maximumError.maxError(cRef, cTest);

    cout << std::setprecision(2) << std::fixed;
    cout << "a[ 0 ]: " << a[0] << ", b[ 0 ]: " << b[0] << ", c[ 0 ]: " << cRef[0] << endl;
    cout << "a[end]: " << a[N - 1] << ", b[end]: " << b[N - 1] << ", c[end]: " << cRef[N - 1] << endl;

    cout << "\nRuntimes: " << endl;
    cout << std::setprecision(2) << std::fixed;
    cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
    cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
    cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;
}
