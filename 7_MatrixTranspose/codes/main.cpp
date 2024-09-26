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
#include "utilities/include/PrintTensor.h"

// Solver factory
#include "solvers/factory/SolverFactory.h"
#include "solvers/interface/ISolver.h"

using std::cout;
using std::endl;
using std::vector;
using std::left;
using std::setprecision;
using std::setw;
using std::fixed;

int main()
{
    cout << "Nvidia Blog: Matrix Transpose" << endl;
    cout << left << setw(25) << "Matrix Size " << std::right << setw(3) << " : " << N << " x " << N << endl;

    // Maximum error evaluator
    MaxError<Real> maximumError;
    
    vector<Real> A(N * N, 0.0);
    vector<Real> yRef(N * N, 0.0);
    vector<Real> yTest(N * N, 0.0);
    
    PrintTensor<Real> printTensor;
    RandomVectorGenerator<Real> randomVector;
    randomVector.randomVector(A);
    //printTensor.printTensor(A, 1, N, N);
    
#ifdef WARMUP
    WarmupGPU warmupGPU;
    warmupGPU.setup(refGPU, testGPU);
#endif

    // Reference solver
    cout << "\nSolver: " << refSolverName << endl;
    SolverFactory<Real> refSolverFactory(A, yRef);
    std::shared_ptr<ISolver<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
#ifdef WARMUP
    if (refGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for reference solver: " << refSolverName << endl;
    }
#endif
    auto tInit = omp_get_wtime();
    refSolver->solver();
    auto tFin = omp_get_wtime();
    auto runtimeRef = (tFin - tInit) * 1000.0; // in ms
    
    // Test solver
    cout << "\nSolver: " << testSolverName << endl;
    SolverFactory<Real> testSolverFactory(A, yTest);
    std::shared_ptr<ISolver<Real>> testSolver = testSolverFactory.getSolver(testSolverName);  
#ifdef WARMUP
    if ((!refGPU) && testGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for test solver: " << testSolverName << endl;
    }
#endif
    tInit = omp_get_wtime();
    testSolver->solver();
    tFin = omp_get_wtime();
    auto runtimeTest = (tFin - tInit) * 1000.0; // in ms
    
    cout << "\nVerifying the test code" << endl;
    maximumError.maxError(yRef, yTest);

    //printTensor.printTensor(yRef, 1, N, N);
    //printTensor.printTensor(yTest, 1, N, N);

    cout << "\nRuntimes: " << endl;
    cout << std::setprecision(3) << std::fixed;
    cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
    cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
    cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;
    
}
