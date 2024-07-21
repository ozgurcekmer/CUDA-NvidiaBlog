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
    cout << "Nvidia Blog: Analysis-Driven Optimization (ADO)" << endl;
    cout << left << setw(25) << "Vector Size " << std::right << setw(3) << " : " << L << endl;
    cout << left << setw(25) << "Number of vectors " << std::right << setw(3) << " : " << M << endl;
    cout << left << setw(25) << "Number of vector sets " << std::right << setw(3) << " : " << N << endl << endl;

    // Maximum error evaluator
    MaxError<Real> maximumError;
    
    vector<Real> v(N * M * L, 0.0);
    vector<Real> A(L * L, 0.0);
    vector<Real> yRef(N * L, 0.0);
    vector<Real> yTest(N * L, 0.0);
    
    PrintTensor<Real> printTensor;

    RandomVectorGenerator<Real> randomVector;
    randomVector.randomVector(v);
    randomVector.randomVector(A);
    printTensor.printTensor(v, N, M, L);
    printTensor.printTensor(A, 1, L, L);
    
    WarmupGPU warmupGPU;
    warmupGPU.setup(refGPU, testGPU);
    
    // Reference solver
    cout << "\nSolver: " << refSolverName << endl;
    SolverFactory<Real> refSolverFactory(v, A, yRef);
    std::shared_ptr<ISolver<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
    if (refGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for reference solver: " << refSolverName << endl;
    }
    auto tInit = omp_get_wtime();
    refSolver->solver();
    auto tFin = omp_get_wtime();
    auto runtimeRef = (tFin - tInit) * 1000.0; // in ms
    
    /*
    // Test solver
    cout << "\nSolver: " << testSolverName << endl;
    SolverFactory<Real> testSolverFactory(x, yTest);
    std::shared_ptr<ISolver<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
    if ((!refGPU) && testGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for test solver: " << testSolverName << endl;
    }
    tInit = omp_get_wtime();
    testSolver->solver();
    tFin = omp_get_wtime();
    auto runtimeTest = (tFin - tInit) * 1000.0; // in ms

    cout << "\nVerifying the test code" << endl;
    maximumError.maxError(yRef, yTest);

    cout << "\nRuntimes: " << endl;
    cout << std::setprecision(3) << std::fixed;
    cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
    cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
    cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;
    */
}
