// Standard Libraries
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>

// Parameters
#include "Parameters.h"

// Utilities
#include "utilities/include/MaxError.h"
#include "utilities/include/RandomVectorGenerator.h"
#include "utilities/include/WarmupGPU.h"

// Solver factory
#include "solvers/factory/SaxpyFactory.h"

using std::cout;
using std::endl;
using std::vector;
using std::left;
using std::setprecision;
using std::setw;
using std::fixed;

int main()
{
    // Maximum error evaluator
    MaxError<Real> maximumError;

    RandomVectorGenerator<Real> randomVector;
    
    vector<Real> x(N, 0.0);
    vector<Real> y(N, 0.0);
    vector<Real> yRef(N, 0.0);
    vector<Real> yTest(N, 0.0);
    
    randomVector.randomVector(x);
    randomVector.randomVector(y);
    yRef = y;
    yTest = y;

    WarmupGPU warmupGPU;
    warmupGPU.setup(refGPU, testGPU);

    cout << "RefGPU = " << refGPU << endl;
    cout << "TestGPU = " << testGPU << endl;

    // Reference solver
    cout << "\nSolver: " << refSolverName << endl;
    SaxpyFactory<Real> refSolverFactory(x, yRef);
    std::shared_ptr<ISaxpy<Real>> refSolver = refSolverFactory.getSolver(refSolverName);
    if (refGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for reference solver: " << refSolverName << endl;
    }
    auto tInit = omp_get_wtime();
    refSolver->saxpy();
    auto tFin = omp_get_wtime();
    auto runtimeRef = (tFin - tInit) * 1000.0; // in ms

    // Test solver
    cout << "\nSolver: " << testSolverName << endl;
    SaxpyFactory<Real> testSolverFactory(x, yTest);
    std::shared_ptr<ISaxpy<Real>> testSolver = testSolverFactory.getSolver(testSolverName);
    if ((!refGPU) && testGPU)
    {
        warmupGPU.warmup();
        cout << "Warmup for test solver: " << testSolverName << endl;
    }
    tInit = omp_get_wtime();
    testSolver->saxpy();
    tFin = omp_get_wtime();
    auto runtimeTest = (tFin - tInit) * 1000.0; // in ms
    
    cout << "\nVerifying the code" << endl;
    maximumError.maxError(yRef, yTest);


    cout << "\nRuntimes: " << endl;
    cout << std::setprecision(2) << std::fixed;
    cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
    cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
    cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;
}
