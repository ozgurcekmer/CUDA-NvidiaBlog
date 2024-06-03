// Standard Libraries
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

// Parameters
#include "Parameters.h"

// Utilities
#include "utilities/include/PrintVector.h"
#include "utilities/include/RandomVectorGenerator.h"
#include "utilities/include/WarmupGPU.h"
#include "utilities/include/MaxError.h"

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
	cout << "Exercise: Device to device copy" << endl;
	cout << "Vector Size: " << N << endl;
	
	// Print vector object
//	PrintVector<Real> printVector;

	// Max Error class
	MaxError<Real> maxError;

	// Vector A - A random vector
	vector<Real> A(N);
	RandomVectorGenerator<Real> randomVector;
	randomVector.randomVector(A);

	// Resulting reference & test vectors
	vector<Real> refB(N, 0.0);
	vector<Real> testB(N, 0.0);

	WarmupGPU warmupGPU;
	warmupGPU.setup(refGPU, testGPU);

	cout << "RefGPU = " << refGPU << endl;
	cout << "TestGPU = " << testGPU << endl;

	// Reference Solver
	cout << "\nSolver: " << refSolverName << endl;
	SolverFactory<Real> refSolverFactory(A, refB);
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
	// Validate the ref result
	cout << "Validating reference solution" << endl;
	maxError.maxError(A, refB);

	// Test Solver
	cout << "\nSolver: " << testSolverName << endl;
	SolverFactory<Real> testSolverFactory(A, testB);
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
	// Validate the test result
	cout << "Validating test solution" << endl;
	maxError.maxError(A, testB);

	// Testing purposes
	/*
	printVector.printVector(A);
	printVector.printVector(refSum);
	printVector.printVector(testSum);
	*/

	cout << "\nRuntimes: " << endl;
	cout << std::setprecision(2) << std::fixed;
	cout << std::left << std::setw(20) << refSolverName << ": " << runtimeRef << " ms." << endl;
	cout << std::left << std::setw(20) << testSolverName << ": " << runtimeTest << " ms." << endl;
	cout << std::left << std::setw(20) << "Speedup" << ": " << runtimeRef / runtimeTest << endl;

}