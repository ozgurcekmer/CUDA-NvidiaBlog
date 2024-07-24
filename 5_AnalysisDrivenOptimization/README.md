# Analysis-Driven Optimization
- The original NVIDIA developer blog post by [Robert Crovella](https://developer.nvidia.com/blog/author/bob-crovella/) has three parts and the first part can be accessed [here](https://developer.nvidia.com/blog/analysis-driven-optimization-preparing-for-analysis-with-nvidia-nsight-compute-part-1/).
- I am sharing my experience with code development and optimization here.
## Code for analysis
- The code does the following two jobs ***N*** times:
  - Averaging a set of ***M*** vectors each has a size of ***L***
  - Multiplying the average vector by a matrix, which has a size of ***L $\times$ L***.
- I named the initial vector, the matrix, and the output vector as ***v***, ***A***, and ***y***, respectively.
- ***v*** has randomly placed **1**s and **2**s inside. It has been constituted by ***N*** different random vectors. Hence it is an ***L*** by ***N*** matrix.
- A single matrix, ***A***, is used in the entire problem for the matrix - vector multiplication. It's also randomly made by **1**s and **2**s.
- The output is a team of ***N*** vectors with a size of ***L***.
- I developed object-oriented codes for the problem, using ***strategy*** and ***factory method*** design patterns to develop and try many different solvers.
- Only ***SolverFactory.h*** and ***Parameters.h*** is changed to play with the parameters and the new solvers.
- There are two CPU solvers under codes/solvers/, which are ***CpuOriginal***, the original code from [Mr. Bob Crovella's repo](https://github.com/NVIDIA/nsight-training/tree/master/cuda/2020_ncu_smem), and ***CpuSolver***, which I've initially developed.
- The first problem that I encountered was the indexing. I used a different indexing, and then the results seemed different with the author's although they are actually just the transpose of each other.
- Hence, I set up the following indexing to match my results with the original post:

<img src="images/indexing.jpg" alt="Indexing" width="600"/>