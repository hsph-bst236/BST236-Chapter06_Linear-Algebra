# Sparse Matrix Solver Comparison

This project compares the performance of different sparse matrix solvers:

1. Naive solve (using NumPy's dense solver)
2. Sparse direct solve (using SciPy's spsolve)
3. GMRES (Generalized Minimal Residual Method)
4. BiCG (Biconjugate Gradient Method)

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- NetworkX (for graph examples)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Examples

This repository contains three different examples of sparse matrix solvers:

### 1. Random Sparse Matrices

```bash
python sparse_matrix_comparison.py
```

This script:
- Creates three random sparse matrices of different sizes (1000×1000, 5000×5000, and 10000×10000)
- Solves a linear system Ax = b using each solver
- Measures and compares the execution time
- Generates a plot showing the performance comparison
- Prints a summary table of the results

### 2. 2D Poisson Equation

```bash
python sparse_matrix_pde.py
```

This script:
- Creates sparse matrices representing the 2D Poisson equation on grids of different sizes
- Solves the resulting linear systems using different solvers
- Visualizes the solution for the smallest grid
- Compares the performance of the solvers
- Generates a plot showing the performance comparison

### 3. Graph Laplacian

```bash
python sparse_matrix_graph.py
```

This script:
- Creates sparse matrices representing graph Laplacians for random graphs of different sizes
- Visualizes the structure of the smallest graph
- Solves linear systems involving the graph Laplacian using different solvers
- Compares the performance of the solvers
- Generates a plot showing the performance comparison

## Output

Each script will:
- Print detailed timing information for each solver and matrix size
- Generate a summary table
- Create plots saved as PNG files

## Notes

- The naive solver is only used for smaller matrices to avoid memory issues
- All matrices are created in CSR (Compressed Sparse Row) format, which is efficient for matrix-vector products
- The iterative solvers (GMRES and BiCG) may have convergence issues for ill-conditioned matrices
- The graph Laplacian example adds a small regularization term to make the matrix invertible 