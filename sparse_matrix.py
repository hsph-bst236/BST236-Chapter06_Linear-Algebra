import numpy as np
from scipy import sparse, linalg
import time


def time_operation(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    return result, elapsed

# Create sparse matrices


def create_sparse_matrices(n=1000):
    # Create a sparse matrix with diagonal and off-diagonal elements
    diagonals = [np.ones(n), -2*np.ones(n), np.ones(n)]
    positions = [-1, 0, 1]
    A_sparse = sparse.spdiags(diagonals, positions, n, n, format='csr')

    # Create dense version for comparison
    A_dense = A_sparse.toarray()

    # Create a random right-hand side
    b = np.random.rand(n)

    return A_sparse, A_dense, b


# Demonstration
n = 1000
A_sparse, A_dense, b = create_sparse_matrices(n)

# 1. Compare solving linear equations
print("Solving linear equations:")
# Dense solver
x_dense, dense_time = time_operation(linalg.solve, A_dense, b)
print(f"Dense solve time: {dense_time:.6f} seconds")

# Sparse solver
x_sparse, sparse_time = time_operation(linalg.spsolve, A_sparse, b)
print(f"Sparse solve time: {sparse_time:.6f} seconds")

# 2. Compare eigenvalue computations
print("\nComputing eigenvalues:")
# Dense eigenvalues (compute only largest k)
k = 5
eigvals_dense, dense_eig_time = time_operation(
    linalg.eigs, A_sparse, k=k, which='LM', return_eigenvectors=False
)
print(f"Dense eigenvalues time (k={k}): {dense_eig_time:.6f} seconds")

# Sparse eigenvalues
eigvals_sparse, sparse_eig_time = time_operation(
    linalg.eigsh, A_sparse, k=k, which='LM', return_eigenvectors=False
)
print(f"Sparse eigenvalues time (k={k}): {sparse_eig_time:.6f} seconds")

# 3. Sparse matrix operations
print("\nSparse matrix operations:")
B_sparse = sparse.rand(n, n, density=0.01)

# Matrix multiplication
mult_sparse, sparse_mult_time = time_operation(
    A_sparse.dot, B_sparse
)
print(f"Sparse multiplication time: {sparse_mult_time:.6f} seconds")

# Dense equivalent
B_dense = B_sparse.toarray()
mult_dense, dense_mult_time = time_operation(
    np.dot, A_dense, B_dense
)
print(f"Dense multiplication time: {dense_mult_time:.6f} seconds")

# Memory usage comparison
sparse_size = A_sparse.data.nbytes + \
    A_sparse.indptr.nbytes + A_sparse.indices.nbytes
dense_size = A_dense.nbytes
print(f"\nMemory usage:")
print(f"Sparse matrix: {sparse_size/1e6:.2f} MB")
print(f"Dense matrix: {dense_size/1e6:.2f} MB")
