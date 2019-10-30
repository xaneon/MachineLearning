import numpy as np
from pprint import pprint
np.random.seed(42)

A = np.random.randint(0, 2**8, (4, 3))
b = np.random.randint(0, 2**8, (3, 1))
I = np.identity(3)
n, m = A.shape
r = np.linalg.matrix_rank(A)
# x = np.linalg.solve(A, b)  # A must be square (n x n matrix)
# A^T A x_hat = A^T b
x_hat = np.linalg.solve(np.dot(A.T, A), b)
# in this case solving is possible


pprint(A)
print(n, m, r)
pprint(I)
print(np.dot(np.dot(A.T, A), x_hat), b)
