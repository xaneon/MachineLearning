import numpy as np
import pandas as pd
# 3 examples (Ex1-3) of Matching
# Ex 1: Data integration - Data from multiple sources

# Ex 2: enrichment, enhance dataset

# Ex 3: Record linkage (e.g. medical and financial research)

A = np.arange(1, 10).reshape((3, 3))
B = A * np.random.randint(1, 3, A.shape)  # print(A, B)
dfA = pd.DataFrame(data=A, columns=["A", "B", "C"])
dfB = pd.DataFrame(data=B, columns=["A", "B", "C"])
print(dfA, dfB)

# 1. Pairwise Matching
# no duplicates, all possible pairs are A x B
print(np.cross(A, B))

M, U = list(), list()
for rowa, rowb in zip(A, B):
    for a, b in zip(rowa, rowb):
        if a == b:
            M.append((a, b))
        else:
            U.append((a, b))
print(M, U)
AxB = set(M).union(U)
print("all pairs: ", AxB)

