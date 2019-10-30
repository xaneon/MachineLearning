import numpy as np

def avg(x):
   return sum(x) / len(x)

def describe_matrix(data: np.ndarray) -> str:
   n, m = data.shape
   r = np.linalg.matrix_rank(data)
   dtype = data.dtype

   res = (f"| Parameter: | Value | \n" +
          f"| -- | -- | \n" +
          f"| num rows $n$: | {n} | \n" +
          f"| num cols $m$: | {m} | \n" +
          f"| dim $N(A)$: | {m - r} | \n" +
          f"| dim $C(A)$ (=rank $r$): | {r} | \n" +
          f"| Data type: | {dtype} | \n")
   return res


def testavg():
   assert avg([1, 3, 5]) == 3.0
