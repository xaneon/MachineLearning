import numpy as np

def avg(x):
   return sum(x) / len(x)

def describe_matrix(data: np.ndarray) -> str:
   n, m = data.shape
   r = np.linalg.matrix_rank(data)
   dtype = data.dtype
   res = (f"# rows n: \t\t\t{n}\n" +
          f"# cols m: \t\t\t{m}\n" +
          f"dim N(A): \t\t\t{m - r}\n" +
          f"dim C(A) (=rank r): \t\t{r}\n" +
          f"Data type: \t\t\t{dtype}\n")
   return res


def testavg():
   assert avg([1, 3, 5]) == 3.0
