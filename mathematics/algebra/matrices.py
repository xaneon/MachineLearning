from pprint import pprint

A = [[1, 2, 3],  # 2 rows, 3 columns
     [4, 5, 6]]
B = [[1, 2],  # 3 rows, 2 columns
     [3, 4],
     [5, 6]]

def shape(A):
     num_rows = len(A)
     num_columns = len(A[0]) if A else 0
     return num_rows, num_columns


def get_row(A, i):
     return A[i]  # i-th row


def get_column(A, j):
     return [A_i[j] for A_i in A]  # j-th element of each row


def make_matrix(num_rows, num_cols, entry_fn):
     return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]

def is_diagonal(i, j):
     return 1 if i == j else 0


if __name__ == "__main__":
     pprint(A, width=20)
     pprint(B, width=20)
     print(shape(A), shape(B))
     print(get_row(A, 0))
     print(get_row(A, 1))
     print(get_column(A, 0))
     print(get_column(A, 1))
     pprint(make_matrix(3, 3, lambda a, b: a+b), width=20)
     pprint(make_matrix(4, 4, is_diagonal), width=20)
