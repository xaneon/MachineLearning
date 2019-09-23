from pprint import pprint

A = [[1, 2, 3],  # 2 rows, 3 columns
     [4, 5, 6]]
B = [[1, 2],  # 3 rows, 2 columns
     [3, 4],
     [5, 6]]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
user = {user for user, _ in friendships} | {user for _, user in friendships}

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

def zeros(i, j):
     return 0


def apply_coordinates(M, coordinates):
     for i, j in coordinates:
         M[i][j] = 1
         M[j][i] = 1
     return M


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
     pprint(make_matrix(len(user), len(user), is_diagonal))
     pprint(user)
     pprint(apply_coordinates(make_matrix(len(user), len(user), zeros),
                              friendships))
