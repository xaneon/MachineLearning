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

if __name__ == "__main__":
     pprint(A)
     pprint(B)
     print(shape(A), shape(B))
