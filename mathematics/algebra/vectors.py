from functools import reduce, partial
from math import sqrt

height_weigt_age = [180,  # cm
                    72,  # kg
                    40]  # yrs

grades = [95,  # exam 1
          80,  # exam 2
          77,  # exam 3
          99]  # exam 4

v = [1, 3, 9]
w = [4, 5, 6]


def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors):
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result


def vector_sum(vectors):
    return reduce(vector_add, vectors)

vector_sum = partial(reduce, vector_add)


def scalar_multiply(c, v):
    return [c * v_i for v_i in v]


def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v, w):
    return sum([v_i * w_i for v_i, w_i in zip(v, w)])


def sum_of_squares(v):
    return dot(v, v)


def magnitude(v):
    return sqrt(sum_of_squares(v))


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
    return sqrt(squared_distance(v, w))
    # return magnitude(vector_subtract(v, w))


if __name__ == "__main__":
    print(height_weigt_age)
    print(grades)
    print(v)
    print(w)
    g = vector_add(v, w)
    print(vector_add(v, w))
    print(vector_subtract(v, w))
    print(vector_sum([v, w, g]))
    print(scalar_multiply(3, v))
    print(vector_mean([v, w]))
    print(dot(v, w))
    print(sum_of_squares(v))
    print(sum_of_squares(w))
    print(magnitude(v))
    print(magnitude(w))
    print(squared_distance(v, w))
    print(distance(v, w))

