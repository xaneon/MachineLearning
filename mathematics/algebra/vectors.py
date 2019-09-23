from functools import reduce, partial

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


if __name__ == "__main__":
    print(height_weigt_age)
    print(grades)
    print(v)
    print(w)
    g = vector_add(v, w)
    print(vector_add(v, w))
    print(vector_subtract(v, w))
    print(vector_sum([v, w, g]))

