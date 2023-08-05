import numpy as np


def get_a(k):
    return np.array([[1, 1], [10**(-k), 0], [0, 10**(-k)]])


def get_b(k):
    return np.array([[-10**(-k)], [1 + 10**(-k)], [1 - 10**(-k)]])


def start():
    for k in range(6, 16):
        print("k = " + str(k))
        A = get_a(k)
        # print("A = " + str(A))
        b = get_b(k)
        # print("b = " + str(b))
        A_t = A.transpose()
        # print("A_t = " + str(A_t))
        A_t_A = np.dot(A_t, A)
        # print("A_t_A = " + str(A_t_A))
        A_t_b = np.dot(A_t, b)
        # print("A_t_b = " + str(A_t_b))
        x = np.linalg.solve(A_t_A, A_t_b)
        print("x = " + str(x))
        print()


if __name__ == "__main__":
    start()
