import math
import numpy as np
import matplotlib.pyplot as plt


def print_x(x1, x2, r1, r2):
    print("Solved Using Normal Equation")
    print("Alpha = " + str(x1[0][0]))
    print("Beta = " + str(x1[1][0]))
    print("2-Norm Residual = " + str(r1))
    print()
    print("Solved Using np.linalg.lstsq")
    print("Alpha = " + str(x2[0][0]))
    print("Beta = " + str(x2[1][0]))
    print("2-Norm Residual = " + str(r2))
    print()


def solve_using_lstsq(A, b):
    x = np.linalg.lstsq(A, b, rcond=None)
    return x[0]


def solve_using_normal_equation(A, b):  # AtAx = Atb
    A_t = A.transpose()
    A_t_A = np.dot(A_t, A)
    A_t_b = np.dot(A_t, b)
    x = np.linalg.solve(A_t_A, A_t_b)
    return x


def get_a(t):
    # create a matrix of size len(t) x 2
    A = np.zeros((len(t), 2))
    for i in range(len(t)):
        A[i][0] = -t[i]
        A[i][1] = -1
    return A


def get_b(y):
    # create a matrix of size len(y) x 1
    b = np.zeros((len(y), 1))
    for i in range(len(y)):
        b[i] = math.log10((1/y[i])-1)
    return b


def start():
    t, y = np.loadtxt("hw2_data_ty.txt").T
    b = get_b(y)
    A = get_a(t)
    x1 = solve_using_normal_equation(A, b)
    x2 = solve_using_lstsq(A, b)
    # caclulate the 2 norm residual
    r1 = np.linalg.norm(b - np.dot(A, x1))
    r2 = np.linalg.norm(b - np.dot(A, x2))
    print_x(x1, x2, r1, r2)
    # plot both fitted functions in a single plot
    y1 = np.zeros(len(t))
    for i in range(len(t)):
        y1[i] = (math.exp(x1[0][0]*t[i] + x1[1][0])) / \
            (1 + math.exp(x1[0][0]*t[i] + x1[1][0]))
    plt.plot(t, y1, label="Normal Equation")
    y2 = np.zeros(len(t))
    for i in range(len(t)):
        y2[i] = (math.exp(x2[0][0]*t[i] + x2[1][0])) / \
            (1 + math.exp(x2[0][0]*t[i] + x2[1][0]))
    plt.plot(t, y2, label="np.linalg.lstsq")
    # original data
    plt.plot(t, y, 'bo', ms=2.5, label="Original Data")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    start()
