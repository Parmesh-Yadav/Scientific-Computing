import numpy as np
import scipy as sc

def get_a(k):
    return np.array([[1,1],[10**(-k),0],[0,10**(-k)]])

def get_b(k):
    return np.array([[-10**(-k)],[1 + 10**(-k)],[1 - 10**(-k)]])

def get_qr(A):
    return np.linalg.qr(A)

def solve_triangular(A, b):
    return sc.linalg.solve_triangular(A, b)

def start():
    for k in range(6,16):
        print("k = " + str(k))
        A = get_a(k)
        # print("A = " + str(A))
        b = get_b(k)
        # print("b = " + str(b))
        Q, R = get_qr(A)
        # print("Q = " + str(Q))
        # print("R = " + str(R))
        x = solve_triangular(R, np.dot(Q.transpose(), b))
        print("x = " + str(x))
        print()

if __name__ == "__main__":
    start()