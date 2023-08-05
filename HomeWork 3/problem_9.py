import numpy as np
import scipy as sp

def modified_qr_iteration(A):
    print('The modified QR iteration is: ')
    for i in range(100):
        sigma = A[2,2]
        Q, R = sp.linalg.qr(A - sigma*np.identity(3))
        A = R.dot(Q) + sigma*np.identity(3)
    print("The eigenvalues are:\n", np.diag(A))
    print("The eigenvectors are:\n", Q)
    return np.diag(A), Q

def start():
    print("Problem 6 matrix")
    x = np.array([[0],[0],[1]]) #starting vector
    A = np.array([[2,3,2], [10,3,4], [3,6,1]]) #matrix
    print("The matrix is:\n", A)
    print("The starting vector is:\n", x)
    print("-------------------------")
    e_v, x = modified_qr_iteration(A)
    print("-------------------------")
    print("Using real eigensystem library routine:")
    print(np.linalg.eig(A))
    print("-------------------------")
    print("Problem 7 matrix")
    x = np.array([[0],[0],[1]]) #starting vector
    A = np.array([[6,2,1], [2,3,1], [1,1,1]]) #matrix
    print("The matrix is:\n", A)
    print("The starting vector is:\n", x)
    print("-------------------------")
    e_v, x = modified_qr_iteration(A)
    print("-------------------------")
    print("Using real eigensystem library routine:")
    print(np.linalg.eigh(A))

if __name__ == '__main__':
    start()
