import numpy as np

def shifted_inverse_iteration(A, x, sigma):
    print("The shifted inverse iteration is: ")
    for i in range(100):
        x = np.linalg.inv(A - sigma*np.identity(3)).dot(x)
        e_v = np.linalg.norm(x)
        x = x/e_v
    print("The smallest eigenvalue is:\n", e_v)
    print("The smallest eigenvector is:\n", x)
    return e_v, x

def start():
    x = np.array([[0],[0],[1]]) #starting vector
    A = np.array([[6,2,1], [2,3,1], [1,1,1]]) #matrix
    print("The matrix is:\n", A)
    print("The starting vector is:\n", x)
    print("-------------------------")
    e_v, x = shifted_inverse_iteration(A, x, 5)
    print("-------------------------")
    print("Using real eigensystem library routine:")
    print(np.linalg.eigh(A))

if __name__ == '__main__':
    start()