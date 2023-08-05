import numpy as np

def normalized_power_iteration(A, x):#find the largest eigenvalue and eigenvector
    print("The normalized power iteration is: ")
    for i in range(100):
        x = A.dot(x)
        e_v = np.linalg.norm(x)
        x = x/e_v
    print("The largest eigenvalue is:\n", e_v)
    print("The largest eigenvector is:\n", x)
    return e_v, x

def inverse_iteration(A, x):#find the smallest eigenvalue and eigenvector
    print("The inverse iteration is: ")
    for i in range(100):
        x = np.linalg.inv(A).dot(x)
        e_v = np.linalg.norm(x)
        x = x/e_v
    print("The smallest eigenvalue is:\n", e_v)
    print("The smallest eigenvector is:\n", x)
    return e_v, x
        
def start():
    x = np.array([[0],[0],[1]]) #starting vector
    A = np.array([[2,3,2], [10,3,4], [3,6,1]]) #matrix
    print("The matrix is:\n", A)
    print("The starting vector is:\n", x)
    print("-------------------------")
    e_v, x = normalized_power_iteration(A, x)
    print("-------------------------")
    e_v, x = inverse_iteration(A, x)
    print("Using real eigensystem library routine:")
    print(np.linalg.eig(A))
    

if __name__ == '__main__':
    start()