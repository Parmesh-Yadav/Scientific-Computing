import numpy as np
import random

def rayleigh_quotient_iteration(A, x, l_m_e):
    error_p = 10
    print('The Rayleigh quotient iteration is: ')
    for i in range(30):
        sigma = x.T.dot(A).dot(x)/x.T.dot(x)
        x = np.linalg.inv(A - sigma*np.identity(3)).dot(x)
        e_v = np.linalg.norm(x)
        x = x/e_v
        #compute the rate of convergence of the Rayleigh quotient iteration ot the largest magnitude eigenvalue
        error = np.abs(l_m_e - sigma)
        rate = np.log(error)/np.log(error_p)
        print("The rate of convergence is:\n", rate)
    print("The closest eigenvalue to the rayleigh quotient is:\n", e_v)
    print("The closest eigenvector to the rayleigh quotient is:\n", x)
    return e_v, x

def start():
    x = np.array([[random.uniform(0,1)],[random.uniform(0,1)],[random.uniform(0,1)]]) #starting vector
    A = np.array([[2,3,2], [10,3,4], [3,6,1]]) #matrix
    print("The matrix is:\n", A)
    print("The starting vector is:\n", x)
    print("-------------------------")
    #largest magnitude eigenvalue of A
    l_m_e = np.linalg.eigh(A)[0][np.argmax(np.abs(np.linalg.eigh(A)[0]))]
    e_v, x = rayleigh_quotient_iteration(A, x,l_m_e)
    print("-------------------------")
    print("Using real eigensystem library routine:")
    print(np.linalg.eigh(A))
    # print("The largest magnitude eigenvalue is:\n", l_m_e)

if __name__ == '__main__':
    start()