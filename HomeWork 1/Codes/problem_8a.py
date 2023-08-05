import numpy as np
from scipy.linalg import hilbert
import copy

def createX_star():
    x1 = np.ones((10,1))
    x2 = np.ones((20,1))
    x3 = np.ones((30,1))
    x4 = np.ones((40,1))
    return x1, x2, x3, x4

def randomMatrix(n):
    return np.random.random_sample((n,n))

def createRandomMatrix():
    r1 = randomMatrix(10)
    r2 = randomMatrix(20)
    r3 = randomMatrix(30)
    r4 = randomMatrix(40)
    return r1, r2, r3, r4

def hilbertMatrix(n):
    return hilbert(n)

def createHilbertMatrix():
    h1 = hilbertMatrix(10)
    h2 = hilbertMatrix(20)
    h3 = hilbertMatrix(30)
    h4 = hilbertMatrix(40)
    return h1, h2, h3, h4

def thirdMatrix(n):
    M = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            if(i>j):
                M[i][j] = -1
    return M

def createThirdMatrix():
    t1 = thirdMatrix(10)
    t2 = thirdMatrix(20)
    t3 = thirdMatrix(30)
    t4 = thirdMatrix(40)
    return t1, t2, t3, t4

def printResults(A_copy, b_copy, x_copy,xStar,n):
    print("Matrix of size n =",n)
    print("--------------------------------------------")
    print("Condition Number based on the input A =", np.linalg.cond(A_copy))
    print("Error from unpivoted gaussian elimination =", np.linalg.norm(np.subtract(x_copy,xStar)))
    print("Residual from unpivoted gaussian elimination =", np.linalg.norm(np.subtract(b_copy,np.dot(A_copy,x_copy))))
    print("The error from np.linalg.solve =", np.linalg.norm(np.subtract(np.linalg.solve(A_copy,b_copy),xStar)))
    print("The residual from np.linalg.solve =", np.linalg.norm(np.subtract(b_copy,np.dot(A_copy,np.linalg.solve(A_copy,b_copy)))))

def gaussianEliminationWithoutPivot(A,b,n,xStar):
    A_copy = copy.deepcopy(A)
    b_copy = copy.deepcopy(b)
    x = np.zeros(n)
    for k in range(n-1):
        for i in range(k+1,n):
            a_temp = A[i][k]/A[k][k]
            A[i][k] = a_temp
            for j in range(k+1,n):
                A[i][j] = A[i][j] - (a_temp*A[k][j])
            b[i] = b[i] - (a_temp*b[k])
    x[n-1] = b[n-1]/A[n-1][n-1]
    for i in range(n-2,-1,-1):
        sum = b[i]
        for j in range(i+1,n):
            sum = sum - (A[i][j]*x[j])
        x[i] = sum/A[i][i]
    x_copy = np.zeros((n,1))
    for i in range(n):
        x_copy[i][0] = x[i]
    printResults(A_copy, b_copy, x_copy,xStar,n)

def start(r1,r2,r3,r4,h1,h2,h3,h4,t1,t2,t3,t4,x1,x2,x3,x4):
    print("Random Matrix Generated\n=============================================")
    gaussianEliminationWithoutPivot(r1,np.dot(r1,x1),10,x1)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(r2,np.dot(r2,x2),20,x2)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(r3,np.dot(r3,x3),30,x3)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(r4,np.dot(r4,x4),40,x4)
    print("Hilbert Matrix Generated\n=============================================")
    gaussianEliminationWithoutPivot(h1,np.dot(h1,x1),10,x1)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(h2,np.dot(h2,x2),20,x2)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(h3,np.dot(h3,x3),30,x3)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(h4,np.dot(h4,x4),40,x4)
    print("Third Example Matrix Generated\n=============================================")
    gaussianEliminationWithoutPivot(t1,np.dot(t1,x1),10,x1)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(t2,np.dot(t2,x2),20,x2)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(t3,np.dot(t3,x3),30,x3)
    print("--------------------------------------------")
    gaussianEliminationWithoutPivot(t4,np.dot(t4,x4),40,x4)

if __name__ == '__main__':
    r1, r2, r3, r4 = createRandomMatrix()
    h1, h2, h3, h4 = createHilbertMatrix()
    t1, t2, t3, t4 = createThirdMatrix()
    x10, x20, x30, x40 = createX_star()
    start(r1,r2,r3,r4,h1,h2,h3,h4,t1,t2,t3,t4,x10,x20,x30,x40)
