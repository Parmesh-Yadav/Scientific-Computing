import copy
import numpy as np
from scipy.linalg import hilbert

def createX_star():
    x1 = np.ones(10)
    x2 = np.ones(20)
    x3 = np.ones(30)
    x4 = np.ones(40)
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
    print("Condition Number based on the input A =", np.linalg.cond(A_copy))
    print("Error from Pivoted gaussian elimination =", np.linalg.norm(np.subtract(x_copy,xStar)))
    print("Residual from Pivoted gaussian elimination =", np.linalg.norm(np.subtract(b_copy,np.dot(A_copy,x_copy))))
    print("The error from np.linalg.solve =", np.linalg.norm(np.subtract(np.linalg.solve(A_copy,b_copy),xStar)))
    print("The residual from np.linalg.solve =", np.linalg.norm(np.subtract(b_copy,np.dot(A_copy,np.linalg.solve(A_copy,b_copy)))))


def gaussianEliminationWithPivot(A,b,n,xStar):
    A_copy = copy.deepcopy(A)
    b_copy = copy.deepcopy(b)
    x = np.zeros(n)
    s = np.zeros(n)
    l = np.zeros(n, dtype=int)
    for i in range(n):
        l[i] = i
        s_max = 0
        for j in range(n):
            s_max = max(s_max, abs(A[i][j]))
        s[i] = s_max
    for k in range(n-1):
        r_max = 0
        for i in range(k,n):
            r = abs(A[l[i]][k]/s[l[i]])
            if(r > r_max):
                r_max = r
                j = i
        l_temp = l[k]
        l[k] = l[j]
        l[j] = l_temp
        for i in range(k+1,n):
            a_mult = A[l[i]][k]/A[l[k]][k]
            A[l[i]][k] = a_mult
            for j in range(k+1,n):
                A[l[i]][j] = A[l[i]][j] - (a_mult*A[l[k]][j])
    for k in range(n-1):
        for i in range(k+1,n):
            b[l[i]] = b[l[i]] - (A[l[i]][k]*b[l[k]])
    x[n-1] = b[l[n-1]]/A[l[n-1]][n-1]
    for i in range(n-2,-1,-1):
        sum = b[l[i]]
        for j in range(i+1,n):
            sum = sum - (A[l[i]][j]*x[j])
        x[i] = sum/A[l[i]][i]
    x_copy = np.zeros((n,1))
    for i in range(n):
        x_copy[i][0] = x[i]
    printResults(A_copy, b_copy, x_copy,xStar,n)

def start(r1,r2,r3,r4,h1,h2,h3,h4,t1,t2,t3,t4,x1,x2,x3,x4):
    print("Random Matrix Generated\n=======================")
    gaussianEliminationWithPivot(r1,np.dot(r1,x1),10,x1)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(r2,np.dot(r2,x2),20,x2)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(r3,np.dot(r3,x3),30,x3)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(r4,np.dot(r4,x4),40,x4)
    print("Hilbert Matrix Generated\n=======================")
    gaussianEliminationWithPivot(h1,np.dot(h1,x1),10,x1)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(h2,np.dot(h2,x2),20,x2)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(h3,np.dot(h3,x3),30,x3)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(h4,np.dot(h4,x4),40,x4)
    print("Third Example Matrix Generated\n=======================")
    gaussianEliminationWithPivot(t1,np.dot(t1,x1),10,x1)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(t2,np.dot(t2,x2),20,x2)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(t3,np.dot(t3,x3),30,x3)
    print("--------------------------------------------")
    gaussianEliminationWithPivot(t4,np.dot(t4,x4),40,x4)

if __name__ == '__main__':
    r1, r2, r3, r4 = createRandomMatrix()
    h1, h2, h3, h4 = createHilbertMatrix()
    t1, t2, t3, t4 = createThirdMatrix()
    x10, x20, x30, x40 = createX_star()
    start(r1,r2,r3,r4,h1,h2,h3,h4,t1,t2,t3,t4,x10,x20,x30,x40)