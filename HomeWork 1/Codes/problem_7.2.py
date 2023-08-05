import numpy as np

def eulers(n):
    k1 = 0
    for k in range(1, n+1):
        k1 += 1/k
    return k1 - np.log(n+(1/2))

def start():
    for n in range(1,5001):
        if(n%100 == 0):
            print("n = %d, e = %1.16f"%(n, eulers(n)))

if __name__ == '__main__':
    start()