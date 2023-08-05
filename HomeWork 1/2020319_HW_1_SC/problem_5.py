import numpy as np

def tan(x):
    return np.tan(x)

def start():
    for j in range(21):
        x = (np.pi/4) + ((2*np.pi)*(10**j)) 
        print("(x, tan(x)) = (%1.16f, %1.16f)"%(x, tan(x)))

if __name__ == '__main__':
    start()