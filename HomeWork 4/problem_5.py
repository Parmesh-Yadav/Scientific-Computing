import numpy as np
import matplotlib.pyplot as plt
import math as m

def func(t):
    return m.exp(-m.sin(t**3)/4)

def func_der(t):
    return ((-3/4) * (t**2)) * (m.cos(t**3)) * (m.exp(-m.sin(t**3)/4))

def print_table(H_,C,E_):
    print("Value of H\t\tCalculated Value\t\tAbsolute Error")
    for i in range(len(H_)):
        print(str(H_[i]) + "\t\t\t\t" + str(C[i]) + "\t\t" + str(E_[i]))

def start():
    H = np.zeros(15)
    H_ = np.zeros(15)
    E = np.zeros(15)
    E_ = np.zeros(15)
    C = np.zeros(15)
    for j in range(1,16):
        h = 10 ** (-j)
        calc_val = (func(1+h) - func(1))/h
        C[j-1] = calc_val
        orig_val = func_der(1)
        e_j = np.abs(calc_val - orig_val)
        H[j-1] = np.log10(h)
        H_[j-1] = h
        E[j-1] = np.log10(e_j)
        E_[j-1] = e_j
    print_table(H_,C,E_)
    plt.plot(H,E,color="red",marker="o",linestyle="solid",linewidth=2,markersize=12)
    plt.grid(True)
    plt.xlabel("log10(h)",fontsize=16)
    plt.ylabel("log10(error)",fontsize=16)
    plt.title("Error in forward difference",fontsize=16)
    plt.show()

if __name__ == "__main__":
    start()