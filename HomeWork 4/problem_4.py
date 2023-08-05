import numpy as np

def func(t):
    return (100/t) * np.sin(10/t)

def func_integral(a,b,n):#composite quadrature
    h = (b-a)/n
    ans = 0
    for i in range(1,int((n/2) + 1)):
        ans = ans + (h/3) * (func(a + (2*i - 2)*h) + 4*func(a + (2*i - 1)*h) + func(a + 2*i*h))
    return ans

def print_table(N,I,RE):
    print("n\t\tI\t\t\t\t\t\tRelative Error")
    for i in range(len(N)):
        print(str(N[i]) + "\t\t" + str(I[i]) + "\t\t" + str(RE[i]))

def cacl_integral(a,b,min_n,max_n):
    N = list()
    I = list()
    RE = list()
    for n in range(min_n,max_n+1,2):
        N.append(n)
        ans = func_integral(a,b,n)
        I.append(ans)
        orig_val = -18.79829683678703
        a_error = np.abs(ans - orig_val)
        r_error = a_error / np.abs(orig_val)
        RE.append(r_error)
    return N,I,RE

def start():
    N,I,RE = cacl_integral(1,3,2,64)
    print_table(N,I,RE)

if __name__ == "__main__":
    start()