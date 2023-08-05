import numpy as np
import matplotlib.pyplot as plt


def D_matrix():
    d = list()
    for i in range(999):
        r = list()
        for j in range(1000):
            if i == j:
                r.append(-1)
            elif i == j-1:
                r.append(1)
            else:
                r.append(0)
        d.append(r)
    return np.array(d)


def plot_denoise(x_noisy, L):
    L = "Signal with Lambda = " + str(L)
    plt.plot(np.arange(1, len(x_noisy)+1), x_noisy, label=L)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc="best")
    plt.gcf().tight_layout()


def LLS_denoise(x_noisy):
    identity_Matrix = np.identity(x_noisy.shape[0])
    D_Matrix = D_matrix()
    D_T_Matrix = D_Matrix.T
    D_T_D_Matrix = np.dot(D_T_Matrix, D_Matrix)
    for lam in [1, 100, 10000]:
        A = identity_Matrix + lam * D_T_D_Matrix
        b = x_noisy
        x = np.linalg.solve(A, b)  # Ax = b
        plot_denoise(x, lam)


if __name__ == "__main__":
    x_noisy = np.loadtxt("hw2_data_denoising.txt")
    plt.figure()
    plot_denoise(x_noisy, 0)
    LLS_denoise(x_noisy)
    plt.show()
