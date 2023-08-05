import numpy as np
import matplotlib.pyplot as plt


def runges_function(t):
    return 1 / (1 + (25 * (t ** 2)))


def polynomial(coef, t):
    ans = 0
    for j in range(len(coef)):
        ans = ans + coef[j] * (t ** j)
    return ans


def polynomial_interpolation(n):
    points = np.linspace(-1, 1, n)
    amptitude = np.array([runges_function(t_i) for t_i in points])
    vandermond = np.stack([points ** i for i in range(n)], axis=1)
    coef = np.linalg.solve(vandermond, amptitude)
    return coef


def plot_runges():
    points = np.linspace(-1, 1, 100)
    amptitude = np.array([runges_function(t_i) for t_i in points])
    plt.subplot(1, 3, 1)
    plt.plot(points, amptitude, color="red", marker="o",
             linestyle="solid", linewidth=2, markersize=5)
    plt.grid(True)
    plt.xlabel("t", fontsize=16)
    plt.ylabel("f(t)", fontsize=16)
    plt.title("Runge's function", fontsize=16)


def plot_polynomial():
    no_of_points = [11, 21]
    points = np.linspace(-1, 1, 100)
    for i in range(len(no_of_points)):
        coef = polynomial_interpolation(no_of_points[i])
        amptitude_at_each_point = np.array(
            [polynomial(coef, t_i) for t_i in points])
        plt.subplot(1, 3, i+2)
        plt.plot(points, amptitude_at_each_point, color="red",
                 marker="o", linestyle="solid", linewidth=2, markersize=5)
        plt.grid(True)
        plt.xlabel("t", fontsize=16)
        plt.ylabel("f(t)", fontsize=16)
        plt.title("Polynomial interpolation with n = " +
                  str(no_of_points[i]) + " points", fontsize=16)
    plt.show()


def start():
    plot_runges()
    plot_polynomial()


if __name__ == "__main__":
    start()
