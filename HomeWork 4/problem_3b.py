import numpy as np
import matplotlib.pyplot as plt


def runges_function(t):
    return 1 / (1 + (25 * (t ** 2)))


def polynomial(coef, t):
    ans = 0
    for j in range(len(coef)):
        ans = ans + coef[j] * (t ** j)
    return ans


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


def cubic_spline_interpolation(n):
    points = np.linspace(-1, 1, n)
    amptitude_at_each_point = np.array(
        [runges_function(t_i) for t_i in points])
    no_of_cubic_pol = n - 1
    no_of_inner_points = n - 2
    matrix_of_variables = list()
    vector_of_amptitudes = list()
    for poly in range(no_of_cubic_pol):
        jth_polynomial_row = list()
        for each_point in range(poly):
            jth_polynomial_row.append([0, 0, 0, 0])
        jth_polynomial_row.append(
            [1, points[poly], points[poly]**2, points[poly]**3])
        for each_point in range(poly+1, no_of_cubic_pol):
            jth_polynomial_row.append([0, 0, 0, 0])
        matrix_of_variables.append(np.array(jth_polynomial_row).flatten())
        jth_polynomial_row = list()
        for each_point in range(poly):
            jth_polynomial_row.append([0, 0, 0, 0])
        jth_polynomial_row.append(
            [1, points[poly+1], points[poly+1]**2, points[poly+1]**3])
        for each_point in range(poly+1, no_of_cubic_pol):
            jth_polynomial_row.append([0, 0, 0, 0])
        matrix_of_variables.append(np.array(jth_polynomial_row).flatten())

    for poly in range(n-1):
        vector_of_amptitudes.append(amptitude_at_each_point[poly])
        vector_of_amptitudes.append(amptitude_at_each_point[poly+1])
    # first order derivative
    for poly in range(no_of_inner_points):
        vector_of_amptitudes.append(0)

    for poly in range(no_of_inner_points):
        jth_polynomial_row = list()
        for each_point in range(poly):
            jth_polynomial_row.append([0, 0, 0, 0])
        jth_polynomial_row.append(
            [0, 1, 2*points[poly+1], 3*(points[poly+1]**2)])
        jth_polynomial_row.append(
            [0, -1, -2*points[poly+1], -3*(points[poly+1]**2)])
        for each_point in range(no_of_inner_points-poly-1):
            jth_polynomial_row.append([0, 0, 0, 0])
        matrix_of_variables.append(np.array(jth_polynomial_row).flatten())

    # second order derivative
    for poly in range(no_of_inner_points):
        vector_of_amptitudes.append(0)

    for poly in range(no_of_inner_points):
        jth_polynomial_row = list()
        for each_point in range(poly):
            jth_polynomial_row.append([0, 0, 0, 0])
        jth_polynomial_row.append([0, 0, 2, 6*points[poly+1]])
        jth_polynomial_row.append([0, 0, -2, -6*points[poly+1]])
        for each_point in range(no_of_inner_points-poly-1):
            jth_polynomial_row.append([0, 0, 0, 0])
        matrix_of_variables.append(np.array(jth_polynomial_row).flatten())

    # spline interpolation
    vector_of_amptitudes.append(0)
    vector_of_amptitudes.append(0)
    ith_row = list()
    ith_row.append([0, 0, 2, 6*points[0]])
    for poly in range(1, no_of_cubic_pol):
        ith_row.append([0, 0, 0, 0])
    matrix_of_variables.append(np.array(ith_row).flatten())
    ith_row = list()
    for poly in range(1, no_of_cubic_pol):
        ith_row.append([0, 0, 0, 0])
    ith_row.append([0, 0, 2, 6*points[-1]])
    matrix_of_variables.append(np.array(ith_row).flatten())
    coef = np.linalg.solve(np.array(matrix_of_variables),
                           np.array(vector_of_amptitudes))
    return coef


def spline(coef, t):
    no_of_points = len(t)
    if no_of_points == 11:
        pts = 10
    else:
        pts = 5
    for j in range(no_of_points-1):
        T = np.linspace(t[j], t[j+1], pts)
        Y = [polynomial(coef[4*j:4*j+4], t_i) for t_i in T]
        plt.plot(T, Y, color="red", marker="o",
                 linestyle="solid", linewidth=2, markersize=5)
        plt.grid(True)
        plt.xlabel("t", fontsize=16)
        plt.ylabel("f(t)", fontsize=16)
        plt.title("Cubic spline interpolation with n = " +
                  str(no_of_points) + " points", fontsize=16)


def plot_cubic_spline():
    no_of_points = [11, 21]
    for i in range(len(no_of_points)):
        t = np.linspace(-1, 1, no_of_points[i])
        coef = cubic_spline_interpolation(no_of_points[i])
        plt.subplot(1, 3, i+2)
        spline(coef, t)
    plt.show()


def start():
    plot_runges()
    plot_cubic_spline()


if __name__ == "__main__":
    start()
