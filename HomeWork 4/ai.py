import numpy as np

# Define the Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# Sample the Runge function at several points
x_values = np.linspace(-1, 1, num=11)
y_values = runge(x_values)

# Compute the second derivatives of the cubic spline at the sample points
n = len(x_values)
h = x_values[1:] - x_values[:-1]
A = 2 * np.diag(h[:-1], -1) + 2 * np.diag(h[1:], 1) + np.diag(h[:-1] + h[1:])
B = 6 * (y_values[2:] - y_values[1:-1]) / h[1:] - 6 * (y_values[1:-1] - y_values[:-2]) / h[:-1]
second_derivatives = np.linalg.solve(A, B)
second_derivatives = np.r_[0, second_derivatives, 0]

# Define a function that evaluates the cubic spline at a given point
def cubic_spline(x, x_values, y_values, second_derivatives):
    # Find the interval in which x lies
    i = np.searchsorted(x_values, x) - 1
    xi, xi1 = x_values[i], x_values[i + 1]
    yi, yi1 = y_values[i], y_values[i + 1]
    di, di1 = second_derivatives[i], second_derivatives[i + 1]

    # Compute the cubic spline at x
    t = (x - xi) / (xi1 - xi)
    ai = di * (xi1 - xi) - (yi1 - yi)
    bi = -di1 * (xi1 - xi) + (yi1 - yi)
    y = (1 - t) * yi + t * yi1 + t * (1 - t) * (ai * (1 - t) + bi * t)

    return y

# Evaluate the cubic spline at intermediate points
x_interp = np.linspace(-1, 1, num=100)
y_interp = [cubic_spline(x, x_values, y_values, second_derivatives) for x in x_interp]

# Plot the original function and the interpolated curve
import matplotlib.pyplot as plt
plt.plot(x_values, y_values, "o", label="Original points")
plt.plot(x_interp, y_interp, label="Interpolated curve")
plt.legend()
plt.show()
