import jacobi_utilities
from sympy import *

x = Symbol('x')


def natural_cubic_spline(f, x0):
    """
    Natural Cubic Spline Interpolation

    Parameters:
    f (list): List of tuples representing data points (x, y).
    x0 (float): The value of x for which interpolation is required.

    Returns:
    None
    """
    # Sort data points by x-values
    f.sort(key=lambda point: point[0])

    # Check if the input data is valid
    if not all(isinstance(point, tuple) and len(point) == 2 for point in f):
        print("Invalid input format. Please provide a list of tuples containing (x, y) data points.")
        return
    if len(f) < 2:
        print("Insufficient data points for interpolation.")
        return
    if len(set(point[0] for point in f)) != len(f):
        print("Duplicate x-values found. Please provide distinct x-values.")
        return

    # Calculate differences between consecutive x-values
    h = [f[i + 1][0] - f[i][0] for i in range(len(f) - 1)]

    # Check for division by zero
    if any(h_val == 0 for h_val in h):
        print("Two consecutive x-values are the same. Unable to perform interpolation.")
        return

    # Calculate coefficients g, m, and d
    g = [0] + [h[i] / (h[i] + h[i - 1]) for i in range(1, len(f) - 1)] + [0]
    m = [0] + [1 - g[i] for i in range(1, len(f))]
    d = [0] + [((6 / (h[i - 1] + h[i])) * (((f[i + 1][1] - f[i][1]) / h[i]) - ((f[i][1] - f[i - 1][1]) / h[i - 1])))
               for i in range(1, len(f) - 1)] + [0]

    # Create the matrix
    mat = [[2 if i == j else m[i] if j == i - 1 else g[i] if j == i + 1 else 0 for j in range(len(f))]
           for i in range(len(f))]

    # Solve the system of equations using Jacobi iteration
    M = jacobi_utilities.Jacobi(mat, d)

    # Interpolate for each interval
    for loc in range(1, len(f)):
        s = (((f[loc][0] - x) ** 3) * M[loc - 1] + ((x - f[loc - 1][0]) ** 3) * M[loc]) / (6 * h[loc - 1])
        s += (((f[loc][0] - x) * f[loc - 1][1]) + ((x - f[loc - 1][0]) * f[loc][1])) / h[loc - 1]
        s -= (((f[loc][0] - x) * M[loc - 1] + (x - f[loc - 1][0]) * M[loc]) * h[loc - 1]) / 6
        print("s" + str(loc - 1) + "(x) = " + str(s))

    # Find the interval containing x0
    loc = 0
    for i in range(1, len(f)):
        if f[i - 1][0] < x0 < f[i][0]:
            loc = i
            break

    if loc == 0:
        print("No range found for x0")
        return

    # Interpolate at x0
    s = (((f[loc][0] - x) ** 3) * M[loc - 1] + ((x - f[loc - 1][0]) ** 3) * M[loc]) / (6 * h[loc - 1])
    s += (((f[loc][0] - x) * f[loc - 1][1]) + ((x - f[loc - 1][0]) * f[loc][1])) / h[loc - 1]
    s -= (((f[loc][0] - x) * M[loc - 1] + (x - f[loc - 1][0]) * M[loc]) * h[loc - 1]) / 6

    # Print the interpolated value at x0
    print("\nx0 between f(x" + str(loc - 1) + ") = " + str(f[loc - 1][0]) + " and f(x" + str(loc) + ") = " + str(
        f[loc][0]) + " so:")
    print("s" + str(loc - 1) + "(" + str(x0) + ") = " + str(float(s.subs(x, x0))))


if __name__ == '__main__':
    # Example data points
    f = [(1, 1), (2, 2), (3, 1), (4, 1.5), (5, 1)]
    x0 = 6

    print("func: " + str(f))
    print("x0 = " + str(x0) + "\n")
    natural_cubic_spline(f, x0)
