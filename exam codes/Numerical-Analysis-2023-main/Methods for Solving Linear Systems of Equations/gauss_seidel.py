import numpy as np
from numpy.linalg import norm

from colors import bcolors
from matrix_utility import is_diagonally_dominant

def gauss_seidel(A, b, X0, TOL=1e-16, N=200, print_iterations=None):
    """
    Performs Gauss-Seidel iterations to solve the linear system of equations, Ax=b,
    starting from an initial guess, ``X0``.

    Terminates when the change in x is less than ``TOL``, or
    if ``N`` iterations have been exceeded.

    :param A: Coefficient matrix.
    :param b: Solution vector.
    :param X0: Initial guess.
    :param TOL: Tolerance for convergence (default is 1e-16).
    :param N: Maximum number of iterations (default is 200).
    :param print_iterations: List of iteration numbers to print (default is None, which prints all iterations).

    :return: The estimated solution.
    """
    n = len(A)  # Number of equations
    k = 1  # Iteration counter

    # Check if the matrix is diagonally dominant
    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - performing Gauss-Seidel algorithm\n')

    # Print header for the iteration table
    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    x = np.zeros(n, dtype=np.double)  # Initialize the solution vector
    while k <= N:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]  # Calculate the new value of x[i] using Gauss-Seidel formula

        if print_iterations is None or k in print_iterations:
            print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        # Check for convergence
        if norm(x - X0, np.inf) < TOL:
            return tuple(x)  # Return the solution if the tolerance is met

        k += 1  # Increment the iteration counter
        X0 = x.copy()  # Update the initial guess for the next iteration

    print("Maximum number of iterations exceeded")
    return tuple(x)  # Return the solution after maximum iterations are reached

if __name__ == '__main__':
    A = np.array([[3, -1, 1], [0, 1, -1], [1, 1, -2]])
    b = np.array([4, -1, -3])
    X0 = np.zeros_like(b)

    # ________ change here to print only specific values of iterations _____
    # solution = gauss_seidel(A, b, X0, print_iterations=[1, 5, 10, 15])
    solution = gauss_seidel(A, b, X0)

    print(bcolors.OKBLUE, "\nApproximate solution:", solution)
