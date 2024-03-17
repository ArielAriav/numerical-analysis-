import numpy as np
from numpy.linalg import norm

from colors import bcolors  # Importing custom color definitions
from matrix_utility import is_diagonally_dominant

"""
Performs Jacobi iterations to solve the linear system of equations Ax=b, 
starting from an initial guess, ``X0``.

Terminates when the change in x is less than ``TOL``, or
if ``N`` [default=200] iterations have been exceeded.

Receives 5 parameters:
    1. a: The NxN coefficient matrix of the linear system.
    2. b: The vector representing the right-hand side of the linear system.
    3. X0: The desired initial guess for the solution.
        If X0 is None, the initial guess will be determined as a vector of zeros.
    4. TOL: Tolerance - the desired limitation of the solution's anomaly.
        If tolerance is None, the default value will be set as 1e-16.
    5. N: The maximum number of possible iterations to achieve the most exact solution.
        If N is None, the default value will be set as 200.

Returns:
    x: The estimated solution.

"""

def jacobi_iterative(A, b, X0, TOL=1e-16, N=200, print_iterations=None):
    """
    Performs Jacobi iterations to solve the linear system of equations, Ax=b,
    starting from an initial guess, ``X0``.

    Terminates when the change in x is less than ``TOL``, or
    if ``N`` iterations have been exceeded.

    :param A: Coefficient matrix.
    :param b: Solution vector.
    :param X0: Initial guess.
    :param TOL: Tolerance for convergence (default is 1e-16).
    :param N: Maximum number of iterations (default is 200).
    :param print_iterations: Optional list of iteration numbers to print. If None, all iterations are printed.

    :return: The estimated solution.
    """
    n = len(A)  # Number of equations
    k = 1  # Iteration counter

    # Check if the matrix is diagonally dominant
    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - performing Jacobi algorithm\n')

    # Print header for the iteration table
    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    # Iterative process
    while k <= N:
        x = np.zeros(n, dtype=np.double)  # Initialize the solution vector for this iteration
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * X0[j]
            x[i] = (b[i] - sigma) / A[i][i]  # Calculate the new value of x[i] using Jacobi formula

        # Print iteration values if the iteration number is in the specified list or if no list is provided
        if print_iterations is None or k in print_iterations:
            print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        # Check for convergence
        if norm(x - X0, np.inf) < TOL:
            return tuple(x)  # Return the solution if the tolerance is met

        k += 1  # Increment the iteration counter
        X0 = x.copy()  # Update the initial guess for the next iteration

    print("Maximum number of iterations exceeded")
    return tuple(x)  # Return the solution after maximum iterations are reached


# ________________________________________________________________________________________
# Example usage:
A = np.array([[3, -1, 1],
              [0, 1, -1],
              [1, 1, -2]])
b = np.array([4, -1, -3])

x = np.zeros_like(b, dtype=np.double)

# ________ change here to print only specific values of iterations _____
# print_iterations = [1, 3, 5]

# ________ change here to print only specific values of iterations _____
# solution = jacobi_iterative(A, b, x, print_iterations=[1, 3, 5])
solution = jacobi_iterative(A, b, x)


print(bcolors.OKBLUE,"\nApproximate solution:", solution)
