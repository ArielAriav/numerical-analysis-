import math
import numpy as np
from colors import bcolors


def max_steps(a, b, err):
    """
    Calculate the maximum number of iterations required to achieve the desired accuracy.

    Parameters:
        a (float): Start value of the interval.
        b (float): End value of the interval.
        err (float): Tolerable error.

    Returns:
        int: The minimum number of iterations required.
    """
    s = int(np.floor(-np.log2(err / (b - a)) / np.log2(2) - 1))
    return s


def bisection_method(f, a, b, tol=1e-6, print_iterations=None):
    """
    Perform the bisection method to find the root of a function within the given interval.

    Parameters:
        f (function): The function for which to find the root.
        a (float): Start value of the interval.
        b (float): End value of the interval.
        tol (float): Tolerable error. Default is 1e-6.
        print_iterations (list): Optional list of iteration numbers to print.

    Returns:
        float: The approximate root of the function.
    """

    c, k = 0, 0
    steps = max_steps(a, b, tol)

    # Print header for iteration table
    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    # Perform bisection iterations
    while abs(b - a) > tol and k < steps:
        c = a + (b - a) / 2

        if f(c) == 0:
            return c

        if f(c) * f(a) < 0:
            b = c
        elif f(c) * f(b) < 0:
            a = c

        # Print iteration details if the current iteration is in the print_iterations list
        if print_iterations is None or k in print_iterations:
            print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, f(a), f(b), c,
                                                                                              f(c)))

        k += 1

    return c


# ____________________________________________________________________________________________________________
if __name__ == '__main__':
    # Example function: f(x) = x^2 - 4 * sin(x)
    f = lambda x: x ** 2 - 4 * math.sin(x)

    # ________ change here to print only specific values of iterations _____
    # roots = bisection_method(f, 1, 3, print_iterations=[1, 3, 5, 10])
    roots = bisection_method(f, -1, 3)

    print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}", bcolors.ENDC)
    print(bcolors.OKBLUE, f"\nSearching for another root",bcolors.ENDC)
    roots = bisection_method(f,roots,3)
    print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}", bcolors.ENDC)