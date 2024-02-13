import numpy as np
from colors import bcolors
from matrix_utility import swap_row


def gaussian_elimination(mat):
    n = len(mat)

    # If the matrix is singular the flag will be different from -1
    singular_flag = forward_substitution(mat)

    if singular_flag != -1:

        # is checking if the value at position [singular_flag][n] is truthy
        # values like 0, None, and empty strings are considered falsy.
        if mat[singular_flag][n]:
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return backward_substitution(mat)


def forward_substitution(mat):
    n = len(mat)
    for k in range(n):
        print(f"\nStep {k + 1}:")

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = k
        v_max = mat[pivot_row][k]
        for i in range(k + 1, n):
            if abs(mat[i][k]) > v_max:
                v_max = mat[i][k]
                pivot_row = i

        if mat[k][pivot_row] == 0:
            return k  # Matrix is singular

        if pivot_row != k:
            swap_row(mat, k, pivot_row)
            print(f"Swap Row {k + 1} with Row {pivot_row + 1}:")
            print(np.array(mat))  # Print the matrix after swapping

        for i in range(k + 1, n):
            m = mat[i][k] / mat[k][k]
            print(f"Multiplier for Row {i + 1}: {m}")

            for j in range(k + 1, n + 1):
                mat[i][j] -= mat[k][j] * m

            mat[i][k] = 0
            print(f"Row {i + 1} - {m} * Row {k + 1}:")
            print(np.array(mat))  # Print the matrix after row operations

    return -1


# function to calculate the values of the unknowns
def backward_substitution(mat):
    n = len(mat)
    x = np.zeros(n)  # An array to store solution

    # Start calculating from last equation up to the first
    for i in range(n - 1, -1, -1):

        x[i] = mat[i][n]

        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, n):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])

    return x


def check_solution(original_matrix, solution_vector_for_check):
    original_matrix = np.array(original_matrix)

    # Extract coefficients and constants from the original_matrix
    # This selects all rows and all columns except the last one
    coefficients = original_matrix[:, :-1]
    # This selects all rows and only the last column
    vector_b = original_matrix[:, -1]

    # Solve the system of equations
    calculated_solution = np.linalg.solve(coefficients, vector_b)

    # Check if the calculated solution is close to the provided solution_vector_for_check
    if np.allclose(calculated_solution, solution_vector_for_check):
        print("The test worked")
        return True
    else:
        print("The test didn't work")
        return False

# ___________________________________________________________________________________________________________


if __name__ == '__main__':

    A_b = [[0, -1, 2, -1, -8],
        [2, 0, 3, -3, -20],
        [1, 1, 0, 0, -2],
        [1, -1, 4, 0, 4]]

    result = gaussian_elimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print(bcolors.OKBLUE, "\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))

    check_solution(A_b, result)