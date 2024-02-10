import numpy as np
from colors import bcolors
from matrix_utility import swap_row


def gaussian_elimination(mat):
    n = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:

        if mat[singular_flag][n]:
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return backward_substitution(mat)


def forward_substitution(mat):
    n = len(mat)
    for k in range(n):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = k
        v_max = mat[pivot_row][k]
        for i in range(k+1, n):
            if abs(mat[i][k]) > v_max:
                v_max = mat[i][k]
                # pivot_row will contain the index of the row with the largest absolute value in the current column.
                pivot_row = i

        # if a principal diagonal element is zero,it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if mat[k][pivot_row] == 0:
            # Matrix is singular
            return k

        # Swap the current row with the pivot row
        if pivot_row != k:
            swap_row(mat, k, pivot_row)
        # End Partial Pivoting

        for i in range(k + 1, n):

            #  Compute the multiplier
            m = mat[i][k] / mat[k][k]

            # subtract fth multiple of corresponding kth row element
            for j in range(k + 1, n + 1):
                mat[i][j] -= mat[k][j] * m

            # filling lower triangular matrix with zeros
            mat[i][k] = 0

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


def check_solution(original_matrix, solution_vector):
    # Check if the solution satisfies the original system of equations
    original_matrix = np.array(original_matrix)
    rows, cols = original_matrix.shape
    for i in range(rows):
        sum = 0
        for j in range(cols - 1):  # Exclude the last column (augmented column)
            sum += original_matrix[i][j] * solution_vector[j]

        # Check if the sum is close to the corresponding element in the solution vector
        if not np.isclose(sum, original_matrix[i, -1]):
            print("The test didn't work")
            return False

    print("The test worked")
    return True


# ___________________________________________________________________________________________________________


if __name__ == '__main__':

    A_b = [[1, -1, 2, -1, -8],
        [2, -2, 3, -3, -20],
        [1, 1, 1, 0, -2],
        [1, -1, 4, 3, 4]]

    result = gaussian_elimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print(bcolors.OKBLUE, "\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))

    check_solution(A_b, result)