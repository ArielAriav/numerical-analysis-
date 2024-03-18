import numpy as np

from colors import bcolors
from matrix_utility import swap_rows_elementary_matrix, row_addition_elementary_matrix


def lu(A):
    N = len(A)
    L = np.eye(N)  # Create an identity matrix of size N x N
    elementary_matrix = []
    for i in range(N):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_index = i
        v_max = abs(A[i][i])
        for j in range(i + 1, N):
            if abs(A[j][i]) > v_max:
                v_max = abs(A[j][i])
                pivot_index = j

        # if a principal diagonal element is zero, it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if A[i][pivot_index] == 0:
            raise ValueError("Matrix is singular")

        # Swap the current row with the pivot row
        if pivot_index != i:
            e_matrix = swap_rows_elementary_matrix(N, i, pivot_index)

            # performs matrix multiplication between the elementary matrix e_matrix and the matrix A.
            A = np.dot(e_matrix, A)
            print(f"Elementary matrix for swapping row {i} with row {pivot_index}:\n{e_matrix}\n")
            elementary_matrix.append(e_matrix)  # An array to store all elementary matrices
            print(f"The matrix after swapping:\n{A}\n")
            print(bcolors.OKGREEN, "---------------------------------------------------------------------------", bcolors.ENDC)

        for j in range(i + 1, N):

            # Compute the multiplier
            m = -A[j][i] / A[i][i]
            e_matrix = row_addition_elementary_matrix(N, j, i, m)
            e_inverse = np.linalg.inv(e_matrix)
            L = np.dot(L, e_inverse)
            A = np.dot(e_matrix, A)
            print(f"Elementary matrix to zero the element in row {j} below the pivot in column {i}:\n{e_matrix}\n")
            print(f"The matrix after elementary operation:\n{A}\n")
            print(bcolors.OKGREEN, "---------------------------------------------------------------------------", bcolors.ENDC)

    U = A
    return L, U


# function to calculate the values of the unknowns using backward substitution
def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):
        x[i] = mat[i][N]

        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        # Guard against division by zero
        if mat[i][i] == 0:
            raise ValueError("Division by zero")
        x[i] = (x[i] / mat[i][i])

    return x


def lu_solve(A_b):
    L, U = lu(A_b)

    print("Lower triangular matrix L:\n", L)
    print("Upper triangular matrix U:\n", U)

    result = backward_substitution(U)

    print(bcolors.OKBLUE, "\nSolution for the system:")
    for x in result:
        print("{:.6f}".format(x))


if __name__ == '__main__':
    A_b = [[1, -1, 2, -1, -8],
           [2, -2, 3, -3, -20],
           [1, 1, 1, 0, -2],
           [1, -1, 4, 3, 4]]

    lu_solve(A_b)

