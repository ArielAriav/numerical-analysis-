import numpy as np
from inverse_matrix import inverse
from colors import bcolors
from matrix_utility import print_matrix


def norm(mat):
    size = len(mat)
    max_row = 0

    # Iterate through each row of the matrix
    for row in range(size):
        sum_row = 0

        # Iterate through each element in the current row
        for col in range(size):
            sum_row += abs(mat[row][col])  # Sum the absolute values of elements in the row

        # Update max_row if the current row sum is greater
        if sum_row > max_row:
            max_row = sum_row

    return max_row  # Return the maximum row sum


def condition_number(A):
    # Calculation of the norm of A
    norm_A = norm(A)
    # np.linalg.inv(A) is a NumPy function that computes the inverse of a square matrix
    A_inv = inverse(A)
    # Calculating the norm of the inverse A
    norm_A_inv = norm(A_inv)
    # Calculation of the condition
    cond = norm_A * norm_A_inv


    print(bcolors.OKBLUE, "A:", bcolors.ENDC)
    print_matrix(A)

    print(bcolors.OKBLUE, "Inverse of A:", bcolors.ENDC)
    print_matrix(A_inv)

    print(bcolors.OKBLUE, "Max Norm of A:", bcolors.ENDC, norm_A, "\n")

    print(bcolors.OKBLUE, "Max Norm of the Inverse of A:", bcolors.ENDC, norm_A_inv)

    return cond


if __name__ == '__main__':
    A = np.array(([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 6]]))

    cond = condition_number(A)
    print(bcolors.OKGREEN, "\n condition number: ", cond, bcolors.ENDC)








