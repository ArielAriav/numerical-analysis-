from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix, matrix_multiply, swap_row
import numpy as np

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""


def inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)

    # Checks whether the matrix is square
    # "matrix.shape" is a tuple representing the dimensions of the array
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    mat_size = matrix.shape[0]

    # creates an identity matrix - An identity matrix is a square matrix with 1 on the main diagonal and 0 elsewhere.
    # "np.identity" function from the NumPy library is used to create identity matrices.
    identity = np.identity(mat_size)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(mat_size):

        # Checks if the matrix is singular
        if matrix[i][i] == 0:
            flag = False
            for j in range(1, mat_size):
                if matrix[j][i] != 0:
                    swap_row(matrix,j,i)
                    flag = True
                    break
            if not flag :
                raise ValueError("Matrix is singular, cannot find its inverse")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]

            # Creates a suitable elementary matrix so that it turns into 1
            elementary_matrix = scalar_multiplication_elementary_matrix(mat_size, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")

            # performs matrix multiplication between the elementary_matrix and matrix using "NumPy's np.dot function".
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)

            #
            identity = np.dot(elementary_matrix, identity)


        # Zero out the elements above and below the diagonal
        for j in range(mat_size):
            if i != j:
                scalar = -matrix[j, i] / matrix[i, i]
                elementary_matrix = row_addition_elementary_matrix(mat_size, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                # use of np.dot for matrix multiplication
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)
                # use of np.dot for matrix multiplication
                identity = np.dot(elementary_matrix, identity)

    return identity


def final_inverse_test(mat, inverse_mat):
    if inverse_mat is not None:
        mat_size = mat.shape[0]
        result = matrix_multiply(mat, inverse_mat)
        # np.linalg.inv(A) is a NumPy function that computes the inverse of a square matrix
        if np.allclose(result, np.identity(mat_size)) and np.allclose(np.linalg.inv(mat), inverse_mat):
            return True
        else:
            return False


if __name__ == '__main__':

    A = np.array([[0, 2, 3],
                  [2, 0, 4],
                  [2, 4, 6]])

    try:
        A_inverse = inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================", bcolors.ENDC)
        if A_inverse is not None:
            is_inverse_work = final_inverse_test(A, A_inverse)
            if is_inverse_work:
                print("Test work successful (A * A_inverse = I)")
            else:
                print("Test didn't work(A * A_inverse != I)")

    except ValueError as e:
        print(str(e))

    print(np.linalg.inv(A))
