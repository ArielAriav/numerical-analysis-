from colors import bcolors
from matrix_utility import *


def GaussJordanElimination(matrix, vector):
    """
    Function for solving a linear equation using Gauss-Jordan elimination method
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=A(-1)b
    """
    # Pivoting process
    matrix, vector = RowXchange(matrix, vector)
    # Inverse matrix calculation
    invert = InverseMatrix(matrix, vector)
    return MulMatrixVector(invert, vector)


def DecomposeLU(matrix, is_upper):
    """
    Function for decomposition into an upper or lower triangular matrix (U/L)
    :param matrix: Matrix nxn
    :param is_upper: Boolean indicating whether to decompose into U (True) or L (False)
    :return: U or L matrix
    """
    result = MakeIMatrix(len(matrix), len(matrix))
    for i in range(len(matrix[0])):
        matrix, _ = RowXchageZero(matrix, [])
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            sign = 1 if is_upper else -1
            elementary[j][i] = -sign * (matrix[j][i]) / matrix[i][i]
            if not is_upper:
                result[j][i] = sign * (matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    return MultiplyMatrix(result, matrix)


def SolveUsingLU(matrix, vector):
    """
    Function for solving a linear equation by decomposing LU
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=U(-1)L(-1)b
    """
    matrixU = DecomposeLU(matrix, is_upper=True)
    matrixL = DecomposeLU(matrix, is_upper=False)
    if matrixU is None or matrixL is None:
        return None
    return MultiplyMatrix(InverseMatrix(matrixU), MultiplyMatrix(InverseMatrix(matrixL), vector))


def solveMatrix(matrixA, vectorb):
    detA = Determinant(matrixA, 1)
    print(bcolors.YELLOW, "\nDET(A) = ", detA)

    if detA != 0:
        print("CondA = ", Cond(matrixA, InverseMatrix(matrixA, vectorb)), bcolors.ENDC)
        print(bcolors.OKBLUE, "\nNon-singular Matrix - Perform Gauss-Jordan Elimination", bcolors.ENDC)
        result = GaussJordanElimination(matrixA, vectorb)
        print(np.array(result))
        return result
    else:
        print("Singular Matrix - Unable to solve.")
        return None


def polynomialInterpolation(table_points, x):
    if len(table_points) < len(table_points[0]):
        print("Insufficient data points for interpolation.")
        return None

    matrix = [[point[0] ** i for i in range(len(table_points))] for point in table_points]
    b = [[point[1]] for point in table_points]

    print(bcolors.OKBLUE, "The matrix obtained from the points: ", bcolors.ENDC, '\n', np.array(matrix))
    print(bcolors.OKBLUE, "\nb vector: ", bcolors.ENDC, '\n', np.array(b))
    matrixSol = solveMatrix(matrix, b)

    if matrixSol is not None:
        result = sum([matrixSol[i][0] * (x ** i) for i in range(len(matrixSol))])
        print(bcolors.OKBLUE, "\nThe polynomial:", bcolors.ENDC)
        print('P(X) = ' + '+'.join(['(' + str(matrixSol[i][0]) + ') * x^' + str(i) + ' ' for i in range(len(matrixSol))]))
        print(bcolors.OKGREEN, f"\nThe Result of P(X={x}) is:", bcolors.ENDC)
        print(result)
        return result
    else:
        return None

if __name__ == '__main__':

    table_points = [(1, 1), (2, 0), (5, 2)]
    x = 3
    print(bcolors.OKBLUE, "----------------- Interpolation & Extrapolation Methods -----------------\n", bcolors.ENDC)
    print(bcolors.OKBLUE, "Table Points: ", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation to the point: ", bcolors.ENDC, x,'\n')
    polynomialInterpolation(table_points, x)
    print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n", bcolors.ENDC)