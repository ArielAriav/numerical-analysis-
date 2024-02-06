# this function create a new matrix
def create_matrix(size):
    matrix = []
    for i in range(size):
        # A loop that will run until "break"
        while True:
            try:
                # f string - for the expressions inside the curly braces will evaluate at runtime
                row = [int(input(f"Enter element at position ({i+1}, {j+1}): ")) for j in range(size)]
                # add the new row for the matrix
                matrix.append(row)
                # exit the while loop if input is valid
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer: ")

    return matrix


def print_matrix(matrix):
    # Runs over every row in the matrix and print it
    for row in matrix:
        print(row)


def add_matrices(matrix1, matrix2):
    lines = len(matrix1[0])
    cols = len(matrix1)
    # Initialize result matrix with empty lists for each row
    result = [[] for _ in range(lines)]
    # Iterate over each element and perform addition
    for i in range(lines):
        for j in range(cols):
           result[i].append(matrix1[i][j] + matrix2[i][j])
    return result


def multiply_matrices(matrix1, matrix2):
    # creates a 2D matrix where each element is initialized to zero.
    result = [[0 for _ in range(size)] for _ in range(size)]
    #  iterates over each element in the resulting matrix and computes its value based on the matrix multiplication
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


# _______________________________________________________________________________________________________________
# This block get the size of the matrices
while True:
    # Checks if the input is correct and if not - asks again
        size = int(input("Enter the size of the matrices: "))
        if size > 0:
            break
        else:
            print("Invalid input. Please enter a positive integer: ")


# Create matrices 1
print("Enter elements for the first matrix:")
matrix1 = create_matrix(size)


# Create matrices 2
print("\nEnter elements for the second matrix:")
matrix2 = create_matrix(size)

# Print the two matrices using print function
print("\nMatrix 1:")
print_matrix(matrix1)
print("\nMatrix 2:")
print_matrix(matrix2)

# Sum of matrices using our "add_matrices" function
sum_matrices = add_matrices(matrix1, matrix2)
print("\nSum of the matrices:")
print_matrix(sum_matrices)

# calculate the multiplication of the matrices using "multiply_matrices" function
product_matrices = multiply_matrices(matrix1, matrix2)
print("\nThe multiplication of the matrices:")
print_matrix(product_matrices)
