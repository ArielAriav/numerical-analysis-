from colors import bcolors


def lagrange_interpolation(x_data, y_data, x):
    """
    Lagrange Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value at the given x.

    Raises:
    ValueError: If the lengths of x_data and y_data are not equal or if they are empty.
    """
    # Input validation
    if not x_data or not y_data or len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must be non-empty lists of equal lengths.")

    n = len(x_data)  # Number of data points

    # Check if the interpolation point matches any of the given data points
    for i in range(n):
        if x == x_data[i]:
            return y_data[i]

    # If the interpolation point does not match any data point, proceed with Lagrange interpolation
    result = 0.0

    for i in range(n):
        term = y_data[i]  # Initialize the term with the y-value at the i data point
        for j in range(n):
            if i != j:  # Exclude the current data point when calculating the term
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])  # Calculate the term using Lagrange basis polynomials
        result += term  # Add the term to the result

    return result


if __name__ == '__main__':
    try:
        x_data = [1.2, 1.3, 1.4, 1.5, 1.6]
        y_data = [-3.50, -3.69, 0.9043, 1.1293, 2.3756]
        x_interpolate1 = 1.35  # The first x-value where we want to interpolate
        x_interpolate2 = 1.55  # The second x-value where we want to interpolate

        # Interpolate y-value at x_interpolate1
        y_interpolate1 = lagrange_interpolation(x_data, y_data, x_interpolate1)

        # Interpolate y-value at x_interpolate1
        y_interpolate2 = lagrange_interpolation(x_data, y_data, x_interpolate2)

        # Print the interpolated1 value
        print(bcolors.OKBLUE, "\nInterpolated value at x =", x_interpolate1, "is y =", y_interpolate1, bcolors.ENDC)

        # Print the interpolated1 value
        print(bcolors.OKBLUE, "\nInterpolated value at x =", x_interpolate2, "is y =", y_interpolate2, bcolors.ENDC)

    except ValueError as ve:
        print(bcolors.FAIL, "Error:", ve, bcolors.ENDC)
