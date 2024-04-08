from colors import bcolors


def linearInterpolation(table_points, point):
    # Iterate through each pair of adjacent table points
    for i in range(len(table_points) - 1):
        # Extract x and y values of the current and next table points
        x1, y1 = table_points[i]
        x2, y2 = table_points[i + 1]

        # Check if the point lies within the range of the current and next table points
        if x1 <= point < x2:
            # Calculate the slope and perform linear interpolation
            m = (y1 - y2) / (x1 - x2)
            result = y1 + m * (point - x1)
            # Print the interpolated value
            print(bcolors.OKGREEN, "\nThe approximation of the point", point, "is:", bcolors.ENDC, round(result, 4))
            return
        # Handle the case where the point matches a table point exactly
        elif point == x1:
            print(bcolors.OKGREEN, "\nThe point", point, "matches a table point exactly. Interpolation is not needed.",
                  bcolors.ENDC)
            return
    # If the point is outside the range of the table points, print a message
    print(bcolors.OKGREEN, "\nThe point", point, "is outside the range of the table points.", bcolors.ENDC)


if __name__ == '__main__':
    table_points = [(1,1), (2,0), (5, 2)]
    x1 = 3
    x2 = 4
    # Print header for the output
    print(bcolors.OKBLUE, "----------------- Interpolation & Extrapolation Methods -----------------\n", bcolors.ENDC)
    # Print the table points and the point to be approximated
    print(bcolors.OKBLUE, "Table Points:", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation to the point:", bcolors.ENDC, x1)
    # Perform linear interpolation
    linearInterpolation(table_points, x1)
    # Print footer for the output
    print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n",
          bcolors.ENDC)
