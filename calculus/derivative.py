def derivative(f, x, dx=0.0001):
    """
    Compute the derivative of f at point x
    f: a function that takes a number and returns a number
    x: the point at which to compute the derivative
    dx: tiny nudge (default 0.0001)
    """
    return (f(x + dx) - f(x)) / dx


def gradient(f, point, dx=0.0001):
    """
    Compute the gradient of f at a point
    f: function that takes a list of numbers
    point: list of numbers [x, y, ...]
    Returns: list of partial derivatives
    """
    result = []
    for i in range(len(point)):
        # Make a copy of point
        point_nudged = point.copy()
        # Nudge only the i-th variable
        point_nudged[i] += dx
        # Compute rate of change
        partial = (f(point_nudged) - f(point)) / dx
        result.append(partial)
    return result



 

if __name__ == "__main__":
    # Test with f(x) = x²
    # def square(x):
    #     return x ** 2
    
    # print("Derivative of x² at x=3:", derivative(square, 3))  # Should be ~6
    
    # # Test with f(x) = x³
    # def cube(x):
    #     return x ** 3
    
    # print("Derivative of x³ at x=2:", derivative(cube, 2))  # Should be ~12

    def sum_of_squares(p):
        return p[0]**2 + p[1]**2
    
    print("Gradient at (3, 4):", gradient(sum_of_squares, [3, 4]))
    """
    Compute the gradient of f at a point
    f: function that takes a list of numbers
    point: list of numbers [x, y, ...]
    Returns: list of partial derivatives
    """
