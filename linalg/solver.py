from linalg.vector import Vector
from metrics.matrix import Matrix

def solve(A, b):
    """
    Solve the system Ax = b
    A: Matrix (coefficients)
    b: Vector (constants)
    Returns: Vector (solution)
    """
    # Inside the function:
    # Check if determinant is 0 (raise error if so)
    if A.determinant() == 0:
        raise ValueError("Matrix is not invertible")
    # Find inverse of A
    inverse = A.inverse()
    # Multiply inverse by b
    return inverse.multiply_vector(b)


if __name__ == "__main__":
    A = Matrix([[1, 2], [2, 4]])  # Row 2 is just 2× Row 1
    b = Vector([3, 7])
    print("solution:", solve(A, b))