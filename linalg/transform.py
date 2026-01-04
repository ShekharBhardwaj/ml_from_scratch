import math
from linalg.vector import Vector
from metrics.matrix import Matrix

def rotation_matrix(degrees):
    """Create a rotation matrix for given degrees"""
    radians = math.radians(degrees)
    return Matrix([
        [math.cos(radians), -math.sin(radians)],
        [math.sin(radians), math.cos(radians)]
    ])

def transform_shape(shape, matrix):
    return [matrix.multiply_vector(v) for v in shape]



def scale_matrix(sx, sy):
    """Scale by sx in x-direction, sy in y-direction"""
    return Matrix([
        [sx, 0],
        [0, sy]
    ])

def shear_matrix(shx, shy):
    """Shear by shx in x-direction, shy in y-direction"""
    return Matrix([
        [1, shx],
        [shy, 1]
    ])



if __name__ == "__main__":
    square = [
        Vector([0, 0]),
        Vector([1, 0]),
        Vector([1, 1]),
        Vector([0, 1])
    ]
    # rot90 = rotation_matrix(90)
    # rotated = transform_shape(square, rot90)
    # print("Original:", square)
    # print("Rotated 90°:", rotated)


    # scaled = transform_shape(square, scale_matrix(2, 3))
    # print("Scaled:", scaled)

    # # Shear
    # sheared = transform_shape(square, shear_matrix(0.5, 0))
    # print("Sheared:", sheared)

    combined = scale_matrix(2, 2).multiply_matrix(rotation_matrix(45))
    transformed = transform_shape(square, combined)
    print("Rotated 45° then scaled 2x:", transformed)