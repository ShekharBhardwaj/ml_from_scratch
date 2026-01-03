# Spiral Learning Log

## Day 1 - Vectors
**Date:** January 1, 2026
**Concepts:** Vector representation, addition, subtraction, scaling, magnitude (Pythagorean), dot product, normalization
**Key insight:** Dot product measures directional alignment (0 = perpendicular, positive = same direction, negative = opposite)
**Code:** linalg/vector.py
**Real-world connections:** Wind velocity, airplane velocity, electrical current


## Day 2 - Matrices
**Date:** January 2, 2026
**Concepts:** Matrix as transformation, shape, columns as transformed basis vectors, matrix-vector multiplication, matrix-matrix multiplication, identity matrix
**Key insight:** Matrix columns show where basis vectors [1,0] and [0,1] land after transformation. Matrix × vector = transformed vector.
**Code:** linalg/matrix.py (uses vector.py)
**Spiral connections:** Used Vector.dot() for all multiplication operations


## Day 3 - Determinants
**Date:** January 1, 2026
**Concepts:** Determinant as area/volume scaling factor, zero determinant means dimension collapse, cofactor expansion for n×n matrices
**Key insight:** det = 0 means transformation is irreversible (can't invert)
**Code:** Added minor(), cofactor(), determinant() to matrix.py
**Spiral connections:** Used Matrix class from Day 2, recursive structure builds on matrix operations
