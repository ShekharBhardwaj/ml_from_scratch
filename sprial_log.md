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


## Day 4 - Matrix Inverse
**Date:** January 1, 2026
**Concepts:** Inverse as "undo" transformation, A × A⁻¹ = I, inverse only exists when det ≠ 0
**Key insight:** Inverse lets you solve equations—if you know the transformation and the result, you can find the original input
**ML connection:** Linear regression normal equation uses (XᵀX)⁻¹ to find optimal weights directly
**Code:** Added inverse() to matrix.py
**Spiral connections:** Uses determinant (Day 3), verified with multiply_matrix (Day 2)


## Day 5 - Eigenvalues
**Date:** January 1, 2026
**Concepts:** Eigenvectors (direction unchanged by transformation), eigenvalues (scaling factor), trace, characteristic equation
**Key insight:** Eigenvectors are special directions that only get scaled, not rotated. Eigenvalues tell you the scaling factor.
**ML connection:** PCA finds directions of maximum variance (eigenvectors of covariance matrix), PageRank, neural network stability
**Code:** Added eigenvalues() to matrix.py
**Spiral connections:** Uses determinant (Day 3), derived from A - λI concept



## Day 6 - Linear Equation Solver
**Date:** January 2, 2026
**Concepts:** Systems of linear equations as Ax = b, solving via inverse
**Key insight:** Matrix inverse lets you "undo" the transformation to find the original input
**ML connection:** Same pattern used in linear regression's normal equation
**Code:** linalg/solver.py
**Spiral connections:** Used Vector (Day 1), Matrix (Day 2), determinant (Day 3), inverse (Day 4)
