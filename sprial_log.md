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



## Day 7 - Geometric Transformations
**Date:** January 2, 2026
**Concepts:** Rotation matrices (sin/cos), scaling matrices, shear matrices, matrix composition
**Key insight:** Combining transformations = multiplying matrices. Order matters.
**ML connection:** Image augmentation (rotating/scaling training images), feature transformations
**Code:** linalg/transform.py
**Spiral connections:** Used Vector (Day 1), Matrix (Day 2), multiply_vector (Day 2), multiply_matrix (Day 2)


## Day 8 - Derivatives
**Date:** January 2, 2026
**Concepts:** Derivative as rate of change, partial derivatives, gradient as vector of partial derivatives
**Key insight:** Gradient points toward steepest increase. To minimize, go in the negative gradient direction.
**ML connection:** Gradient descent uses the gradient to find the minimum of the loss function
**Code:** calculus/derivative.py (derivative, gradient)
**Spiral connections:** Gradient is a vector (Day 1) — has direction and magnitude


## Day 9 - Gradient Descent
**Date:** January 2, 2026
**Concepts:** Gradient descent algorithm, learning rate, iterative optimization
**Key insight:** To minimize a function, repeatedly step in the negative gradient direction
**ML connection:** This exact algorithm adjusts weights during model training to minimize loss
**Code:** calculus/gradient_descent.py
**Spiral connections:** Uses gradient (Day 8), which uses derivatives (Day 8), gradient is a vector (Day 1)


## Day 10 - Linear Regression
**Date:** January 2, 2026
**Concepts:** Linear regression (y = weight*x + bias), Mean Squared Error loss, training a model
**Key insight:** ML training = use gradient descent to find weight and bias that minimize loss
**ML connection:** This IS the ML model — predicting a number from input
**Code:** models/linear_regression.py
**Spiral connections:** Uses gradient descent (Day 9), gradient (Day 8), derivatives (Day 8)

## Day 11 - Logistic Regression
**Date:** January 2, 2026
**Concepts:** Classification (predict category), sigmoid function (squash to 0-1), logistic regression
**Key insight:** Logistic regression = linear regression + sigmoid. Everything else (loss, gradient, training) stays the same.
**Code:** models/logistic_regression.py
**Spiral connections:** Uses gradient descent (Day 9), loss function (Day 10), same training loop as linear regression (Day 10)


## Day 12 - Evaluation Metrics
**Date:** January 2, 2026
**Concepts:** Accuracy, precision, recall, F1 score, confusion matrix (TP, FP, TN, FN)
**Key insight:** Accuracy alone isn't enough. Precision = "when I predict yes, am I right?" Recall = "did I find all the yes cases?"
**ML connection:** Choose metric based on problem—spam filter needs precision, disease detection needs recall
**Code:** metrics/classification.py
**Spiral connections:** Evaluates logistic regression (Day 11), applies to any classification model


## Day 13 - Train/Test Split
**Date:** January 2, 2026
**Concepts:** Training set, test set, generalization, overfitting
**Key insight:** Always evaluate on data the model hasn't seen. Training accuracy can lie, test accuracy tells the truth.
**Code:** utils/split.py, updated logistic_regression.py
**Spiral connections:** Uses logistic regression (Day 11), accuracy metric (Day 12)