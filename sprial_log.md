# Spiral Learning Log

## Day 1 - Vectors
**Date:** January 1, 2026
**Concepts:** Vector representation, addition, subtraction, scaling, magnitude (Pythagorean), dot product, normalization
**Key insight:** Dot product measures directional alignment (0 = perpendicular, positive = same direction, negative = opposite)
**Code:** linalg/vector.py
**Real-world connections:** Wind velocity, airplane velocity, electrical current


## Day 2 - Matrices
**Date:** January 1, 2026
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
**Date:** January 2, 2026
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
**Date:** January 3, 2026
**Concepts:** Rotation matrices (sin/cos), scaling matrices, shear matrices, matrix composition
**Key insight:** Combining transformations = multiplying matrices. Order matters.
**ML connection:** Image augmentation (rotating/scaling training images), feature transformations
**Code:** linalg/transform.py
**Spiral connections:** Used Vector (Day 1), Matrix (Day 2), multiply_vector (Day 2), multiply_matrix (Day 2)


## Day 8 - Derivatives
**Date:** January 4, 2026
**Concepts:** Derivative as rate of change, partial derivatives, gradient as vector of partial derivatives
**Key insight:** Gradient points toward steepest increase. To minimize, go in the negative gradient direction.
**ML connection:** Gradient descent uses the gradient to find the minimum of the loss function
**Code:** calculus/derivative.py (derivative, gradient)
**Spiral connections:** Gradient is a vector (Day 1) — has direction and magnitude


## Day 9 - Gradient Descent
**Date:** January 5, 2026
**Concepts:** Gradient descent algorithm, learning rate, iterative optimization
**Key insight:** To minimize a function, repeatedly step in the negative gradient direction
**ML connection:** This exact algorithm adjusts weights during model training to minimize loss
**Code:** calculus/gradient_descent.py
**Spiral connections:** Uses gradient (Day 8), which uses derivatives (Day 8), gradient is a vector (Day 1)


## Day 10 - Linear Regression
**Date:** January 6, 2026
**Concepts:** Linear regression (y = weight*x + bias), Mean Squared Error loss, training a model
**Key insight:** ML training = use gradient descent to find weight and bias that minimize loss
**ML connection:** This IS the ML model — predicting a number from input
**Code:** models/linear_regression.py
**Spiral connections:** Uses gradient descent (Day 9), gradient (Day 8), derivatives (Day 8)

## Day 11 - Logistic Regression
**Date:** January 7, 2026
**Concepts:** Classification (predict category), sigmoid function (squash to 0-1), logistic regression
**Key insight:** Logistic regression = linear regression + sigmoid. Everything else (loss, gradient, training) stays the same.
**Code:** models/logistic_regression.py
**Spiral connections:** Uses gradient descent (Day 9), loss function (Day 10), same training loop as linear regression (Day 10)


## Day 12 - Evaluation Metrics
**Date:** January 8, 2026
**Concepts:** Accuracy, precision, recall, F1 score, confusion matrix (TP, FP, TN, FN)
**Key insight:** Accuracy alone isn't enough. Precision = "when I predict yes, am I right?" Recall = "did I find all the yes cases?"
**ML connection:** Choose metric based on problem—spam filter needs precision, disease detection needs recall
**Code:** metrics/classification.py
**Spiral connections:** Evaluates logistic regression (Day 11), applies to any classification model


## Day 13 - Train/Test Split
**Date:** January 9, 2026
**Concepts:** Training set, test set, generalization, overfitting
**Key insight:** Always evaluate on data the model hasn't seen. Training accuracy can lie, test accuracy tells the truth.
**Code:** utils/split.py, updated logistic_regression.py
**Spiral connections:** Uses logistic regression (Day 11), accuracy metric (Day 12)


## Day 14 - Feature Scaling
**Date:** January 10, 2026
**Concepts:** Min-max scaling, feature normalization, why scale matters for gradient descent
**Key insight:** Features with different ranges make gradient descent unstable. Scale to 0-1 for faster, stable training.
**Code:** utils/scaling.py
**Spiral connections:** Prepares data for gradient descent (Day 9), used before training models (Days 10-11)


## Day 15 - Neural Network Concepts
**Date:** January 11, 2026
**Concepts:** Neural network structure, hidden layers, neurons, ReLU activation, forward pass
**Key insight:** Neural networks stack simple operations (weight × input + bias + activation) to learn complex non-linear patterns
**Code:** models/neural_network.py (forward pass only)
**Spiral connections:** Uses same building blocks as linear regression (Day 10), ReLU is like sigmoid (Day 11) but simpler


## Day 16 - Backpropagation
**Date:** January 11, 2026
**Concepts:** Chain rule, backpropagation, computing gradients for multiple weights
**Key insight:** Backpropagation = chain rule applied through the network. "When this weight changes, how much does loss change?"
**Code:** models/neural_network.py (training with numerical gradients)
**Spiral connections:** Same gradient descent loop (Day 9), same loss function (Day 10), just more weights



## Day 17 - Training on Multiple Data Points
**Date:** January 11, 2026
**Concepts:** Training on datasets, average loss, generalization to unseen data
**Key insight:** Train on multiple points, test on new points. If it predicts well on unseen data, it learned the pattern.
**Code:** models/neural_network.py (multi-point training)
**Spiral connections:** Same training loop (Day 16), average loss like Day 10



## Day 18 - Neural Network Class
**Date:** January 11, 2026
**Concepts:** Organizing neural network into a class, gradient explosion, proper gradient computation
**Key insight:** Compute ALL gradients before updating ANY weights. Otherwise gradients are wrong and training explodes.
**Code:** models/neural_net.py (clean NeuralNetwork class)
**Spiral connections:** Uses same gradient descent (Day 9), same loss function (Day 10), class structure like Vector/Matrix (Day 1-2)



## Day 19 - Loading MNIST
**Date:** January 11, 2026
**Concepts:** MNIST dataset, image as pixel array, image classification task
**Key insight:** An image is just a list of numbers (784 pixels). Neural network's job: pixels in → digit out.
**Code:** data/mnist_loader.py
**Spiral connections:** This is classification like logistic regression (Day 11), but with 784 inputs instead of 1


## Day 20 - MNIST Network (Part 1)
**Date:** January 11, 2026
**Concepts:** Softmax (outputs to probabilities), cross-entropy loss, numerical gradients for large network
**Key insight:** Network CAN learn — loss dropped from 2.4 to 0 on one image. But numerical gradients are too slow.
**Code:** models/mnist_net.py (forward, softmax, loss, gradients, train_one_image)
**Spiral connections:** Same gradient descent (Day 9), same loss concept (Day 10), scaled up to 50,890 weights



## Day 21 - Training Limits
**Date:** January 11, 2026
**Concepts:** Mini-batch training, accuracy measurement, limits of numerical gradients
**Key insight:** Numerical gradients are too slow for real networks. Need analytical backpropagation (PyTorch).
**Code:** models/mnist_net_fast.py (attempted faster training)
**Spiral connections:** Hit the wall — we understand the concepts, now we need better tools


## Day 22 - PyTorch Basics
**Date:** January 12, 2026
**Concepts:** Tensors, automatic gradients, nn.Linear, loss functions, optimizers, training loop
**Key insight:** PyTorch does the same thing we built from scratch, but faster and with less code. backward() replaces our manual gradient calculations.
**Code:** pytorch/linear_regression.py
**Spiral connections:** Same concepts as Day 10, now automated


## Day 23 - MNIST in PyTorch
**Date:** January 12, 2026
**Concepts:** torchvision datasets, DataLoader batching, nn.Module, training loop, testing accuracy
**Key insight:** Same network as Day 20 (784→64→10), but PyTorch computes gradients automatically. Training that would take days now takes 30 seconds.
**Code:** pytorch/mnist_pytorch.py
**Results:** 97.24% accuracy on 10,000 test images
**Spiral connections:** Same architecture as Day 20, same training loop as Day 22, now at scale


## Day 24 - Save/Load and Gradio App
**Date:** January 12, 2026
**Concepts:** torch.save(), load_state_dict(), Gradio interface, image preprocessing
**Key insight:** Trained models can be saved and deployed. A few lines of Gradio code creates a working web app.
**Code:** pytorch/gradio_app.py
**Spiral connections:** Uses trained model from Day 23, preprocessing like Day 14 (scaling)

