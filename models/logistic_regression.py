import math
import sys
sys.path.append('..')
from utils.split import train_test_split
from metrics.classification import accuracy

# Full data
hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
passed = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Split
X_train, X_test, y_train, y_test = train_test_split(hours, passed, test_size=0.2)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(x, weight, bias):
    z = weight * x + bias
    return sigmoid(z)

def loss(weight, bias, X, y):
    """Loss on given data"""
    total = 0
    for i in range(len(X)):
        pred = predict(X[i], weight, bias)
        error = pred - y[i]
        total += error ** 2
    return total / len(X)

def gradient(weight, bias, X, y, dx=0.0001):
    current_loss = loss(weight, bias, X, y)
    
    loss_w = loss(weight + dx, bias, X, y)
    d_weight = (loss_w - current_loss) / dx
    
    loss_b = loss(weight, bias + dx, X, y)
    d_bias = (loss_b - current_loss) / dx
    
    return [d_weight, d_bias]

def train(X, y, learning_rate=0.5, steps=100):
    weight = 0
    bias = 0
    
    for i in range(steps):
        grad = gradient(weight, bias, X, y)
        weight = weight - learning_rate * grad[0]
        bias = bias - learning_rate * grad[1]
    
    return weight, bias

# Train on training set only
w, b = train(X_train, y_train)
print(f"Trained: weight={w:.3f}, bias={b:.3f}")

# Evaluate on test set
y_pred = []
for x in X_test:
    p = predict(x, w, b)
    y_pred.append(1 if p > 0.5 else 0)

print(f"\nTest set predictions: {y_pred}")
print(f"Test set actual: {y_test}")
print(f"Test accuracy: {accuracy(y_test, y_pred)}")