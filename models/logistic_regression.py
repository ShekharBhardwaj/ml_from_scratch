import math

# Training data
hours = [1, 2, 3, 4]
passed = [0, 0, 1, 1]

def sigmoid(z):
    """Squash any number to range 0-1"""
    return 1 / (1 + math.exp(-z))

def predict(x, weight, bias):
    """Logistic regression prediction"""
    z = weight * x + bias
    return sigmoid(z)

def loss(weight, bias):
    """Mean squared error loss"""
    total = 0
    for i in range(len(hours)):
        pred = predict(hours[i], weight, bias)
        error = pred - passed[i]
        total += error ** 2
    return total / len(hours)

def gradient(weight, bias, dx=0.0001):
    """Compute gradient by nudging"""
    current_loss = loss(weight, bias)
    
    loss_w = loss(weight + dx, bias)
    d_weight = (loss_w - current_loss) / dx
    
    loss_b = loss(weight, bias + dx)
    d_bias = (loss_b - current_loss) / dx
    
    return [d_weight, d_bias]

def train(learning_rate=0.5, steps=100):
    weight = 0
    bias = 0
    
    for i in range(steps):
        grad = gradient(weight, bias)
        weight = weight - learning_rate * grad[0]
        bias = bias - learning_rate * grad[1]
        
        if i % 20 == 0:
            print(f"Step {i}: weight={weight:.3f}, bias={bias:.3f}, loss={loss(weight, bias):.4f}")
    
    return weight, bias

# Train
print("Training logistic regression...\n")
w, b = train()
print(f"\nFinal: weight={w:.3f}, bias={b:.3f}")

# Test predictions
print("\nPredictions:")
for h in [1, 2, 3, 4]:
    p = predict(h, w, b)
    result = "Pass" if p > 0.5 else "Fail"
    print(f"Hours={h}: {p:.2f} -> {result}")