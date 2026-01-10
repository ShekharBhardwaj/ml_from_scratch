# Training data
# These are our known examples: size -> price
# The pattern is: price = 2 * size + 1
sizes = [1, 2, 3]
prices = [3, 5, 7]


def predict(size, weight, bias):
    """
    The line equation: y = mx + b
    In ML terms: predicted = weight * size + bias
    
    weight = how much price increases per unit size (slope)
    bias = base price when size is 0 (y-intercept)
    """
    return weight * size + bias


def loss(weight, bias):
    """
    Loss function: Mean Squared Error (MSE)
    
    Measures how wrong our predictions are.
    
    Steps:
    1. For each data point, compute predicted price
    2. Compute error = predicted - actual
    3. Square the error (makes all errors positive, punishes big errors)
    4. Average all squared errors
    
    Low loss = good predictions
    High loss = bad predictions
    Loss = 0 = perfect predictions
    """
    total = 0
    for i in range(len(sizes)):
        pred = predict(sizes[i], weight, bias)  # What we guessed
        error = pred - prices[i]                 # How wrong we are
        total += error ** 2                      # Square it
    return total / len(sizes)                    # Average


def gradient(weight, bias, dx=0.0001):
    """
    Gradient: tells us which direction to adjust weight and bias
    
    Gradient = [∂loss/∂weight, ∂loss/∂bias]
    
    ∂loss/∂weight = "If I nudge weight a tiny bit, how much does loss change?"
    ∂loss/∂bias = "If I nudge bias a tiny bit, how much does loss change?"
    
    We compute this numerically:
    - Nudge the value by a tiny amount (dx)
    - See how much loss changed
    - Divide change in loss by change in value
    """
    current_loss = loss(weight, bias)
    
    # Partial derivative with respect to weight
    # Hold bias constant, nudge only weight
    loss_weight_nudged = loss(weight + dx, bias)
    d_weight = (loss_weight_nudged - current_loss) / dx
    
    # Partial derivative with respect to bias
    # Hold weight constant, nudge only bias
    loss_bias_nudged = loss(weight, bias + dx)
    d_bias = (loss_bias_nudged - current_loss) / dx
    
    return [d_weight, d_bias]


def train(learning_rate=0.1, steps=20):
    """
    Gradient Descent: find weight and bias that minimize loss
    
    Algorithm:
    1. Start with random weight and bias (we use 0, 0)
    2. Compute gradient (which direction increases loss?)
    3. Move opposite to gradient (to decrease loss)
    4. Repeat until loss is small
    
    learning_rate = how big each step is
        - Too big: overshoot, bounce around
        - Too small: takes forever
    
    Update formula:
        new_weight = old_weight - learning_rate * gradient_weight
        new_bias = old_bias - learning_rate * gradient_bias
    
    We subtract because gradient points toward INCREASING loss,
    and we want to DECREASE loss.
    """
    # Start with a guess
    weight = 0
    bias = 0
    
    for i in range(steps):
        # Step 1: Compute gradient (direction of steepest increase)
        grad = gradient(weight, bias)
        
        # Step 2: Move opposite to gradient (toward lower loss)
        weight = weight - learning_rate * grad[0]
        bias = bias - learning_rate * grad[1]
        
        # Print progress every 5 steps
        if i % 5 == 0:
            print(f"Step {i}: weight={weight:.3f}, bias={bias:.3f}, loss={loss(weight, bias):.4f}")
    
    return weight, bias


# Train the model
print("Training linear regression...\n")
final_weight, final_bias = train()
print(f"\nFinal: weight={final_weight:.3f}, bias={final_bias:.3f}")
print(f"Expected: weight=2.000, bias=1.000")