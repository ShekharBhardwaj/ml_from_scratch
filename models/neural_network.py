def forward(x, w1, w2, b1, b2, w_out1, w_out2, b_out):
    """Network with 2 hidden neurons, no activation"""
    # Hidden layer (no ReLU)
    a1 = w1 * x + b1
    a2 = w2 * x + b2
    
    # Output layer
    output = w_out1 * a1 + w_out2 * a2 + b_out
    return output

def total_loss(X, Y, w1, w2, b1, b2, w_out1, w_out2, b_out):
    total = 0
    for i in range(len(X)):
        pred = forward(X[i], w1, w2, b1, b2, w_out1, w_out2, b_out)
        total += (pred - Y[i]) ** 2
    return total / len(X)

def train(X, Y, epochs=200, lr=0.01):
    # Smaller random starting weights
    w1, w2 = 0.1, 0.1
    b1, b2 = 0.0, 0.0
    w_out1, w_out2 = 0.1, 0.1
    b_out = 0.0
    dx = 0.0001
    
    for epoch in range(epochs):
        current_loss = total_loss(X, Y, w1, w2, b1, b2, w_out1, w_out2, b_out)
        
        # Gradients for all weights
        grad_w1 = (total_loss(X, Y, w1+dx, w2, b1, b2, w_out1, w_out2, b_out) - current_loss) / dx
        grad_w2 = (total_loss(X, Y, w1, w2+dx, b1, b2, w_out1, w_out2, b_out) - current_loss) / dx
        grad_b1 = (total_loss(X, Y, w1, w2, b1+dx, b2, w_out1, w_out2, b_out) - current_loss) / dx
        grad_b2 = (total_loss(X, Y, w1, w2, b1, b2+dx, w_out1, w_out2, b_out) - current_loss) / dx
        grad_w_out1 = (total_loss(X, Y, w1, w2, b1, b2, w_out1+dx, w_out2, b_out) - current_loss) / dx
        grad_w_out2 = (total_loss(X, Y, w1, w2, b1, b2, w_out1, w_out2+dx, b_out) - current_loss) / dx
        grad_b_out = (total_loss(X, Y, w1, w2, b1, b2, w_out1, w_out2, b_out+dx) - current_loss) / dx
        
        # Update all weights
        w1 = w1 - lr * grad_w1
        w2 = w2 - lr * grad_w2
        b1 = b1 - lr * grad_b1
        b2 = b2 - lr * grad_b2
        w_out1 = w_out1 - lr * grad_w_out1
        w_out2 = w_out2 - lr * grad_w_out2
        b_out = b_out - lr * grad_b_out
        
        if epoch % 40 == 0:
            print(f"Epoch {epoch}: loss={current_loss:.4f}")
    
    return w1, w2, b1, b2, w_out1, w_out2, b_out


if __name__ == "__main__":
    X = [1, 2, 3, 4]
    Y = [2, 4, 6, 8]
    
    print("Training neural network (no activation)...")
    print("Pattern: output = 2 * input\n")
    
    w1, w2, b1, b2, w_out1, w_out2, b_out = train(X, Y, epochs=200, lr=0.01)
    
    print("\nTesting:")
    for x in [1, 2, 3, 4, 5]:
        pred = forward(x, w1, w2, b1, b2, w_out1, w_out2, b_out)
        print(f"Input {x} -> Predicted {pred:.2f}, Expected {2*x}")