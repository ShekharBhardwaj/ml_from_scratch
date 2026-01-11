def relu(x):
    """ReLU activation: returns x if positive, 0 otherwise"""
    if x > 0:
        return x
    return 0

def forward(x, w1, b1, w2, b2, w_out, b_out):
    """
    Forward pass through a tiny neural network.
    
    x: input
    w1, b1: weights and bias for neuron 1
    w2, b2: weights and bias for neuron 2
    w_out, b_out: output layer weights and bias
    """
    # Hidden layer
    z1 = w1 * x + b1
    a1 = relu(z1)
    
    z2 = w2 * x + b2
    a2 = relu(z2)
    
    # Output layer
    output = w_out[0] * a1 + w_out[1] * a2 + b_out
    
    return output


if __name__ == "__main__":
    # Our weights from the example
    w1, b1 = 0.5, 0
    w2, b2 = -0.5, 0
    w_out = [1, 1]
    b_out = 0
    
    # Test
    print("Input 4 ->", forward(4, w1, b1, w2, b2, w_out, b_out))
    print("Input -2 ->", forward(-2, w1, b1, w2, b2, w_out, b_out))
    print("Input 0 ->", forward(0, w1, b1, w2, b2, w_out, b_out))

    # Test different inputs
    for x in [-4, -2, 0, 2, 4]:
        result = forward(x, w1, b1, w2, b2, w_out, b_out)
        print(f"Input {x:2} -> Output {result}")