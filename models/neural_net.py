class NeuralNetwork:
    def __init__(self, hidden_size=2):
        """
        Simple neural network: 1 input, hidden layer, 1 output
        
        hidden_size: how many neurons in the hidden layer
        """
        # Hidden layer weights - each hidden neuron has one weight for the input
        # We start with small values (0.1) to avoid gradient explosion
        self.w_hidden = [0.1] * hidden_size
        
        # Hidden layer biases - each hidden neuron has one bias
        # Biases start at 0
        self.b_hidden = [0.0] * hidden_size
        
        # Output layer weights - one weight connecting each hidden neuron to output
        self.w_out = [0.1] * hidden_size
        
        # Output layer bias - single value added to final output
        self.b_out = 0.0
        
        # Store hidden_size so we can loop through neurons later
        self.hidden_size = hidden_size
    
    def forward(self, x):
        """
        Forward pass: push input through the network to get output
        
        x: single input number
        returns: single output number
        """
        # Step 1: Compute hidden layer values
        # Each hidden neuron computes: weight * input + bias
        self.hidden = []
        for i in range(self.hidden_size):
            # Neuron i: multiply input by its weight, add its bias
            h = self.w_hidden[i] * x + self.b_hidden[i]
            self.hidden.append(h)
        # self.hidden now contains output of each hidden neuron
        
        # Step 2: Compute output layer
        # Output = sum of (hidden_value * output_weight) + output_bias
        output = self.b_out  # Start with bias
        for i in range(self.hidden_size):
            # Add contribution from each hidden neuron
            output += self.w_out[i] * self.hidden[i]
        
        return output
    
    def loss(self, X, Y):
        """
        Compute mean squared error across all data points
        
        X: list of inputs [1, 2, 3, 4]
        Y: list of expected outputs [2, 4, 6, 8]
        returns: average of squared errors
        """
        total = 0
        for i in range(len(X)):
            # Get prediction for this input
            pred = self.forward(X[i])
            # Compute squared error: (predicted - actual)^2
            total += (pred - Y[i]) ** 2
        # Return average (divide by number of data points)
        return total / len(X)
    
    def train(self, X, Y, epochs=200, lr=0.01):
        """
        Train the network using numerical gradients
        
        X: list of inputs
        Y: list of expected outputs
        epochs: number of training iterations
        lr: learning rate (how big each weight update is)
        """
        # dx: tiny nudge for computing numerical gradient
        dx = 0.0001
        
        # Training loop: repeat for specified number of epochs
        for epoch in range(epochs):
            
            # Compute loss with current weights (before any changes)
            current_loss = self.loss(X, Y)
            
            # We'll store all gradients first, then update weights
            # This prevents one weight change from affecting other gradient calculations
            grad_w_hidden = []  # Will hold gradient for each hidden weight
            grad_b_hidden = []  # Will hold gradient for each hidden bias
            grad_w_out = []     # Will hold gradient for each output weight
            
            # --- Compute gradients for hidden layer ---
            for i in range(self.hidden_size):
                
                # Gradient for w_hidden[i]:
                # "If I increase this weight slightly, how much does loss change?"
                original = self.w_hidden[i]      # Save original value
                self.w_hidden[i] = original + dx  # Nudge weight up
                nudged_loss = self.loss(X, Y)     # Compute new loss
                grad = (nudged_loss - current_loss) / dx  # Rate of change
                self.w_hidden[i] = original       # IMPORTANT: Reset to original
                grad_w_hidden.append(grad)        # Store gradient for later
                
                # Gradient for b_hidden[i]:
                # Same process for the bias
                original = self.b_hidden[i]
                self.b_hidden[i] = original + dx
                nudged_loss = self.loss(X, Y)
                grad = (nudged_loss - current_loss) / dx
                self.b_hidden[i] = original       # Reset to original
                grad_b_hidden.append(grad)
            
            # --- Compute gradients for output layer ---
            for i in range(self.hidden_size):
                # Gradient for w_out[i]
                original = self.w_out[i]
                self.w_out[i] = original + dx
                nudged_loss = self.loss(X, Y)
                grad = (nudged_loss - current_loss) / dx
                self.w_out[i] = original          # Reset to original
                grad_w_out.append(grad)
            
            # Gradient for b_out (single value, not a list)
            original = self.b_out
            self.b_out = original + dx
            nudged_loss = self.loss(X, Y)
            grad_b_out = (nudged_loss - current_loss) / dx
            self.b_out = original                 # Reset to original
            
            # --- Now update all weights using stored gradients ---
            # Update rule: new_weight = old_weight - learning_rate * gradient
            # We subtract because gradient points toward INCREASING loss
            # and we want to DECREASE loss
            for i in range(self.hidden_size):
                self.w_hidden[i] -= lr * grad_w_hidden[i]
                self.b_hidden[i] -= lr * grad_b_hidden[i]
                self.w_out[i] -= lr * grad_w_out[i]
            self.b_out -= lr * grad_b_out
            
            # Print progress every 40 epochs
            if epoch % 40 == 0:
                print(f"Epoch {epoch}: loss={current_loss:.4f}")
    
    def predict(self, x):
        """
        Make a prediction for a single input
        
        x: input value
        returns: predicted output
        """
        return self.forward(x)


if __name__ == "__main__":
    # Our training data: output should be 2 * input
    X = [1, 2, 3, 4]      # Inputs
    Y = [2, 4, 6, 8]      # Expected outputs
    
    # Create network with 2 hidden neurons
    nn = NeuralNetwork(hidden_size=2)
    
    print("Training...")
    # Train for 200 epochs with learning rate 0.01
    nn.train(X, Y, epochs=200, lr=0.01)
    
    print("\nTesting:")
    # Test on training data (1-4) and new unseen data (5)
    for x in [1, 2, 3, 4, 5]:
        pred = nn.predict(x)
        print(f"Input {x} -> Predicted {pred:.2f}, Expected {2*x}")