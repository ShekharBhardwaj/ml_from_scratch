import random
import math

class MNISTNetwork:
    
    # ============================================================
    # STEP 0: INITIALIZATION - Create all weights and biases
    # ============================================================
    
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        """
        Create a neural network for MNIST digit classification.
        
        input_size: 784 (28x28 pixels)
        hidden_size: 64 neurons in hidden layer
        output_size: 10 (digits 0-9)
        """
        
        # Store sizes for later use
        self.input_size = input_size    # 784
        self.hidden_size = hidden_size  # 64
        self.output_size = output_size  # 10
        
        # ------------------------------------------------------
        # STEP 2 PREP: Create hidden layer weights and biases
        # 64 neurons, each with 784 weights + 1 bias
        # ------------------------------------------------------
        
        # Hidden layer weights
        # Structure: w_hidden[neuron_index][input_index]
        # Example: w_hidden[0][100] = weight from pixel 100 to neuron 0
        self.w_hidden = []
        for neuron in range(hidden_size):  # 64 neurons
            neuron_weights = []
            for input_pixel in range(input_size):  # 784 inputs
                # Small random weight between -0.1 and 0.1
                weight = random.uniform(-0.1, 0.1)
                neuron_weights.append(weight)
            self.w_hidden.append(neuron_weights)
        # Result: w_hidden is 64 lists of 784 weights each
        
        # Hidden layer biases: one per neuron
        self.b_hidden = []
        for neuron in range(hidden_size):  # 64 biases
            self.b_hidden.append(0.0)
        
        # ------------------------------------------------------
        # STEP 3 PREP: Create output layer weights and biases
        # 10 neurons, each with 64 weights + 1 bias
        # ------------------------------------------------------
        
        # Output layer weights
        # Structure: w_output[output_index][hidden_index]
        # Example: w_output[5][10] = weight from hidden neuron 10 to output 5
        self.w_output = []
        for output_neuron in range(output_size):  # 10 outputs
            neuron_weights = []
            for hidden_neuron in range(hidden_size):  # 64 hidden neurons
                weight = random.uniform(-0.1, 0.1)
                neuron_weights.append(weight)
            self.w_output.append(neuron_weights)
        # Result: w_output is 10 lists of 64 weights each
        
        # Output layer biases: one per output
        self.b_output = []
        for output_neuron in range(output_size):  # 10 biases
            self.b_output.append(0.0)


    # ============================================================
    # STEPS 1-3: FORWARD PASS - Input → Hidden → Output
    # ============================================================
    
    def forward(self, image):
        """
        Pass an image through the network to get predictions.
        
        image: list of 784 pixel values (0 to 1)
        returns: list of 10 confidence values (one per digit)
        """
        
        # ------------------------------------------------------
        # STEP 1: INPUT
        # The image is already a list of 784 numbers
        # Nothing to do here — just use it
        # ------------------------------------------------------
        
        # ------------------------------------------------------
        # STEP 2: HIDDEN LAYER
        # Each of 64 neurons computes:
        #   output = (w0 × pixel0) + (w1 × pixel1) + ... + (w783 × pixel783) + bias
        # Then applies ReLU activation
        # ------------------------------------------------------
        
        self.hidden_outputs = []  # Will store 64 values
        
        for neuron in range(self.hidden_size):  # Loop through 64 neurons
            # Start with bias
            total = self.b_hidden[neuron]
            
            # Add weighted sum of all inputs
            for pixel in range(self.input_size):  # Loop through 784 pixels
                total += self.w_hidden[neuron][pixel] * image[pixel]
            
            # Apply ReLU activation: if negative, make it 0
            if total < 0:
                total = 0
            
            self.hidden_outputs.append(total)
        
        # Now we have 64 hidden outputs
        
        # ------------------------------------------------------
        # STEP 3: OUTPUT LAYER
        # Each of 10 neurons computes:
        #   output = (w0 × hidden0) + (w1 × hidden1) + ... + (w63 × hidden63) + bias
        # No activation here (we'll use softmax later for probabilities)
        # ------------------------------------------------------
        
        self.final_outputs = []  # Will store 10 values
        
        for output_neuron in range(self.output_size):  # Loop through 10 outputs
            # Start with bias
            total = self.b_output[output_neuron]
            
            # Add weighted sum of all hidden outputs
            for hidden in range(self.hidden_size):  # Loop through 64 hidden
                total += self.w_output[output_neuron][hidden] * self.hidden_outputs[hidden]
            
            self.final_outputs.append(total)
        
        # Now we have 10 output values
        return self.final_outputs

    # ============================================================
    # STEP 4a: SOFTMAX - Convert outputs to probabilities
    # ============================================================
    
    def softmax(self, outputs):
        """
        Convert raw outputs to probabilities that sum to 1.
        
        outputs: list of 10 raw values from forward pass
        returns: list of 10 probabilities
        """
        import math
        
        # Step 1: Find the maximum value (for numerical stability)
        # Subtracting max prevents overflow when computing e^x
        max_val = max(outputs)
        
        # Step 2: Compute e^(output - max) for each output
        exp_values = []
        for output in outputs:
            exp_values.append(math.exp(output - max_val))
        
        # Step 3: Sum all the exponentials
        exp_sum = sum(exp_values)
        
        # Step 4: Divide each by the sum to get probabilities
        probabilities = []
        for exp_val in exp_values:
            probabilities.append(exp_val / exp_sum)
        
        return probabilities
    
    # ============================================================
    # STEP 4b: CROSS-ENTROPY LOSS - How wrong is our prediction?
    # ============================================================
    
    def cross_entropy_loss(self, probabilities, actual_label):
        """
        Compute cross-entropy loss for one prediction.
        
        probabilities: list of 10 probabilities from softmax
        actual_label: correct digit (0-9)
        returns: loss value (lower is better)
        """
        import math
        
        # Get probability assigned to the correct answer
        correct_prob = probabilities[actual_label]
        
        # Avoid log(0) which is undefined — use tiny value instead
        if correct_prob < 1e-15:
            correct_prob = 1e-15
        
        # Cross-entropy loss = -log(probability of correct answer)
        # If correct_prob is high (0.9), loss is low (0.1)
        # If correct_prob is low (0.1), loss is high (2.3)
        loss = -math.log(correct_prob)
        
        return loss

    # ============================================================
    # STEP 5a: COMPUTE LOSS FOR ONE IMAGE
    # ============================================================
    
    def compute_loss(self, image, label):
        """
        Complete forward pass and loss for one image.
        
        image: list of 784 pixels
        label: correct digit (0-9)
        returns: loss value
        """
        # Forward pass
        outputs = self.forward(image)
        
        # Convert to probabilities
        probs = self.softmax(outputs)
        
        # Compute loss
        loss = self.cross_entropy_loss(probs, label)
        
        return loss
    
    # ============================================================
    # STEP 5b: COMPUTE GRADIENT FOR ONE WEIGHT
    # ============================================================
    
    def compute_gradient_for_weight(self, image, label, layer, neuron_idx, weight_idx, dx=0.0001):
        """
        Compute gradient for a single weight.
        
        image: input image (784 pixels)
        label: correct digit (0-9)
        layer: 'hidden' or 'output'
        neuron_idx: which neuron (0-63 for hidden, 0-9 for output)
        weight_idx: which weight in that neuron
        dx: tiny nudge amount
        
        returns: gradient value
        """
        # Step 1: Compute current loss
        current_loss = self.compute_loss(image, label)
        
        # Step 2: Nudge the weight up
        if layer == 'hidden':
            original = self.w_hidden[neuron_idx][weight_idx]
            self.w_hidden[neuron_idx][weight_idx] = original + dx
        elif layer == 'output':
            original = self.w_output[neuron_idx][weight_idx]
            self.w_output[neuron_idx][weight_idx] = original + dx
        
        # Step 3: Compute new loss with nudged weight
        new_loss = self.compute_loss(image, label)
        
        # Step 4: Compute gradient
        gradient = (new_loss - current_loss) / dx
        
        # Step 5: Reset weight to original value
        if layer == 'hidden':
            self.w_hidden[neuron_idx][weight_idx] = original
        elif layer == 'output':
            self.w_output[neuron_idx][weight_idx] = original
        
        return gradient

    
    # ============================================================
    # STEP 6: TRAIN ON ONE IMAGE
    # ============================================================
    
    def train_one_image(self, image, label, learning_rate=0.01):
        """
        Update all weights based on one image.
        
        image: input image (784 pixels)
        label: correct digit (0-9)
        learning_rate: how big each update step is
        
        returns: loss before update
        """
        dx = 0.0001
        
        # Compute loss before training
        loss_before = self.compute_loss(image, label)
        
        # --- Update hidden layer weights ---
        for neuron in range(self.hidden_size):
            for weight_idx in range(self.input_size):
                # Compute gradient
                grad = self.compute_gradient_for_weight(
                    image, label, 'hidden', neuron, weight_idx, dx
                )
                # Update weight
                self.w_hidden[neuron][weight_idx] -= learning_rate * grad
        
        # --- Update hidden layer biases ---
        for neuron in range(self.hidden_size):
            # Nudge bias
            original = self.b_hidden[neuron]
            self.b_hidden[neuron] = original + dx
            new_loss = self.compute_loss(image, label)
            grad = (new_loss - loss_before) / dx
            self.b_hidden[neuron] = original - learning_rate * grad
        
        # --- Update output layer weights ---
        for neuron in range(self.output_size):
            for weight_idx in range(self.hidden_size):
                grad = self.compute_gradient_for_weight(
                    image, label, 'output', neuron, weight_idx, dx
                )
                self.w_output[neuron][weight_idx] -= learning_rate * grad
        
        # --- Update output layer biases ---
        for neuron in range(self.output_size):
            original = self.b_output[neuron]
            self.b_output[neuron] = original + dx
            new_loss = self.compute_loss(image, label)
            grad = (new_loss - loss_before) / dx
            self.b_output[neuron] = original - learning_rate * grad
        
        return loss_before


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.mnist_loader import load_mnist
    
    # Create network and load data
    net = MNISTNetwork()
    train_images, train_labels, _, _ = load_mnist()
    
    # Forward pass
    image = train_images[0]
    actual = train_labels[0]
    outputs = net.forward(image)
    
    # Convert to probabilities
    probs = net.softmax(outputs)
    
    # Compute loss
    loss = net.cross_entropy_loss(probs, actual)
    
    print(f"Actual label: {actual}")
    print(f"Probability for correct answer: {probs[actual]:.3f}")
    print(f"Loss: {loss:.3f}")
    
    # What would loss be if we were confident and correct?
    print(f"\n--- For comparison ---")
    print(f"If probability was 0.9: loss = {-math.log(0.9):.3f}")
    print(f"If probability was 0.5: loss = {-math.log(0.5):.3f}")
    print(f"If probability was 0.1: loss = {-math.log(0.1):.3f}")
    
    # Test compute_loss
    print("\n--- Test compute_loss ---")
    image = train_images[0]
    label = train_labels[0]
    
    loss = net.compute_loss(image, label)
    print(f"Image label: {label}")
    print(f"Loss: {loss:.3f}")


    print("--- Test gradient computation ---")
    
    # Compute gradient for first weight in first hidden neuron
    grad = net.compute_gradient_for_weight(image, label, 'hidden', 0, 0)
    print(f"Gradient for w_hidden[0][0]: {grad:.6f}")
    
    # Compute gradient for first weight in first output neuron
    grad = net.compute_gradient_for_weight(image, label, 'output', 0, 0)
    print(f"Gradient for w_output[0][0]: {grad:.6f}")


    print(f"--- Training on one image (label: {label}) ---")
    print("This will take a minute...\n")
    
    # Before training
    loss_before = net.compute_loss(image, label)
    outputs = net.forward(image)
    probs = net.softmax(outputs)
    pred_before = probs.index(max(probs))
    
    print(f"Before: loss={loss_before:.3f}, prediction={pred_before}, prob[{label}]={probs[label]:.3f}")
    
    # Train on this one image
    net.train_one_image(image, label, learning_rate=0.1)
    
    # After training
    loss_after = net.compute_loss(image, label)
    outputs = net.forward(image)
    probs = net.softmax(outputs)
    pred_after = probs.index(max(probs))
    
    print(f"After:  loss={loss_after:.3f}, prediction={pred_after}, prob[{label}]={probs[label]:.3f}")