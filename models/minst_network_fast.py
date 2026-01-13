import random
import math

class MNISTNetworkFast:
    
    # ============================================================
    # STEP 0: INITIALIZATION - Smaller network for faster training
    # ============================================================
    
    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        """
        Smaller network: 784 -> 16 -> 10
        
        Total weights: 784*16 + 16*10 = 12,544 + 160 = 12,704
        Much faster than 50,890!
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Hidden layer: 16 neurons × 784 weights
        self.w_hidden = []
        for neuron in range(hidden_size):
            weights = [random.uniform(-0.1, 0.1) for _ in range(input_size)]
            self.w_hidden.append(weights)
        self.b_hidden = [0.0] * hidden_size
        
        # Output layer: 10 neurons × 16 weights
        self.w_output = []
        for neuron in range(output_size):
            weights = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            self.w_output.append(weights)
        self.b_output = [0.0] * output_size
    
    # ============================================================
    # STEPS 1-3: FORWARD PASS
    # ============================================================
    
    def forward(self, image):
        """Input -> Hidden -> Output"""
        # Hidden layer with ReLU
        self.hidden = []
        for neuron in range(self.hidden_size):
            total = self.b_hidden[neuron]
            for i in range(self.input_size):
                total += self.w_hidden[neuron][i] * image[i]
            # ReLU
            if total < 0:
                total = 0
            self.hidden.append(total)
        
        # Output layer (no activation)
        self.output = []
        for neuron in range(self.output_size):
            total = self.b_output[neuron]
            for i in range(self.hidden_size):
                total += self.w_output[neuron][i] * self.hidden[i]
            self.output.append(total)
        
        return self.output
    
    # ============================================================
    # STEP 4: SOFTMAX AND LOSS
    # ============================================================
    
    def softmax(self, outputs):
        """Convert to probabilities"""
        max_val = max(outputs)
        exp_vals = [math.exp(o - max_val) for o in outputs]
        exp_sum = sum(exp_vals)
        return [e / exp_sum for e in exp_vals]
    
    def compute_loss(self, image, label):
        """Forward pass + loss for one image"""
        outputs = self.forward(image)
        probs = self.softmax(outputs)
        # Cross-entropy
        correct_prob = max(probs[label], 1e-15)
        return -math.log(correct_prob)
    
    # ============================================================
    # STEP 5-6: TRAIN WITH NUMERICAL GRADIENTS
    # ============================================================
    
    def train_batch(self, images, labels, learning_rate=0.1):
        """
        Train on a batch of images.
        
        Computes average loss across batch, then updates weights.
        """
        dx = 0.001  # Larger dx for speed
        batch_size = len(images)
        
        # Compute average loss for the batch
        def batch_loss():
            total = 0
            for i in range(batch_size):
                total += self.compute_loss(images[i], labels[i])
            return total / batch_size
        
        current_loss = batch_loss()
        
        # Update hidden weights
        for neuron in range(self.hidden_size):
            for w in range(self.input_size):
                original = self.w_hidden[neuron][w]
                self.w_hidden[neuron][w] = original + dx
                new_loss = batch_loss()
                grad = (new_loss - current_loss) / dx
                self.w_hidden[neuron][w] = original - learning_rate * grad
        
        # Update hidden biases
        for neuron in range(self.hidden_size):
            original = self.b_hidden[neuron]
            self.b_hidden[neuron] = original + dx
            new_loss = batch_loss()
            grad = (new_loss - current_loss) / dx
            self.b_hidden[neuron] = original - learning_rate * grad
        
        # Update output weights
        for neuron in range(self.output_size):
            for w in range(self.hidden_size):
                original = self.w_output[neuron][w]
                self.w_output[neuron][w] = original + dx
                new_loss = batch_loss()
                grad = (new_loss - current_loss) / dx
                self.w_output[neuron][w] = original - learning_rate * grad
        
        # Update output biases
        for neuron in range(self.output_size):
            original = self.b_output[neuron]
            self.b_output[neuron] = original + dx
            new_loss = batch_loss()
            grad = (new_loss - current_loss) / dx
            self.b_output[neuron] = original - learning_rate * grad
        
        return current_loss
    
    # ============================================================
    # PREDICTION AND ACCURACY
    # ============================================================
    
    def predict(self, image):
        """Return predicted digit (0-9)"""
        outputs = self.forward(image)
        probs = self.softmax(outputs)
        return probs.index(max(probs))
    
    def accuracy(self, images, labels):
        """Compute accuracy on a set of images"""
        correct = 0
        for i in range(len(images)):
            if self.predict(images[i]) == labels[i]:
                correct += 1
        return correct / len(images)



if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.mnist_loader import load_mnist
    
    print("Loading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Use small subset for speed
    train_small = train_images[:100]
    labels_small = train_labels[:100]
    test_small = test_images[:50]
    test_labels_small = test_labels[:50]
    
    print(f"Training on {len(train_small)} images")
    print(f"Testing on {len(test_small)} images")
    
    # Create network
    net = MNISTNetworkFast(hidden_size=16)
    
    # Check accuracy before training
    acc_before = net.accuracy(test_small, test_labels_small)
    print(f"\nBefore training: {acc_before:.1%} accuracy")
    
    # Train for a few epochs
    print("\nTraining (this takes a few minutes)...")
    for epoch in range(3):
        loss = net.train_batch(train_small, labels_small, learning_rate=0.5)
        acc = net.accuracy(test_small, test_labels_small)
        print(f"Epoch {epoch}: loss={loss:.3f}, test accuracy={acc:.1%}")
    
    print("\nDone!")