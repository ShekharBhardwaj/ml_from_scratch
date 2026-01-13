import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# STEP 1: LOAD MNIST DATA
# ============================================================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data/pytorch_mnist',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data/pytorch_mnist',
    train=False,
    download=True,
    transform=transform
)

# ============================================================
# STEP 2: CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# ============================================================
# STEP 3: CREATE NEURAL NETWORK
# ============================================================

class MNISTNetwork(nn.Module):
    
    def __init__(self):
        # Initialize parent class (nn.Module)
        super().__init__()
        
        # Flatten layer: [1, 28, 28] → [784]
        self.flatten = nn.Flatten()
        
       # Layer 1: 784 inputs → 128 outputs
        self.hidden1 = nn.Linear(784, 128)

        # Layer 2: 128 inputs → 64 outputs
        self.hidden2 = nn.Linear(128, 64)

        
        # Activation function
        self.relu = nn.ReLU()
        
        # Output layer: 64 inputs → 10 outputs (one per digit)
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        # x starts as shape [batch_size, 1, 28, 28]
        
        # Step 1: Flatten to [batch_size, 784]
        x = self.flatten(x)
        
        # Step 2: Hidden layer + ReLU
        x = self.hidden1(x)    # [batch_size, 128]
        x = self.relu(x)      # Still [batch_size, 128]
        x = self.hidden2(x)    # [batch_size, 64]
        x = self.relu(x)      # Still [batch_size, 64]
        
        # Step 3: Output layer
        x = self.output(x)    # [batch_size, 10]
        
        return x

# Create the network
model = MNISTNetwork()

# Print model structure
print("Model structure:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# ============================================================
# STEP 4: LOSS FUNCTION AND OPTIMIZER
# ============================================================

# Cross-entropy loss for classification (softmax built-in)
loss_fn = nn.CrossEntropyLoss()

# Adam optimizer — smarter than basic SGD
# lr = learning rate (how big each step is)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Loss function:", loss_fn)
print("Optimizer:", optimizer)

# Quick test: run one batch through
images, labels = next(iter(train_loader))  # Get one batch
outputs = model(images)                      # Forward pass
loss = loss_fn(outputs, labels)              # Compute loss

print(f"\nTest batch:")
print(f"  Images shape: {images.shape}")
print(f"  Outputs shape: {outputs.shape}")
print(f"  Loss: {loss.item():.4f}")



# ============================================================
# STEP 5: TRAINING LOOP
# ============================================================

num_epochs = 50  # Go through entire dataset 5 times

print("Training started...")

for epoch in range(num_epochs):
    total_loss = 0  # Track loss for this epoch
    
    # Loop through all batches
    for images, labels in train_loader:
        
        # Forward pass: compute predictions
        outputs = model(images)
        
        # Compute loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        
        # Update weights
        optimizer.step()
        
        # Track total loss
        total_loss += loss.item()
    
    # Average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

print("\nTraining complete!")

# ============================================================
# STEP 6: TEST ACCURACY
# ============================================================

print("\nTesting...")

model.eval()  # Put model in evaluation mode

correct = 0   # Count correct predictions
total = 0     # Count total predictions

with torch.no_grad():  # Don't compute gradients (faster)
    
    for images, labels in test_loader:
        
        # Forward pass
        outputs = model(images)
        
        # Get predictions: index of highest value for each image
        # outputs shape: [32, 10]
        # predictions shape: [32] — one prediction per image
        _, predictions = torch.max(outputs, dim=1)
        
        # Count correct predictions
        # (predictions == labels) gives True/False for each
        # .sum() counts the Trues
        correct += (predictions == labels).sum().item()
        total += labels.size(0)  # Batch size

accuracy = correct / total
print(f"Test accuracy: {accuracy:.2%}")
print(f"Correct: {correct} / {total}")