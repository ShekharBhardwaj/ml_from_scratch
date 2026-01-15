import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split

# Get the absolute path to the data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'digit-recognizer')

# Load CSVs
# Load the data
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Extract labels and pixels as PyTorch tensors
y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
X_train = torch.tensor(train_df.drop('label', axis=1).values, dtype=torch.float32)
X_test = torch.tensor(test_df.values, dtype=torch.float32)

print("="*50)
print("Before reshape:")
print("="*50)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

print("="*50)
print("Reshaping the data...")
print("="*50)

X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

print("X_train shape after reshape:", X_train.shape)
print("X_test shape after reshape:", X_test.shape)

print("="*50)
print("Checking pixel values:")
print("="*50)

print("Min pixel value:", X_train.min())
print("Max pixel value:", X_train.max())

print("="*50)
print("Normalizing the data...")
print("="*50)

X_train = X_train / 255.0
X_test = X_test / 255.0

print("="*50)
print("Checking pixel values after normalization:")
print("="*50)

print("Min pixel value:", X_train.min())
print("Max pixel value:", X_train.max())

print("="*50)


 # Combine features and labels into a Dataset
print("="*50)
print("Combining features and labels into a Dataset...")
print("="*50)

full_dataset = TensorDataset(X_train, y_train)

# Split: 80% train, 20% validation
print("="*50)
print("Splitting the data into training and validation sets...")
print("="*50)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ============================================
        # FIRST CONVOLUTIONAL BLOCK
        # ============================================
        
        # Conv2d: applies 32 different 3x3 filters to the input image
        # - in_channels=1: input is grayscale (1 channel)
        # - out_channels=32: we learn 32 different filters (edge detectors, curve detectors, etc.)
        # - kernel_size=3: each filter is 3x3 pixels
        # - padding=1: add 1-pixel border of zeros so output height/width stays same
        # Shape: (batch, 1, 28, 28) → (batch, 32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # MaxPool2d: shrinks the image by taking the max value in each 2x2 region
        # - kernel_size=2: look at 2x2 regions
        # - stride=2: move 2 pixels between regions (no overlap)
        # This halves height and width, reducing computation and adding translation invariance
        # Shape: (batch, 32, 28, 28) → (batch, 32, 14, 14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ============================================
        # SECOND CONVOLUTIONAL BLOCK
        # ============================================
        
        # Conv2d: applies 64 different 3x3 filters
        # - in_channels=32: input now has 32 channels (from conv1)
        # - out_channels=64: we learn 64 filters that combine the 32 input feature maps
        # These filters detect higher-level patterns (combinations of edges → shapes)
        # Shape: (batch, 32, 14, 14) → (batch, 64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # MaxPool2d: shrink spatial dimensions again
        # Shape: (batch, 64, 14, 14) → (batch, 64, 7, 7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ============================================
        # FULLY CONNECTED (LINEAR) LAYERS
        # ============================================
        
        # Linear: standard neural network layer (like Day 23)
        # - Input: 64 * 7 * 7 = 3136 (flattened output from conv layers)
        # - Output: 128 (we chose this - it's a hyperparameter)
        # This layer learns to combine all the detected features into a decision
        # Shape: (batch, 3136) → (batch, 128)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # Final Linear layer: maps to 10 output classes (digits 0-9)
        # Shape: (batch, 128) → (batch, 10)
        self.fc2 = nn.Linear(128, 10)
        
        # ReLU activation: introduces non-linearity
        # Formula: relu(x) = max(0, x)
        # Without this, stacking linear layers would just be one big linear layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass: defines how data flows through the network.
        Input x has shape (batch, 1, 28, 28)
        """
        
        # ============================================
        # CONV BLOCK 1: detect low-level features (edges, simple patterns)
        # ============================================
        x = self.conv1(x)      # (batch, 1, 28, 28) → (batch, 32, 28, 28)
        x = self.relu(x)       # apply non-linearity (shape unchanged)
        x = self.pool1(x)      # (batch, 32, 28, 28) → (batch, 32, 14, 14)
        
        # ============================================
        # CONV BLOCK 2: detect higher-level features (shapes, parts of digits)
        # ============================================
        x = self.conv2(x)      # (batch, 32, 14, 14) → (batch, 64, 14, 14)
        x = self.relu(x)       # apply non-linearity (shape unchanged)
        x = self.pool2(x)      # (batch, 64, 14, 14) → (batch, 64, 7, 7)
        
        # ============================================
        # FLATTEN: convert 3D feature maps to 1D vector
        # ============================================
        # x.view(-1, 64 * 7 * 7) reshapes the tensor
        # -1 means "figure out this dimension automatically" (it will be batch size)
        # 64 * 7 * 7 = 3136 numbers per image
        x = x.view(-1, 64 * 7 * 7)  # (batch, 64, 7, 7) → (batch, 3136)
        
        # ============================================
        # FULLY CONNECTED LAYERS: make final classification decision
        # ============================================
        x = self.fc1(x)        # (batch, 3136) → (batch, 128)
        x = self.relu(x)       # apply non-linearity
        x = self.fc2(x)        # (batch, 128) → (batch, 10)
        
        # Output: 10 raw scores (logits) for each image
        # Higher score = model thinks that digit is more likely
        # We don't apply softmax here because CrossEntropyLoss does it internally
        return x

# Create an instance of the model
model = CNN()
print(model)



# ============================================
# SETUP
# ============================================

# Loss function: CrossEntropyLoss
# - Combines softmax + negative log likelihood
# - Perfect for multi-class classification (10 digits)
# Math: loss = -log(probability of correct class)
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam
# - Adjusts weights to minimize loss
# - lr=0.001 is a common starting point
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of times to loop through entire training set
epochs = 5

# ============================================
# TRAINING LOOP
# ============================================

for epoch in range(epochs):
    
    # ---------- TRAINING PHASE ----------
    model.train()  # Set model to training mode
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        
        # Forward pass: compute predictions
        outputs = model(X_batch)
        
        # Compute loss: how wrong are we?
        loss = criterion(outputs, y_batch)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights
        
        # Track statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class (0-9)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    
    # ---------- VALIDATION PHASE ----------
    model.eval()  # Set model to evaluation mode
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():  # Don't compute gradients (faster, saves memory)
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    
    # ---------- PRINT PROGRESS ----------
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"  Val Acc: {val_accuracy:.2f}%")
    print()


# ============================================
# PREDICT ON KAGGLE'S TEST SET
# ============================================

model.eval()  # Set to evaluation mode (no dropout, etc.)

# X_test is the 28,000 images from Kaggle's test.csv
# We already reshaped and normalized it earlier
print("X_test shape:", X_test.shape)

with torch.no_grad():  # No gradients needed for prediction
    # Forward pass: get raw scores for all test images
    outputs = model(X_test)
    
    # Convert scores to predicted class (0-9)
    # torch.max returns (max_values, indices) — we want the indices
    _, predictions = torch.max(outputs, 1)

print("Predictions shape:", predictions.shape)
print("First 10 predictions:", predictions[:10].tolist())


# ============================================
# CREATE SUBMISSION FILE
# ============================================



# Create ImageId: 1 to 28000 (not 0 to 27999)
image_ids = list(range(1, 28001))

# Get predictions as a Python list
labels = predictions.tolist()

# Create DataFrame with the two columns Kaggle expects
submission = pd.DataFrame({
    'ImageId': image_ids,
    'Label': labels
})

# Check it looks right
print("Submission shape:", submission.shape)
print("\nFirst 10 rows:")
print(submission.head(10))

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("\nSaved to submission.csv")