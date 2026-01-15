import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms

# ============================================
# STEP 1: LOAD DATA
# ============================================

# Get the absolute path to the data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'digit-recognizer')

# Load CSVs
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Extract labels and pixels
y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
X_train = torch.tensor(train_df.drop('label', axis=1).values, dtype=torch.float32)
X_test = torch.tensor(test_df.values, dtype=torch.float32)

# Reshape: (N, 784) → (N, 1, 28, 28)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# Normalize: 0-255 → 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Data loaded:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")

# Expected output:
# Data loaded:
#   X_train: torch.Size([42000, 1, 28, 28])
#   X_test: torch.Size([28000, 1, 28, 28])


# ============================================
# STEP 2: CUSTOM DATASET WITH AUGMENTATION
# ============================================

from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    """
    Custom dataset that can apply augmentation.
    
    Why custom? TensorDataset doesn't support transforms.
    This class lets us augment training data while keeping validation clean.
    """
    
    def __init__(self, images, labels, transform=None):
        self.images = images      # shape: (N, 1, 28, 28)
        self.labels = labels      # shape: (N,)
        self.transform = transform
    
    def __len__(self):
        # How many samples in dataset?
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get one sample by index
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply augmentation if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define augmentation for training
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
])

# No augmentation for validation
val_transform = None

print("Custom dataset class created.")
print("Training augmentation: rotation ±15°, shift ±10%, zoom 90-110%")



# ============================================
# STEP 3: SPLIT DATA AND CREATE DATALOADERS
# ============================================

# First, split indices (not data itself)
total_samples = len(y_train)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size

# Randomly shuffle indices
indices = torch.randperm(total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Split the actual data
X_train_split = X_train[train_indices]
y_train_split = y_train[train_indices]
X_val_split = X_train[val_indices]
y_val_split = y_train[val_indices]

# Create datasets WITH and WITHOUT augmentation
train_dataset = MNISTDataset(X_train_split, y_train_split, transform=train_transform)
val_dataset = MNISTDataset(X_val_split, y_val_split, transform=val_transform)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# ============================================
# STEP 4: BUILD CNN MODEL (same as Day 26)
# ============================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv block 1: (batch, 1, 28, 28) → (batch, 32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2: (batch, 32, 14, 14) → (batch, 64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected: (batch, 3136) → (batch, 10)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model created.")
print(model)



# ============================================
# STEP 5: TRAINING LOOP
# ============================================

epochs = 10  # More epochs than Day 26 (was 5)

for epoch in range(epochs):
    
    # ---------- TRAINING PHASE ----------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    
    # ---------- VALIDATION PHASE ----------
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
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
# STEP 6: PREDICT ON TEST SET
# ============================================

model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predictions = torch.max(outputs, 1)

print(f"Predictions shape: {predictions.shape}")
print(f"First 10 predictions: {predictions[:10].tolist()}")

# ============================================
# STEP 7: CREATE SUBMISSION FILE
# ============================================

submission = pd.DataFrame({
    'ImageId': range(1, 28001),
    'Label': predictions.tolist()
})

# Save to data directory or current directory
output_path = os.path.join(script_dir, 'submission_augmented.csv')
submission.to_csv(output_path, index=False)

print(f"\nSubmission shape: {submission.shape}")
print(f"Saved to: {output_path}")
print("\nFirst 10 rows:")
print(submission.head(10))