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

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# ============================================================
# STEP 2: DEFINE CNN
# ============================================================

class MNISTConvNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # First conv layer: 1 input channel, 32 filters, 3×3
        # Input: 28×28×1 → Output: 26×26×32
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        
        # MaxPool: halve the size
        # Input: 26×26×32 → Output: 13×13×32
        self.pool1 = nn.MaxPool2d(2)
        
        # Second conv layer: 32 input channels, 64 filters, 3×3
        # Input: 13×13×32 → Output: 11×11×64
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        
        # MaxPool: halve the size
        # Input: 11×11×64 → Output: 5×5×64
        self.pool2 = nn.MaxPool2d(2)
        
        # Flatten: 5×5×64 = 1600
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(1600, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    
    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        
        # Conv block 1
        x = self.conv1(x)    # [batch, 32, 26, 26]
        x = self.relu1(x)
        x = self.pool1(x)    # [batch, 32, 13, 13]
        
        # Conv block 2
        x = self.conv2(x)    # [batch, 64, 11, 11]
        x = self.relu2(x)
        x = self.pool2(x)    # [batch, 64, 5, 5]
        
        # Flatten and fully connected
        x = self.flatten(x)  # [batch, 1600]
        x = self.fc1(x)      # [batch, 128]
        x = self.relu3(x)
        x = self.fc2(x)      # [batch, 10]
        
        return x

# Create model
model = MNISTConvNet()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model}")
print(f"\nTotal parameters: {total_params}")

# ============================================================
# STEP 3: LOSS AND OPTIMIZER
# ============================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# STEP 4: TRAINING LOOP
# ============================================================

print("\nTraining CNN...")

for epoch in range(10):
    total_loss = 0
    
    for images, labels in train_loader:

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/5, Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "mnist_cnn.pth")

# ============================================================
# STEP 5: TEST ACCURACY
# ============================================================

print("\nTesting...")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predictions = torch.max(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.2%}")