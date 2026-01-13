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
# STEP 2: DEFINE NETWORK (same as before)
# ============================================================

class MNISTNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

# ============================================================
# STEP 3: TRAIN THE MODEL
# ============================================================

model = MNISTNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training...")
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
    print(f"Epoch {epoch + 1}/10, Loss: {avg_loss:.4f}")

print("Training complete!")

# ============================================================
# STEP 4: SAVE THE MODEL
# ============================================================

# Save the trained weights to a file
torch.save(model.state_dict(), 'mnist_model.pth')
print("\nModel saved to 'mnist_model.pth'")

# ============================================================
# STEP 5: LOAD THE MODEL (simulating a new session)
# ============================================================

# Create a fresh model (no trained weights)
new_model = MNISTNetwork()

# Load the saved weights
new_model.load_state_dict(torch.load('mnist_model.pth'))
print("Model loaded from 'mnist_model.pth'")

# ============================================================
# STEP 6: TEST THE LOADED MODEL
# ============================================================

new_model.eval()  # Evaluation mode

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = new_model(images)
        _, predictions = torch.max(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\nLoaded model accuracy: {accuracy:.2%}")