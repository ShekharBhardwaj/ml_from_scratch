import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the model architecture (must match the saved model)
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

# Create model instance and load the saved weights
model = MNISTNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Set to evaluation mode

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root='./data/pytorch_mnist',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

