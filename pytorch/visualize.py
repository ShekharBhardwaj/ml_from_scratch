import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms

# ============================================
# LOAD AND PREPARE DATA
# ============================================

# Get the absolute path to the data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'digit-recognizer')

# Load CSV
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# Extract labels and pixels
y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
X_train = torch.tensor(train_df.drop('label', axis=1).values, dtype=torch.float32)

# Reshape: (42000, 784) → (42000, 1, 28, 28)
X_train = X_train.reshape(-1, 1, 28, 28)

# Normalize: 0-255 → 0-1
X_train = X_train / 255.0

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ============================================
# VISUALIZE ORIGINAL IMAGES
# ============================================

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i in range(5):
    # squeeze() removes channel dimension: (1, 28, 28) → (28, 28)
    img = X_train[i].squeeze()
    label = y_train[i].item()
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.suptitle('Original Training Images')
plt.tight_layout()
plt.savefig('original_images.png')
plt.show()
print("Saved to original_images.png")


# ============================================
# VISUALIZE AUGMENTED IMAGES
# ============================================



# Define augmentation: random rotation between -15 and +15 degrees
augment = transforms.RandomRotation(degrees=15)

# Pick one image to augment
original_img = X_train[0]  # shape: (1, 28, 28)
label = y_train[0].item()

# Create 5 augmented versions of the same image
fig, axes = plt.subplots(1, 6, figsize=(14, 3))

# First show original
axes[0].imshow(original_img.squeeze(), cmap='gray')
axes[0].set_title(f'Original: {label}')
axes[0].axis('off')

# Then show 5 random rotations
for i in range(1, 6):
    augmented_img = augment(original_img)
    axes[i].imshow(augmented_img.squeeze(), cmap='gray')
    axes[i].set_title(f'Rotated: {label}')
    axes[i].axis('off')

plt.suptitle('One Image → Multiple Augmented Versions')
plt.tight_layout()
plt.savefig('augmented_images.png')
plt.show()
print("Saved to augmented_images.png")


# ============================================
# COMPARE DIFFERENT AUGMENTATION TYPES
# ============================================

from torchvision import transforms

# Pick one image
original_img = X_train[3]  # Let's use index 3
label = y_train[3].item()

# Define different augmentations
augmentations = {
    'Original': transforms.Lambda(lambda x: x),  # no change
    'Rotate ±15°': transforms.RandomRotation(degrees=15),
    'Shift': transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    'Zoom': transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    'All combined': transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])
}

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
axes = axes.flatten()

plot_idx = 0
for name, aug in augmentations.items():
    # Show 2 examples of each augmentation type
    for _ in range(2):
        augmented = aug(original_img)
        axes[plot_idx].imshow(augmented.squeeze(), cmap='gray')
        axes[plot_idx].set_title(f'{name}\nLabel: {label}')
        axes[plot_idx].axis('off')
        plot_idx += 1

plt.suptitle('Different Augmentation Types')
plt.tight_layout()
plt.savefig('augmentation_types.png')
plt.show()
print("Saved to augmentation_types.png")