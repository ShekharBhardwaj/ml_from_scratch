# ML Techniques Reference Guide

A comprehensive list of techniques to improve model performance.

---

## 1. DATA TECHNIQUES

| Technique | What it does | Best for |
|-----------|--------------|----------|
| **Data augmentation** | Create variations of training data | Images, audio, text |
| **Collect more data** | More real samples | Everything, if available |
| **Feature engineering** | Create new meaningful columns | Tabular data |
| **Feature scaling** | Normalize inputs to similar range | Everything |
| **Handle missing values** | Fill or remove gaps | Tabular data |
| **Remove outliers** | Drop extreme values | Tabular data |
| **Balance classes** | Fix imbalanced datasets | Classification |

---

## 2. MODEL TECHNIQUES

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **Deeper network** | Add more layers | Underfitting (model too simple) |
| **Wider network** | More neurons per layer | Underfitting |
| **Different architecture** | CNN for images, RNN for sequences | Match data type |
| **Dropout** | Randomly turn off neurons during training | Overfitting |
| **Batch normalization** | Stabilize layer inputs | Training instability |
| **Weight decay (L2)** | Penalize large weights | Overfitting |
| **L1 regularization** | Push weights toward zero | Overfitting, feature selection |

---

## 3. TRAINING TECHNIQUES

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **More epochs** | Train longer | Model still improving |
| **Early stopping** | Stop when validation peaks | Prevent overfitting |
| **Learning rate tuning** | Adjust step size | Training too slow or unstable |
| **Learning rate scheduling** | Decrease LR over time | Fine-tune after initial training |
| **Different optimizer** | Adam, SGD, RMSprop | Training issues |
| **Batch size tuning** | Change samples per step | Memory or convergence issues |

---

## 4. VALIDATION TECHNIQUES

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **Train/val/test split** | Hold out data for evaluation | Always |
| **Cross-validation** | Train on multiple splits, average | Small datasets |
| **Stratified split** | Keep class proportions equal | Imbalanced classification |

---

## 5. ADVANCED TECHNIQUES

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **Ensembles** | Combine multiple models | Maximize performance |
| **Transfer learning** | Use pretrained model | Limited data, common task |
| **Hyperparameter search** | Systematically try combinations | Final optimization |
| **Data cleaning** | Fix label errors | Noisy dataset |

---

## Quick Decision Guide

```
Model underfitting (train acc low)?
  → Bigger model, more features, more epochs

Model overfitting (train high, val low)?
  → Augmentation, dropout, regularization, more data, early stopping

Both train and val low?
  → Check data quality, try different architecture

Both train and val high but test low?
  → You're overfitting to validation — use cross-validation
```

---

## Techniques Learned So Far

- ✅ Feature scaling (Day 14)
- ✅ Train/val/test split (Day 13, Day 26)
- ✅ Data augmentation (Day 27)
- ✅ Early stopping concept (Day 27)

## Techniques Coming Soon

- ⬜ Dropout
- ⬜ Batch normalization
- ⬜ Learning rate scheduling
- ⬜ Cross-validation
- ⬜ Ensembles
- ⬜ Transfer learning

---

## Image Augmentation Examples (PyTorch)

```python
from torchvision import transforms

# Safe augmentations for digit recognition
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),        # ±15 degrees
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1),                     # ±10% shift
        scale=(0.9, 1.1)                          # 90-110% zoom
    )
])

# WARNING: Don't use for digits
# - Horizontal flip (3 becomes mirrored)
# - Vertical flip (everything breaks)
# - Large rotation (6 becomes 9)
```

---

## Custom Dataset Template (for augmentation)

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Usage:
# train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
# val_dataset = CustomDataset(X_val, y_val, transform=None)  # No augmentation!
```

---

## Key Principles

1. **Only augment training data** — validation and test stay clean
2. **Augmentation should preserve meaning** — a rotated 7 is still a 7
3. **Training accuracy may drop with augmentation** — this is expected
4. **Validation accuracy matters more** — it predicts real-world performance
5. **Stop training when validation peaks** — not when epochs run out
