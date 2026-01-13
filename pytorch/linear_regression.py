import torch
import torch.nn as nn

# ============================================================
# STEP 1: DATA — Split into training and test
# ============================================================

# Training data
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Test data — model has never seen these
X_test = torch.tensor([[5.0], [6.0], [7.0]])
Y_test = torch.tensor([[10.0], [12.0], [14.0]])

# ============================================================
# STEP 2: CREATE MODEL
# ============================================================
model = nn.Linear(1, 1)

# ============================================================
# STEP 3: LOSS FUNCTION
# ============================================================
loss_fn = nn.MSELoss()

# ============================================================
# STEP 4: OPTIMIZER
# ============================================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ============================================================
# STEP 5: TRAINING LOOP — train only on training data
# ============================================================
print("Training on X_train...")

for epoch in range(2000):
    predictions = model(X_train)
    loss = loss_fn(predictions, Y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# ============================================================
# RESULTS — test on data model has never seen
# ============================================================
print(f"\nFinal weight: {model.weight.data.item():.4f}")
print(f"Final bias: {model.bias.data.item():.4f}")

print("\n--- Training Data (seen during training) ---")
for i in range(len(X_train)):
    pred = model(X_train[i]).item()
    actual = Y_train[i].item()
    print(f"  x={X_train[i].item():.0f}: predicted={pred:.2f}, actual={actual:.0f}")

print("\n--- Test Data (never seen during training) ---")
for i in range(len(X_test)):
    pred = model(X_test[i]).item()
    actual = Y_test[i].item()
    print(f"  x={X_test[i].item():.0f}: predicted={pred:.2f}, actual={actual:.0f}")